import asyncio
import os
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from create_chunks_json import crawl_and_chunk, SITEMAP_PATH, BASE_DOMAIN
import torch
import gc
import logging
import time
import json
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment setup
load_dotenv()
os.environ["SPACY_WARNING_IGNORE"] = "W008"
os.environ["NLTK_DATA"] = "/tmp/nlp_data/nltk"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
FAISS_DIR = ".faiss_db"
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
METADATA_PATH = os.path.join(FAISS_DIR, "metadata.pkl")
CHUNK_JSON_PATH = "data/chunks_memory.json"
EMBEDDINGS_CACHE_PATH = os.path.join(FAISS_DIR, "embeddings_cache.npy")

# Memory management
MAX_CACHE_SIZE = 1000
CLEANUP_INTERVAL = 10
GPU_CACHE_THRESHOLD = 80

class VectorDatabaseManager:
    def __init__(self):
        self._embedding_model = None
        self.index = None
        self.metadata_db = []
        self.query_count = 0
        self.last_cleanup = time.time()
        self._cached_embeddings = {}
        
        # Load existing database if available
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
            try:
                self.index = faiss.read_index(FAISS_INDEX_PATH)
                with open(METADATA_PATH, 'rb') as f:
                    self.metadata_db = pickle.load(f)
                logger.info("Loaded existing FAISS database")
            except Exception as e:
                logger.error(f"Error loading database: {str(e)}")
                self._reset_database()
        else:
            self._reset_database()
            os.makedirs(FAISS_DIR, exist_ok=True)

    def _reset_database(self):
        self.index = None
        self.metadata_db = []

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            # Check if we have cached embeddings
            if os.path.exists(EMBEDDINGS_CACHE_PATH):
                logger.info("Found precomputed embeddings cache")
                return None
                
            # Device selection with memory awareness
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                free_mem = torch.cuda.mem_get_info()[0] / (1024 ** 3)
                
                if free_mem < 2 or total_mem < 8:  # Less than 2GB free or small GPU
                    logger.warning(f"Low GPU memory ({free_mem:.1f}GB free of {total_mem:.1f}GB), using CPU")
                    device = 'cpu'
                else:
                    device = 'cuda'
                    logger.info(f"Using GPU with {free_mem:.1f}GB free memory")
            else:
                device = 'cpu'
                logger.info("Using CPU for embeddings")
            
            # Only load model if we need to compute embeddings
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
            self._embedding_model.eval()
            
            if device == 'cuda':
                try:
                    self._embedding_model.half()
                    logger.info("Using half-precision model for GPU efficiency")
                except Exception:
                    logger.warning("Couldn't convert to half precision, using full precision")
        
        return self._embedding_model

    def _format_chunk(self, chunk: dict) -> tuple:
        context_str = " | ".join(chunk.get("headers_context", []))
        text_content = f"{context_str}\n\n{chunk.get('title', '')}\n{chunk.get('content', '')}"
        
        metadata = {
            "source": chunk.get("source", ""),
            "content_type": chunk.get("content_type", "text"),
            "title": chunk.get("title", ""),
            "headers_context": chunk.get("headers_context", []),
            "keywords": chunk.get("keywords", []),
            "entities": chunk.get("entities", []),
            "chunk_index": chunk.get("chunk_index", 0)
        }

        return chunk["chunk_id"], text_content, metadata

    async def process_and_store_chunks(self):
        try:
            # Load chunks from JSON
            if os.path.exists(CHUNK_JSON_PATH):
                with open(CHUNK_JSON_PATH, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                logger.info(f"Loaded {len(chunks)} chunks from {CHUNK_JSON_PATH}")
            else:
                logger.info("No chunk JSON found, generating chunks...")
                chunks = await crawl_and_chunk(SITEMAP_PATH, BASE_DOMAIN)
                logger.info(f"Generated {len(chunks)} chunks")
            
            # Prepare data for storage
            ids, documents, metadatas = [], [], []
            for chunk in chunks:
                chunk_id, text_content, metadata = self._format_chunk(chunk)
                ids.append(chunk_id)
                documents.append(text_content)
                metadatas.append(metadata)
            
            # Check for embeddings cache
            if os.path.exists(EMBEDDINGS_CACHE_PATH):
                logger.info("Loading embeddings from cache")
                embeddings = np.load(EMBEDDINGS_CACHE_PATH)
            else:
                # Generate embeddings only if needed
                model = self.embedding_model
                if model:
                    embeddings = self._generate_embeddings_efficiently(documents)
                    # Cache embeddings for future runs
                    np.save(EMBEDDINGS_CACHE_PATH, embeddings)
                    logger.info(f"Saved embeddings cache to {EMBEDDINGS_CACHE_PATH}")
                else:
                    logger.error("No embedding model available and no cache found")
                    return
            
            # Create or update FAISS index
            if self.index is None:
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                self.index.add(embeddings)
                self.metadata_db = [
                    {"id": i, "document": doc, "metadata": meta} 
                    for i, doc, meta in zip(ids, documents, metadatas)
                ]
                logger.info(f"Created new FAISS index with {len(ids)} vectors")
            else:
                self.index.add(embeddings)
                for i, doc, meta in zip(ids, documents, metadatas):
                    self.metadata_db.append({
                        "id": i, 
                        "document": doc, 
                        "metadata": meta
                    })
                logger.info(f"Added {len(ids)} new vectors to existing index")
            
            # Persist to disk
            self._save_to_disk()
            
            # Clean up
            del embeddings, chunks
            self._clear_memory()
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise
        finally:
            self._clear_memory()

    def _generate_embeddings_efficiently(self, documents: list) -> np.ndarray:
        """Optimized embedding generation with batching and memory management"""
        # Determine optimal batch size based on hardware
        if torch.cuda.is_available():
            batch_size = 128  # Larger batches for GPU
            precision = torch.float16
            logger.info("Using GPU-optimized embedding generation")
        else:
            # Optimize for CPU with parallel processing
            batch_size = 64
            precision = torch.float32
            torch.set_num_threads(os.cpu_count() or 4)
            logger.info(f"Using CPU with {torch.get_num_threads()} threads")
        
        all_embeddings = []
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        # Use progress bar for visibility
        pbar = tqdm(total=len(documents), desc="Generating embeddings", unit="doc")
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            # Generate embeddings with automatic mixed precision
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(precision==torch.float16)):
                batch_embeddings = self._embedding_model.encode(
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    batch_size=batch_size,
                    show_progress_bar=False
                )
            
            all_embeddings.append(batch_embeddings)
            pbar.update(len(batch))
            
            # Memory management
            if i > 0 and i % (10 * batch_size) == 0:
                self._clear_memory()
        
        pbar.close()
        return np.vstack(all_embeddings)

    def _save_to_disk(self):
        try:
            faiss.write_index(self.index, FAISS_INDEX_PATH)
            with open(METADATA_PATH, 'wb') as f:
                pickle.dump(self.metadata_db, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Database saved to {FAISS_DIR}")
        except Exception as e:
            logger.error(f"Error saving database: {str(e)}")

    def query(self, question: str, top_k: int = 5) -> list:
        try:
            # Query management
            self.query_count += 1
            if self.query_count >= CLEANUP_INTERVAL:
                self.clear_cache()
                self.query_count = 0
            
            # Use cached embeddings if available
            if question in self._cached_embeddings:
                query_embedding = self._cached_embeddings[question]
            else:
                # Use a smaller model for query encoding if available
                if self._embedding_model:
                    with torch.no_grad():
                        query_embedding = self._embedding_model.encode(
                            [question], 
                            convert_to_numpy=True,
                            normalize_embeddings=True
                        )
                else:
                    # Fallback to querying with a smaller model
                    query_model = SentenceTransformer("intfloat/multilingual-e5-small")
                    query_embedding = query_model.encode([question], convert_to_numpy=True)
                
                if len(self._cached_embeddings) < MAX_CACHE_SIZE:
                    self._cached_embeddings[question] = query_embedding
            
            # Search FAISS index
            scores, indices = self.index.search(query_embedding, top_k)
            
            # Format results
            results = []
            for i, score in zip(indices[0], scores[0]):
                if i < 0:
                    continue
                try:
                    item = self.metadata_db[i]
                    context_str = " → ".join(item["metadata"].get("headers_context", []))
                    content_preview = (
                        f"[Context: {context_str}]\n"
                        f"{item['document'][:500]}..."
                    )
                    
                    results.append({
                        "score": float(score),
                        "title": item["metadata"].get("title", ""),
                        "source": item["metadata"].get("source", ""),
                        "content_type": item["metadata"].get("content_type", "text"),
                        "content": content_preview,
                        "full_metadata": item["metadata"]
                    })
                except IndexError:
                    logger.warning(f"Invalid index {i} in metadata database")
            
            return results
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return []
        finally:
            self._clear_memory()

    def clear_cache(self, full: bool = False):
        logger.info("Clearing vector store cache")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._cached_embeddings.clear()
        
        if full and self._embedding_model:
            try:
                self._embedding_model = self._embedding_model.to('cpu')
            except Exception as e:
                logger.warning(f"Error moving model to CPU: {str(e)}")
        
        gc.collect()

    def _clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

async def main():
    db_manager = VectorDatabaseManager()
    await db_manager.process_and_store_chunks()

    # Example query
    results = db_manager.query(
        "Quali sono le differenze tra OGD e OGT?",
        top_k=3
    )

    print("\nTop results:")
    for i, result in enumerate(results, 1):
        print(f"\n--- RESULT {i} ---")
        print(f"Title: {result.get('title')}")
        print(f"Source: {result.get('source')}")
        print(f"Content type: {result.get('content_type')}")
        print(f"Relevance: {result.get('score'):.2%}")
        print(f"Content preview:\n{result.get('content')}")

if __name__ == "__main__":
    asyncio.run(main())