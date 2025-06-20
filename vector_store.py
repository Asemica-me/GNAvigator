import asyncio
import os
import shutil
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["SPACY_WARNING_IGNORE"] = "W008"
os.environ["NLTK_DATA"] = "/tmp/nlp_data/nltk"

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
FAISS_DIR = ".faiss_db"
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
METADATA_PATH = os.path.join(FAISS_DIR, "metadata.pkl")
CHUNK_FILE = os.path.join("data", "chunks_memory.json")

# Memory management constants
MAX_CACHE_SIZE = 1000  # Max cached embeddings
CLEANUP_INTERVAL = 10  # Cleanup every 10 queries
GPU_CACHE_THRESHOLD = 80  # GPU memory usage threshold for offloading

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
        """Initialize empty database state"""
        self.index = None
        self.metadata_db = []

    @property
    def embedding_model(self):
        """Lazy loading of embedding model with memory management"""
        if self._embedding_model is None:
            # Check GPU memory before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                free_mem = torch.cuda.mem_get_info()[0] / (1024 ** 3)
                if free_mem < 2:  # Less than 2GB free
                    logger.warning(f"Low GPU memory ({free_mem:.1f}GB free), using CPU")
                    device = 'cpu'
                else:
                    device = 'cuda'
            else:
                device = 'cpu'
            
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
            self._embedding_model.eval()  # Disable dropout layers
            
            # Reduce memory footprint
            if device == 'cuda':
                self._embedding_model.half()  # Use half precision
                logger.info("Using half-precision model for GPU memory efficiency")
        
        return self._embedding_model

    def _format_chunk(self, chunk: dict) -> tuple:
        """Helper method to format chunk data for storage"""
        # Enhanced text content with headers context
        context_str = " | ".join(chunk.get("headers_context", []))
        text_content = f"{context_str}\n\n{chunk.get('title', '')}\n{chunk.get('content', '')}"
        
        # Include keywords and entities in metadata
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
        """Main processing pipeline with memory management"""
        try:
            # Load chunks from JSON file if available
            if os.path.exists(CHUNK_FILE):
                with open(CHUNK_FILE, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                logger.info(f"Loaded {len(chunks)} chunks from {CHUNK_FILE}")
            else:
                # Generate chunks if JSON file doesn't exist
                logger.info("No chunk JSON found, generating chunks...")
                chunks = await crawl_and_chunk(SITEMAP_PATH, BASE_DOMAIN)
                logger.info(f"Generated {len(chunks)} chunks")
            
            # Process chunks for storage
            ids, documents, metadatas = [], [], []
            for chunk in chunks:
                chunk_id, text_content, metadata = self._format_chunk(chunk)
                ids.append(chunk_id)
                documents.append(text_content)
                metadatas.append(metadata)
            
            # Generate embeddings in batches with memory management
            embeddings = self._generate_embeddings_with_memory_management(documents)
            
            # Create or update FAISS index
            if self.index is None:
                # Initialize new index
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                self.index.add(embeddings)
                
                # Create metadata database
                self.metadata_db = [
                    {"id": i, "document": doc, "metadata": meta} 
                    for i, doc, meta in zip(ids, documents, metadatas)
                ]
                logger.info(f"Created new FAISS index with {len(ids)} vectors")
            else:
                # Add to existing index (not efficient for large updates)
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
            logger.info(f"FAISS database stored in: {os.path.abspath(FAISS_DIR)}")
            
            # Clean up memory
            del embeddings, chunks
            self._clear_memory()
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise
        finally:
            # Ensure resources are released
            self._clear_memory()

    def _generate_embeddings_with_memory_management(self, documents: list) -> np.ndarray:
        """Generate embeddings with memory constraints"""
        batch_size = 32
        all_embeddings = []
        
        # Use half precision if on GPU
        precision = torch.float16 if 'cuda' in str(self.embedding_model.device) else torch.float32
        
        logger.info(f"Generating embeddings for {len(documents)} documents (batch size: {batch_size})")
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            # Generate embeddings
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(precision==torch.float16)):
                batch_embeddings = self.embedding_model.encode(
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    batch_size=batch_size,
                    show_progress_bar=False
                )
            
            all_embeddings.append(batch_embeddings)
            
            # Clean up periodically
            if (i // batch_size) % 10 == 0:
                self._clear_memory()
                logger.debug(f"Processed {i+len(batch)}/{len(documents)} embeddings")
        
        # Concatenate and return
        return np.vstack(all_embeddings)

    def _save_to_disk(self):
        """Save index and metadata to disk"""
        try:
            faiss.write_index(self.index, FAISS_INDEX_PATH)
            with open(METADATA_PATH, 'wb') as f:
                pickle.dump(self.metadata_db, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Database saved to {FAISS_DIR}")
        except Exception as e:
            logger.error(f"Error saving database: {str(e)}")

    def query(self, question: str, top_k: int = 5) -> list:
        """Perform similarity search with memory management"""
        try:
            # Check if we need to cleanup
            self.query_count += 1
            if self.query_count >= CLEANUP_INTERVAL or time.time() - self.last_cleanup > 300:
                self.clear_cache()
                self.query_count = 0
                self.last_cleanup = time.time()
            
            # Check for cached embedding
            if question in self._cached_embeddings:
                logger.debug(f"Using cached embedding for: {question[:20]}...")
                query_embedding = self._cached_embeddings[question]
            else:
                # Generate query embedding
                with torch.no_grad():
                    query_embedding = self.embedding_model.encode(
                        [question], 
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                
                # Cache embedding
                if len(self._cached_embeddings) < MAX_CACHE_SIZE:
                    self._cached_embeddings[question] = query_embedding
            
            # Search FAISS index
            scores, indices = self.index.search(query_embedding, top_k)
            
            # Format results with enhanced context
            results = []
            for i, score in zip(indices[0], scores[0]):
                if i < 0:  # FAISS returns -1 for invalid indices
                    continue
                try:
                    item = self.metadata_db[i]
                    # Create enhanced context string
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
            # Clean up after each query
            self._clear_memory()

    def clear_cache(self, full: bool = False):
        """Release memory resources"""
        logger.info("Clearing vector store cache")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear cached embeddings
        self._cached_embeddings.clear()
        
        # Clear intermediate objects
        if hasattr(self, '_batch_embeddings'):
            del self._batch_embeddings
        
        # Full cleanup
        if full:
            # Move model to CPU to free GPU memory
            if self._embedding_model is not None:
                try:
                    self._embedding_model = self._embedding_model.to('cpu')
                except Exception as e:
                    logger.warning(f"Error moving model to CPU: {str(e)}")
            
            # Force garbage collection
            gc.collect()
            logger.info("Full cache clearance completed")
        else:
            logger.info("Partial cache clearance completed")

    def _clear_memory(self):
        """Release temporary memory resources"""
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear Python garbage
        gc.collect()

    def __del__(self):
        """Destructor to clean up resources"""
        try:
            self.clear_cache(full=True)
        except Exception:
            pass

async def main():
    # Initialize vector database manager
    db_manager = VectorDatabaseManager()

    # Process and store chunks (will load from JSON if available)
    await db_manager.process_and_store_chunks()

    # Example query
    results = db_manager.query(
        "Quali sono le differenze tra OGD e OGT?",
        top_k=3
    )

    # Display results
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