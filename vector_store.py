import asyncio
import os
import shutil
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from create_chunks_dict import *
import torch
import gc
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment setup
os.environ["SPACY_WARNING_IGNORE"] = "W008"
os.environ["NLTK_DATA"] = "/tmp/nlp_data/nltk"

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
FAISS_DIR = ".faiss_db"
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
METADATA_PATH = os.path.join(FAISS_DIR, "metadata.pkl")

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
        
        return self._embedding_model

    def _format_chunk(self, chunk: dict) -> tuple:
        """Helper method to format chunk data for storage"""
        text_content = " ".join([
            chunk["title"],
            chunk["content"]
        ])

        metadata = {
            "source": chunk["source"],
            "content_type": chunk["content_type"],
            "title": chunk["title"]
        }

        return chunk["chunk_id"], text_content, metadata

    async def process_and_store_chunks(self, sitemap_path: str, base_domain: str):
        """Main processing pipeline with memory management"""
        try:
            chunks = await crawl_and_chunk(sitemap_path, base_domain)
            
            # Process chunks for storage
            ids, documents, metadatas = [], [], []
            for chunk_id, chunk in chunks.items():
                formatted_id, text_content, metadata = self._format_chunk(chunk)
                ids.append(formatted_id)
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
            else:
                # Add to existing index (not efficient for large updates)
                self.index.add(embeddings)
                for i, doc, meta in zip(ids, documents, metadatas):
                    self.metadata_db.append({
                        "id": i, 
                        "document": doc, 
                        "metadata": meta
                    })
            
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
        
        # Concatenate and return
        return np.vstack(all_embeddings)

    def _save_to_disk(self):
        """Save index and metadata to disk"""
        try:
            faiss.write_index(self.index, FAISS_INDEX_PATH)
            with open(METADATA_PATH, 'wb') as f:
                pickle.dump(self.metadata_db, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Database saved to disk")
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
            
            # Format results
            results = []
            for i, score in zip(indices[0], scores[0]):
                if i < 0:  # FAISS returns -1 for invalid indices
                    continue
                try:
                    item = self.metadata_db[i]
                    results.append({
                        "score": float(score),
                        "title": item["metadata"]["title"],
                        "source": item["metadata"]["source"],
                        "content": item["document"][:200] + "..." if item["document"] else ""
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
            
            # Reset database references
            self.index = None
            self.metadata_db = []
            
            # Reload from disk if available
            if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
                try:
                    self.index = faiss.read_index(FAISS_INDEX_PATH)
                    with open(METADATA_PATH, 'rb') as f:
                        self.metadata_db = pickle.load(f)
                    logger.info("Reloaded database from disk")
                except Exception as e:
                    logger.error(f"Error reloading database: {str(e)}")
                    self._reset_database()
        
        # Force garbage collection
        gc.collect()
        logger.info("Cache cleared")

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

    # Process and store chunks only if no existing database
    if not db_manager.metadata_db:
        await db_manager.process_and_store_chunks(SITEMAP_PATH, BASE_DOMAIN)
    else:
        print("Using existing FAISS database")

    # Example query
    results = db_manager.query(
        "Esempi di implementazione pratica durante la fase iniziale"
    )

    # Display results
    print("\nTop results:")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Title: {result.get('title')}")
        print(f"Source: {result.get('source')}")
        print(f"Relevance: {result.get('score'):.2%}" if result.get('score') is not None else "N/A")
        print(f"Content: {result.get('content')}")

if __name__ == "__main__":
    asyncio.run(main())