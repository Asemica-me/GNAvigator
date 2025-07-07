import asyncio
import gc
import json
import logging
import os
import pickle
import random
import time

import faiss
import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from create_chunks_sent import BASE_DOMAIN, SITEMAP_PATH, crawl_and_chunk

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
    def __init__(self, device: str = None):
        self._embedding_model = None
        self.index = None
        self.metadata_db = []
        self.query_count = 0
        self.last_cleanup = time.time()
        self._cached_embeddings = {}
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load existing database if available
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
            try:
                self.index = faiss.read_index(FAISS_INDEX_PATH)
                with open(METADATA_PATH, "rb") as f:
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
            self._embedding_model = SentenceTransformer(
                EMBEDDING_MODEL,
                device=self.device,
            )

        return self._embedding_model

    def _format_chunk(self, chunk: dict) -> tuple:
        context_str = " | ".join(chunk.get("headers_context", []))
        text_content = (
            f"{context_str}\n\n{chunk.get('title', '')}\n{chunk.get('content', '')}"
        )

        metadata = {
            "source": chunk.get("source", ""),
            "content_type": chunk.get("content_type", "text"),
            "title": chunk.get("title", ""),
            "headers_context": chunk.get("headers_context", []),
            "keywords": chunk.get("keywords", []),
            "entities": chunk.get("entities", []),
            "chunk_index": chunk.get("chunk_index", 0),
        }

        return chunk["chunk_id"], text_content, metadata

    async def process_and_store_chunks(self):
        try:
            # Load chunks from JSON
            if os.path.exists(CHUNK_JSON_PATH):
                with open(CHUNK_JSON_PATH, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                logger.info(f"Loaded {len(chunks)} chunks from {CHUNK_JSON_PATH}")
            else:
                logger.info("No chunk JSON found, generating chunks...")
                chunks = await crawl_and_chunk(SITEMAP_PATH, BASE_DOMAIN)
                logger.info(f"Generated {len(chunks)} chunks")

            # Prepare data for storage
            ids, documents, metadatas = [], [], []
            for chunk in chunks:
                if chunk is None:
                    continue  # Skip null entries
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
                    embeddings = self._generate_embeddings(documents)
                    # Cache embeddings for future runs
                    np.save(EMBEDDINGS_CACHE_PATH, embeddings)
                    logger.info(f"Saved embeddings cache to {EMBEDDINGS_CACHE_PATH}")
                else:
                    logger.error("No embedding model available and no cache found")
                    return

            # Create or update FAISS index
            self.index = None
            self.metadata_db = []

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
                    self.metadata_db.append(
                        {"id": i, "document": doc, "metadata": meta}
                    )
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

    def _generate_embeddings(self, documents: list) -> np.ndarray:
        """Embedding generation with batching and memory management"""
        # Device is always CPU on Streamlit cloud
        batch_size = 32
        logger.info(
            f"Generating embeddings on {self.device} with batch size {batch_size}"
        )

        all_embeddings = []
        # total_batches = (len(documents) + batch_size - 1) // batch_size

        # Use progress bar for visibility
        pbar = tqdm(total=len(documents), desc="Generating embeddings", unit="doc")

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            # Generate embeddings without autocast
            batch_embeddings = self.embedding_model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
            )

            all_embeddings.extend(batch_embeddings)
            pbar.update(len(batch))

            # Memory management
            if i > 0 and i % (10 * batch_size) == 0:
                self._clear_memory()

        pbar.close()
        return np.array(all_embeddings)

    def _save_to_disk(self):
        try:
            faiss.write_index(self.index, FAISS_INDEX_PATH)
            with open(METADATA_PATH, "wb") as f:
                pickle.dump(self.metadata_db, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Database saved to {FAISS_DIR}")
        except Exception as e:
            logger.error(f"Error saving database: {str(e)}")

    def query(self, question: str, top_k: int = 5) -> list:
        """Query the vector store with a question and return top_k results"""
        try:
            # Query management
            self.query_count += 1
            if self.query_count >= CLEANUP_INTERVAL:
                self.clear_cache()
                self.query_count = 0

            # Generate embedding using main model
            query_embedding = self.embedding_model.encode(
                [question], convert_to_numpy=True, normalize_embeddings=True
            )

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
                    context_str = " → ".join(
                        item["metadata"].get("headers_context", [])
                    )
                    content_preview = (
                        f"[Context: {context_str}]\n{item['document'][:500]}..."
                    )

                    results.append(
                        {
                            "score": float(score),
                            "id": self.metadata_db[i]["id"],
                            "title": item["metadata"].get("title", ""),
                            "source": item["metadata"].get("source", ""),
                            "content_type": item["metadata"].get(
                                "content_type", "text"
                            ),
                            "content": content_preview,
                            "full_metadata": item["metadata"],
                        }
                    )
                except IndexError:
                    logger.warning(f"Invalid index {i} in metadata database")

            return results

        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return []
        finally:
            self._clear_memory()

    def query_batch(self, questions: list[str], top_k: int = 5) -> list:
        """Query the vector store with a question and return top_k results"""
        try:
            # Query management
            self.query_count += len(questions)
            if self.query_count >= CLEANUP_INTERVAL:
                self.clear_cache()
                self.query_count = 0

            # Generate embedding using main model
            query_embeddings = self.embedding_model.encode(
                questions, convert_to_numpy=True, normalize_embeddings=True
            )

            if len(self._cached_embeddings) < MAX_CACHE_SIZE:
                for question, embedding in zip(questions, query_embeddings):
                    self._cached_embeddings[question] = embedding

            # Search FAISS index
            scores, indices = self.index.search(query_embeddings, top_k)

            # Format results
            results = []
            for i_arr, score_arr in zip(indices, scores):
                inner_results = []
                for i, score in zip(i_arr, score_arr):
                    if i < 0:
                        continue
                    try:
                        item = self.metadata_db[i]
                        context_str = " → ".join(
                            item["metadata"].get("headers_context", [])
                        )
                        content_preview = (
                            f"[Context: {context_str}]\n{item['document'][:500]}..."
                        )

                        inner_results.append(
                            {
                                "score": float(score),
                                "id": self.metadata_db[i]["id"],
                                "title": item["metadata"].get("title", ""),
                                "source": item["metadata"].get("source", ""),
                                "content_type": item["metadata"].get(
                                    "content_type", "text"
                                ),
                                "content": content_preview,
                                "full_metadata": item["metadata"],
                            }
                        )
                    except IndexError:
                        logger.warning(f"Invalid index {i} in metadata database")

                results.append(inner_results)
            return results

        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return []
        finally:
            self._clear_memory()

    # query_expander = pipeline("text2text-generation", model="google/mt5-large")

    # def expand_query(self, query: str):
    #     # Lazy load the query expander
    #     if not hasattr(self, '_query_expander'):
    #         self._query_expander = pipeline(
    #             "text2text-generation",
    #             model="google/mt5-base",  # Smaller model
    #             device=-1  # Force CPU
    #         )
    #     return self._query_expander(
    #         f"expand query: {query}",
    #         max_length=64,
    #         num_return_sequences=1,
    #         do_sample=True
    #     )[0]['generated_text']

    def clear_cache(self, full: bool = False):
        logger.info("Clearing vector store cache")
        self._cached_embeddings.clear()

        if full and self._embedding_model:
            try:
                del self._embedding_model
                self._embedding_model = None
                logger.info("Unloaded embedding model")
            except Exception as e:
                logger.warning(f"Error deleting embedding model: {str(e)}")

        self._clear_memory()

    def _clear_memory(self):
        gc.collect()

    def sample_documents(self, n: int) -> list:
        """Randomly sample documents from the database"""
        if not self.metadata_db:
            logger.warning("No documents in database to sample")
            return []

        # Ensure we don't request more than available
        n = min(n, len(self.metadata_db))

        # Generate random indices without replacement
        sample_indices = random.sample(range(len(self.metadata_db)), n)

        # Retrieve sampled documents
        samples = []
        for idx in sample_indices:
            item = self.metadata_db[idx]
            samples.append(
                {
                    "id": item["id"],
                    "document": item["document"],
                    "metadata": item["metadata"],
                    "content": f"{item['metadata'].get('title', '')}\n{item['document'][:500]}...",
                }
            )

        logger.info(f"Sampled {n} documents from database")
        return samples

    def get_documents_by_ids(self, doc_ids: list) -> list:
        """Retrieve documents by their IDs"""
        if not self.metadata_db:
            return []

        # Create lookup dictionary for faster access
        id_to_doc = {item["id"]: item for item in self.metadata_db}

        results = []
        for doc_id in doc_ids:
            if doc_id in id_to_doc:
                item = id_to_doc[doc_id]
                results.append(
                    {
                        "id": doc_id,
                        "document": item["document"],
                        "metadata": item["metadata"],
                        "content": f"{item['metadata'].get('title', '')}\n{item['document'][:500]}...",
                    }
                )

        logger.info(f"Retrieved {len(results)} documents by ID")
        return results


async def main():
    db_manager = VectorDatabaseManager()
    await db_manager.process_and_store_chunks()

    # # Example query
    # results = db_manager.query(
    #     "Quali sono le differenze tra OGD e OGT?",
    #     top_k=3
    # )

    # print("\nTop results:")
    # for i, result in enumerate(results, 1):
    #     print(f"\n--- RESULT {i} ---")
    #     print(f"Title: {result.get('title')}")
    #     print(f"Source: {result.get('source')}")
    #     print(f"Content type: {result.get('content_type')}")
    #     print(f"Relevance: {result.get('score'):.2%}")
    #     print(f"Content preview:\n{result.get('content')}")


if __name__ == "__main__":
    asyncio.run(main())
