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

os.environ["SPACY_WARNING_IGNORE"] = "W008"
os.environ["NLTK_DATA"] = "/tmp/nlp_data/nltk"

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
FAISS_DIR = ".faiss_db"
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
METADATA_PATH = os.path.join(FAISS_DIR, "metadata.pkl")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VectorDatabaseManager:
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.embedding_model = self.embedding_model.to('cpu')
        
        # Load existing database if available
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            with open(METADATA_PATH, 'rb') as f:
                self.metadata_db = pickle.load(f)
        else:
            self.index = None
            self.metadata_db = []
            # Create storage directory if needed
            os.makedirs(FAISS_DIR, exist_ok=True)

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
        """Main processing pipeline"""
        chunks = await crawl_and_chunk(sitemap_path, base_domain)
        
        # Process chunks for storage
        ids, documents, metadatas = [], [], []
        for chunk_id, chunk in chunks.items():
            formatted_id, text_content, metadata = self._format_chunk(chunk)
            ids.append(formatted_id)
            documents.append(text_content)
            metadatas.append(metadata)
        
        # Generate embeddings in batches
        embeddings = self.embedding_model.encode(
            documents, 
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
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
        print(f"FAISS database stored in: {os.path.abspath(FAISS_DIR)}")

    def _save_to_disk(self):
        """Save index and metadata to disk"""
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump(self.metadata_db, f)

    def query(self, question: str, top_k: int = 5) -> list:
        """Perform similarity search"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [question], 
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for i, score in zip(indices[0], scores[0]):
            if i < 0:  # FAISS returns -1 for invalid indices
                continue
            item = self.metadata_db[i]
            results.append({
                "score": float(score),
                "title": item["metadata"]["title"],
                "source": item["metadata"]["source"],
                "content": item["document"][:200] + "..."
            })
        
        return results

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