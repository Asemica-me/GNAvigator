import asyncio
import os
import json
from create_chunks_dict import *
from chromadb.config import Settings
import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHROMA_DIR = ".chroma_db"
COLLECTION_NAME = "gna_docs"

class VectorDatabaseManager:
    def __init__(self):
        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        
        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

    def _format_chunk(self, chunk: dict) -> tuple:
        """Helper method to format chunk data for storage"""
        text_content = " ".join([
            chunk["title"],
            " ".join(chunk["keywords"]),
            chunk["content"]
        ])
        
        metadata = {
            "source": chunk["source"],
            "content_type": chunk["content_type"],
            "title": chunk["title"],
            "headers": " | ".join(chunk["headers_context"]),
            "keywords": ", ".join(chunk["keywords"])
        }
        
        return chunk["chunk_id"], text_content, metadata

    async def process_and_store_chunks(self, sitemap_path: str, base_domain: str):
        """Main processing pipeline"""
        # Generate chunks using existing crawler
        chunks = await crawl_and_chunk(sitemap_path, base_domain)
        
        # Process chunks for storage
        ids, documents, metadatas = [], [], []
        
        for chunk in chunks:
            chunk_id, text_content, metadata = self._format_chunk(chunk)
            ids.append(chunk_id)
            documents.append(text_content)
            metadatas.append(metadata)
        
        # Generate embeddings
        embeddings = self.embedder.encode(documents).tolist()
        
        # Store in ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"Database stored in: {os.path.abspath(CHROMA_DIR)}")
        print(f"Total chunks stored: {len(ids)}")

    def query(self, question: str, top_k: int = 5) -> list:
        """Perform similarity search"""
        query_embedding = self.embedder.encode([question]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        return self._format_results(results)

    def _format_results(self, raw_results: dict) -> list:
        """Format ChromaDB results for display"""
        return [{
            "score": 1 - distance,
            "title": meta["title"],
            "source": meta["source"],
            "content": doc[:200] + "..."
        } for doc, meta, distance in zip(
            raw_results["documents"][0],
            raw_results["metadatas"][0],
            raw_results["distances"][0]
        )]

async def main():
    # Initialize vector database manager
    db_manager = VectorDatabaseManager()
    
    # Process and store chunks
    await db_manager.process_and_store_chunks(SITEMAP_PATH, BASE_DOMAIN)
    
    # Example query
    results = db_manager.query(
        "Esempi di implementazione pratica durante la fase iniziale"
    )
    
    # Display results
    print("\nTop results:")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Title: {result['title']}")
        print(f"Source: {result['source']}")
        print(f"Relevance: {result['score']:.2%}")
        print(f"Content: {result['content']}")

if __name__ == "__main__":
    asyncio.run(main())