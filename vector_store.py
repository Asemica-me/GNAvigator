import asyncio
from dotenv import load_dotenv
import os
import json
from create_chunks_dict import *
from chromadb.config import Settings
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import shutil


EMBEDDING_MODEL = SentenceTransformerEmbeddingFunction(model_name="intfloat/multilingual-e5-large")
CHROMA_DIR = ".chroma_db"
COLLECTION_NAME = "gna_docs"

class VectorDatabaseManager:
    def __init__(self):
        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)

        # Get or create the collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=EMBEDDING_MODEL,
                metadata={"hnsw:space": "cosine"} # Hierarchical Navigable Small Worlds (HNSW)
            )
        except ValueError as e:
            if "Embedding function name mismatch" in str(e):
                print("Embedding function mismatch detected. Deleting and recreating the collection...")
                # Remove the ChromaDB directory
                if os.path.exists(CHROMA_DIR):
                    shutil.rmtree(CHROMA_DIR)
                # Re-initialize the client and collection
                self.client = chromadb.PersistentClient(path=CHROMA_DIR)
                self.collection = self.client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    embedding_function=EMBEDDING_MODEL,
                    metadata={"hnsw:space": "cosine"}
                )
                print("Collection recreated with the correct embedding function.")
            else:
                raise

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
        # Generate chunks using existing crawler
        chunks = await crawl_and_chunk(sitemap_path, base_domain)
        #print(f"\nNumber of chunks received: {len(chunks)}")

        # Process chunks for storage
        ids, documents, metadatas = [], [], []

        for chunk_id, chunk in chunks.items():
            formatted_id, text_content, metadata = self._format_chunk(chunk)
            ids.append(formatted_id)
            documents.append(text_content)
            metadatas.append(metadata)

        # don't explicitly generate embeddings here, ChromaDB will do it using the embedding function
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        print(f"Database stored in: {os.path.abspath(CHROMA_DIR)}")
        #print(f"Total chunks stored: {self.collection.count()}")

    def query(self, question: str, top_k: int = 5) -> list:
        """Perform similarity search"""
        # We don't need to encode the query ourselves, ChromaDB will use the embedding function
        results = self.collection.query(
            query_texts=[question],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        #print(f"\nQuery results from ChromaDB: {results}")
        return self._format_results(results)

    def _format_results(self, raw_results: dict) -> list:
        """Format ChromaDB results for display"""
        formatted_results = []
        if raw_results and 'documents' in raw_results and raw_results['documents'] and raw_results['metadatas'] and raw_results['distances']:
            for doc, meta, distance in zip(
                raw_results["documents"][0],
                raw_results["metadatas"][0],
                raw_results["distances"][0]
            ):
                formatted_results.append({
                    "score": 1 - distance if distance is not None else None,
                    "title": meta["title"] if meta and "title" in meta else None,
                    "source": meta["source"] if meta and "source" in meta else None,
                    "content": doc[:200] + "..." if doc else None
                })
        return formatted_results

async def main():
    # Initialize vector database manager
    db_manager = VectorDatabaseManager()

    # Process and store chunks
    await db_manager.process_and_store_chunks(SITEMAP_PATH, BASE_DOMAIN)

    # # Example query
    # results = db_manager.query(
    #     "Esempi di implementazione pratica durante la fase iniziale"
    # )

    # # Display results
    # print("\nTop results:")
    # for i, result in enumerate(results, 1):
    #     print(f"\nResult {i}:")
    #     print(f"Title: {result.get('title')}")
    #     print(f"Source: {result.get('source')}")
    #     print(f"Relevance: {result.get('score'):.2%}" if result.get('score') is not None else "N/A")
    #     print(f"Content: {result.get('content')}")

if __name__ == "__main__":
    asyncio.run(main())