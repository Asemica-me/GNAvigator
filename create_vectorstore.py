import os
import asyncio
from langchain.document_loaders import Document
from create_embeddings import *

async def get_embeddings_in_batches(client, texts: list[str], embedding_model_name: str, batch_size: int = 32): # Check Mistral's recommended batch size
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            # Each batch is one API call
            response = await invoke_mistral_api_with_retry(
                client.embeddings.create, # Pass the function itself
                model=embedding_model_name,
                input=batch_texts
            )
            all_embeddings.extend([item.embedding for item in response.data])
        except Exception as e:
            print(f"Failed to get embeddings for batch starting at index {i}: {e}")
            # Decide on error handling: skip batch, raise error, or return partial results
            # For simplicity, re-raising here.
            raise
    return all_embeddings

async def create_or_get_vector_store_async(chunks: list[Document], api_key: str, client, embedding_model_name: str) -> FAISS:
    db_dir = "./db"
    vector_store_path = os.path.join(db_dir, "index_gna.faiss")

    if not os.path.exists(db_dir):
        os.makedirs(db_dir)

    # For loading, Langchain's MistralAIEmbeddings is okay as it doesn't make API calls just to load
    # But ensure NO `wait_time` that uses `time.sleep` if you re-use it for anything else.
    # Use a dummy or properly configured async embedding for loading if issues persist.
    # For simplicity and consistency, let's prepare for a scenario where we might need to pass an embedder.
    # We primarily need the embeddings for creation, and an embedder instance for loading.
    embeddings_for_loading = MistralAIEmbeddings(mistral_api_key=api_key) # No wait_time needed for loading

    if not os.path.exists(vector_store_path):
        print("Creating new vector store asynchronously...")
        
        processed_docs = [
            doc for doc in chunks if doc.page_content and doc.page_content.strip()
        ] # Assuming chunks are already Document objects with metadata

        if not processed_docs:
            raise ValueError("No valid documents found to create vector store.")

        texts = [doc.page_content for doc in processed_docs]
        metadatas = [doc.metadata for doc in processed_docs]
        
        print(f"Generating embeddings for {len(texts)} documents. This may take some time...")
        embeddings_list = await get_embeddings_in_batches(client, texts, embedding_model_name)

        if len(embeddings_list) != len(texts):
            raise ValueError(f"Mismatch in number of texts ({len(texts)}) and generated embeddings ({len(embeddings_list)}). Cannot create FAISS index.")

        text_embedding_pairs = list(zip(texts, embeddings_list))
        
        # FAISS.from_embeddings is synchronous. Run in executor.
        loop = asyncio.get_event_loop()
        vector_store = await loop.run_in_executor(
            None, # Uses default ThreadPoolExecutor
            FAISS.from_embeddings,
            text_embedding_pairs,
            embeddings_for_loading, # FAISS still needs an embedding object
            metadatas=metadatas
        )
        await loop.run_in_executor(None, vector_store.save_local, db_dir, "index_gna")
        print("Vector store created and saved.")
    else:
        print(f"Loading vector store from {vector_store_path}...")
        loop = asyncio.get_event_loop()
        vector_store = await loop.run_in_executor(
            None,
            FAISS.load_local,
            db_dir,
            embeddings_for_loading,
            "index_gna",
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded.")
    return vector_store