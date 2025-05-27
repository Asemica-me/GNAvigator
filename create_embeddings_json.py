import json
import os
import asyncio
from tqdm.asyncio import tqdm
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from create_chunks import OUTPUT_FOLDER, OUTPUT_FILENAME, SITEMAP_PATH, BASE_DOMAIN

# --- Configuration for Embeddings ---
load_dotenv()
OUTPUT_FOLDER = "data"
CHUNKS_INPUT_PATH = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)
EMBEDDINGS_OUTPUT_FILENAME = "chunks_with_embeddings.json"
EMBEDDINGS_OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, EMBEDDINGS_OUTPUT_FILENAME)

# --- Initialize Embedding Model ---
# Using a multilingual model suitable for Italian text
try:
    embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    print("Sentence Transformer model 'paraphrase-multilingual-mpnet-base-v2' loaded successfully.")
except Exception as e:
    embedding_model = None
    print(f"Error loading Sentence Transformer model: {e}")
    print("Embedding generation will be skipped.")

async def add_embeddings_to_chunks(chunks: list) -> list:
    """
    Adds embeddings to a list of chunks using the pre-loaded embedding model.
    If the embedding model is not loaded, it skips embedding generation.
    """
    if embedding_model is None:
        print("Embedding model not available. Skipping embedding generation.")
        return chunks

    print(f"Generating embeddings for {len(chunks)} chunks...")
    
    # Extract content for embedding in a batch
    contents = [chunk['content'] for chunk in chunks]
    
    # Generate embeddings in a batch
    try:
        embeddings = embedding_model.encode(contents, show_progress_bar=True).tolist()
    except Exception as e:
        print(f"Error during batch embedding generation: {e}")
        print("Returning chunks without embeddings.")
        return chunks

    # Assign embeddings back to chunks
    for i, chunk in enumerate(chunks):
        chunk['embedding'] = embeddings[i]
    
    print("Embeddings generated and added to chunks.")
    return chunks

def load_chunks_from_json(file_path: str) -> list:
    """Loads chunks from a JSON file."""
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks from {file_path}.")
        return chunks
    except FileNotFoundError:
        print(f"Error: Chunks file not found at {file_path}. Please run create_chunks.py first.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading chunks: {e}")
        return []

def save_chunks_with_embeddings(chunks: list, output_path: str):
    """Saves a list of chunks (with embeddings) to a JSON file."""
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)
    print(f"\nChunks with embeddings saved to: {output_path}")

async def main():
    """Main function to load chunks, add embeddings, and save them."""
    # Ensure the output folder exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. Load chunks that were previously created by create_chunks.py
    chunks_without_embeddings = load_chunks_from_json(CHUNKS_INPUT_PATH)

    if not chunks_without_embeddings:
        print("No chunks to process. Exiting.")
        return

    # 2. Add embeddings to the loaded chunks
    chunks_with_embeddings = await add_embeddings_to_chunks(chunks_without_embeddings)

    # 3. Save the chunks with embeddings
    save_chunks_with_embeddings(chunks_with_embeddings, EMBEDDINGS_OUTPUT_PATH)

if __name__ == "__main__":
    print("Starting embedding generation process...")
    asyncio.run(main())
    print("Embedding generation process completed.")
