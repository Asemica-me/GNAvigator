import os
import pandas as pd
import hashlib
from dotenv import load_dotenv
from mistralai import Mistral, UserMessage
from tqdm import tqdm
from main_copy import create_chunks  # Import your existing chunking function
import time

# Load environment variables
load_dotenv()

def generate_chunk_id(url: str, chunk_index: int) -> str:
    """Generate unique chunk ID using SHA-256 hash"""
    return hashlib.sha256(f"{url}-{chunk_index}".encode()).hexdigest()

def generate_questions(client, chunk_text: str, chunk_id: str, model: str, max_retries: int = 5) -> list:
    """Generate questions for chunks using Mistral API with retry logic."""
    prompt = f"""
        Generate three different questions in ITALIAN that could be answered by the following text chunk:\n
        {chunk_text}\n\n       
        Return only the questions, each on a new line, without any numbering, extra characters, or additional text.
    """
    retries = 0
    delay = 2  # Initial delay in seconds

    while retries < max_retries:
        try:
            messages = [
                UserMessage(
                    content=prompt,
                ),
            ]
            response = client.chat.complete(model=model, messages=messages)
            questions = response.choices[0].message.content.split("\n")
            
            # Post-process questions to remove numbering or extra characters
            cleaned_questions = []
            for q in questions:
                q = q.strip()  # Remove leading/trailing whitespace
                q = q.lstrip("1234567890.)- ")  # Remove leading numbers, dots, parentheses, etc.
                q = q.replace('""', '"') 
                q = q.strip('"')
                if q:  # Ensure the question is not empty
                    cleaned_questions.append({"question": q, "chunk_id": chunk_id})
            
            return cleaned_questions
        except Exception as e:
            if "429" in str(e):  # Handle rate limit errors
                print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                retries += 1
            else:
                print(f"Error generating questions for chunk {chunk_id}: {str(e)}")
                return []
    print(f"Failed to generate questions for chunk {chunk_id} after {max_retries} retries.")
    return []

def create_gold_standard(dataset_path: str = "./data/gna_kg_dataset.csv") -> pd.DataFrame:
    """Main function to generate gold standard dataset"""
    # Load configuration
    api_key = os.getenv("MISTRAL_API_KEY")
    model_name = os.getenv("MODEL")
    
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found in .env file")
    
    # Initialize Mistral client
    client = Mistral(api_key=api_key)
    
    # Load and chunk dataset using existing function from main.py
    df = pd.read_csv(dataset_path)
    chunks = create_chunks(df, chunk_size=1000, chunk_overlap=0)
    
    # Generate gold standard
    gold_standard = []
    
    for chunk_idx, chunk in enumerate(tqdm(chunks, desc="Generating questions")):
        try:
            # Extract metadata from chunk
            url = chunk.metadata.get("url", "")
            chunk_id = chunk.metadata.get("chunk_id", "")  # Retrieve chunk_id from metadata
            chunk_text = chunk.page_content
            
            # Generate questions
            questions = generate_questions(client, chunk_text, chunk_id, model_name)
            
            # Append questions with chunk_id to the gold standard
            for question in questions:
                gold_standard.append({
                    "question": question["question"], 
                    "chunk_id": chunk_id
                })
            
            # Rate limiting
            time.sleep(5)
            
        except Exception as e:
            print(f"Error processing chunk {chunk_idx}: {str(e)}")
            continue
    
    # Convert gold standard to DataFrame
    return pd.DataFrame(gold_standard)

if __name__ == "__main__":
    gold_df = create_gold_standard()
    output_path = "./data/gold_standard.csv"
    gold_df.to_csv(output_path, index=False)
    print(f"Gold standard dataset saved to {output_path}")