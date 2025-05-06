import os
import pandas as pd
import hashlib
from dotenv import load_dotenv
from mistralai import Mistral, UserMessage
from tqdm import tqdm
from main import create_chunks  # Import your existing chunking function
from langchain_community.vectorstores import FAISS
import time

def generate_question(client, chunk_text: str, chunk_id: str, model: str) -> list:
    """Generate questions for chunks using Mistral API"""
    prompt = f"""
        Generate 3 different questions that could be answered by this text chunk:\n
        {chunk_text}\n\n       
        
        Return ONLY the questions separated by newlines as a list, no numbering or extra text.
        """    
    try:
        messages = [
         UserMessage(
             content=prompt,
         ),
     ]
    
        response = client.chat.complete(model=model, messages=messages)
        questions = response.choices[0].message.content.split("\n")
        return [{"chunk_id": chunk_id, "question": q.strip()} 
                for q in questions if q.strip()]
    except Exception as e:
        print(f"Error generating questions for chunk {chunk_id}: {str(e)}")
        return []
    
def generate_gold_standard(vector_store: FAISS, output_path: str, num_questions: int = 500):
    """Generates gold standard CSV linked to vector store chunk IDs"""
    # Initialize Mistral client
    load_dotenv()
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    
    # Get all chunk IDs and contents from the index
    _, metadata = vector_store.reconstruct_n(0, vector_store.ntotal)
    chunk_data = {m["chunk_id"]: m.get("page_content", "") for m in metadata if m}
    
    # Generate questions
    gold_standard = []
    model_name = "mistral-large-latest"  # Update with your preferred model
    
    for chunk_id, content in tqdm(list(chunk_data.items())[:num_questions], 
                               desc="Generating questions"):
        try:
            # Generate questions for this chunk
            questions = generate_question(
                client=client,
                chunk_text=content,
                chunk_id=chunk_id,
                model=model_name
            )
            
            # Add each question as a separate entry
            gold_standard.extend(questions)
            
        except Exception as e:
            print(f"\nError processing chunk {chunk_id}: {str(e)}")
            continue

    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    try:
        pd.DataFrame(gold_standard).to_csv(output_path, index=False)
        print(f"\nSuccessfully saved {len(gold_standard)} questions to {output_path}")
    except Exception as e:
        print(f"\nFailed to save CSV: {str(e)}")