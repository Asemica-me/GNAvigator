from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import faiss as FAISS
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_mistralai.embeddings import MistralAIEmbeddings

def calculate_metrics(retrieved: list, relevant: str, k: int):
    retrieved_set = set(str(idx) for idx in retrieved[:k])
    relevant_set = {str(relevant)}
    
    precision = len(retrieved_set & relevant_set) / k
    recall = len(retrieved_set & relevant_set) / 1
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        f"precision@{k}": precision,
        f"recall@{k}": recall,
        f"f1@{k}": f1
    }

def evaluate_row(row, vector_store, chunk_id_map, question_embedding, k):
    D, I = vector_store.search(np.array([question_embedding]).astype('float32'), k)
    retrieved_chunk_ids = [chunk_id_map[i] for i in I[0]]
    return calculate_metrics(retrieved_chunk_ids, str(row["chunk_id"]), k)

def validate_index_alignment(vector_store, gold_standard):
    """Check if gold standard chunk IDs exist in the vector store"""
    print("\n=== Data Alignment Check ===")
    
    # Get all chunk IDs from the vector store metadata
    try:
        _, index_metadata = vector_store.reconstruct_n(0, vector_store.ntotal)
        index_chunk_ids = {str(m.get('chunk_id')) for m in index_metadata if m}
    except Exception as e:
        print(f"Could not retrieve index metadata: {e}")
        index_chunk_ids = set()

    # Get gold standard chunk IDs
    gs_chunk_ids = set(gold_standard['chunk_id'].astype(str))
    
    # Find common and missing IDs
    common_ids = gs_chunk_ids & index_chunk_ids
    missing_in_index = gs_chunk_ids - index_chunk_ids
    
    print(f"Gold standard entries: {len(gs_chunk_ids)}")
    print(f"Index entries: {len(index_chunk_ids)}")
    print(f"Common entries: {len(common_ids)}")
    print(f"Missing in index: {len(missing_in_index)}")
    
    if len(common_ids) == 0:
        raise ValueError("No overlapping chunk IDs between index and gold standard")
        
    return common_ids

def evaluate_retriever(vector_store_path: str, 
                      gold_standard_path: str, 
                      api_key: str, 
                      k: int = 4):
    # Load data
    vector_store = FAISS.read_index(
        vector_store_path,
        embeddings=MistralAIEmbeddings(api_key=api_key),
        allow_dangerous_deserialization=True
    )
    gold_standard = pd.read_csv(gold_standard_path)

    # Get index metadata
    _, metadata = vector_store.reconstruct_n(0, vector_store.ntotal)
    index_ids = {m["chunk_id"] for m in metadata if m}
    
    # Filter gold standard to only existing chunks
    valid_gs = gold_standard[gold_standard["chunk_id"].isin(index_ids)]
    print(f"Evaluating on {len(valid_gs)}/{len(gold_standard)} valid entries")

    # Create chunk_id to index mapping
    chunk_id_to_idx = {m["chunk_id"]: i for i, m in enumerate(metadata) if m}
    
    # Embed questions
    mistral_embeddings = MistralAIEmbeddings(api_key=api_key)
    question_embeddings = mistral_embeddings.embed_documents(valid_gs["question"].tolist())
    
    results = []
    for idx, row in valid_gs.iterrows():
        try:
            # Get true position in index
            true_index = chunk_id_to_idx[row["chunk_id"]]
            
            # Search nearest neighbors
            D, I = vector_store.search(
                np.array([question_embeddings[idx]]).astype('float32'), 
                k + 1  # Search k+1 to account for self-match
            )
            
            # Exclude self-match if present
            retrieved = [i for i in I[0] if i != true_index][:k]
            
            # Calculate metrics
            relevant_in_retrieved = true_index in retrieved
            results.append({
                "precision@k": relevant_in_retrieved / k,
                "recall@k": relevant_in_retrieved,
                "f1@k": (2 * relevant_in_retrieved) / (k + 1) if relevant_in_retrieved else 0
            })
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
    
    return {
        f"precision@{k}": np.mean([r[f"precision@{k}"] for r in results]),
        f"recall@{k}": np.mean([r[f"recall@{k}"] for r in results]),
        f"f1@{k}": np.mean([r[f"f1@{k}"] for r in results]),
        "coverage": len(valid_gs) / len(gold_standard)
    }

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    
    metrics = evaluate_retriever(
        vector_store_path=os.path.join("db", "index.faiss"),
        gold_standard_path=os.path.join("data", "gold_standard.csv"),
        api_key=api_key,
        k=4,
    )
    print(f"\nFinal Retrieval Metrics: {metrics}")