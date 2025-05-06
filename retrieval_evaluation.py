from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import faiss as FAISS
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_mistralai.embeddings import MistralAIEmbeddings
from concurrent.futures import ThreadPoolExecutor

# Funzione per calcolare le metriche di valutazione (precision, recall, f1)
def calculate_metrics(retrieved: list, relevant: str, k: int):
    retrieved_set = set([str(idx) for idx in retrieved[:k]])  # Considera solo i primi k risultati
    relevant_set = set([str(relevant)])
    
    precision = len(retrieved_set.intersection(relevant_set)) / k
    recall = len(retrieved_set.intersection(relevant_set)) / 1  # Unico documento rilevante
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return {
        f"precision@{k}": precision,
        f"recall@{k}": recall,
        f"f1@{k}": f1
    }


def evaluate_row(row, vector_store, chunk_id_map, question_embedding, k):
    # Perform search with precomputed embedding
    D, I = vector_store.search(np.array([question_embedding]).astype('float32'), k)
    retrieved_chunk_ids = [chunk_id_map[i] for i in I[0]]
    
    return calculate_metrics(retrieved_chunk_ids, str(row["chunk_id"]), k)

# Funzione principale per valutare il recupero dei dati
def evaluate_retriever(vector_store_path: str, gold_standard_path: str, api_key: str, k: int=4):
    vector_store = FAISS.read_index(vector_store_path)
    gold_standard = pd.read_csv(gold_standard_path)

    # Crea un dizionario che mappa ogni indice al suo chunk_id
    chunk_id_map = dict(zip(range(len(gold_standard)), gold_standard['chunk_id']))

    # Carica il modello di embeddings
    mistral_embeddings = MistralAIEmbeddings(
        api_key=api_key,
        max_retries=10,
        timeout=30,
        wait_time=5,
        max_concurrent_requests=4 
    )

   # Precompute all embeddings first
    questions = gold_standard["question"].tolist()
    question_embeddings = mistral_embeddings.embed_documents(questions)
    
    # Process rows with precomputed embeddings
    results = []
    for idx, row in gold_standard.iterrows():
        metrics = evaluate_row(
            row, 
            vector_store,
            chunk_id_map,
            question_embeddings[idx],
            k
        )
        results.append(metrics)
    
    # Calculate average metrics
    return {
        f"precision@{k}": np.mean([r[f"precision@{k}"] for r in results]), #how many of the retrieved items are relevant
        f"recall@{k}": np.mean([r[f"recall@{k}"] for r in results]), #how many of the relevant items were retrieved
        f"f1@{k}": np.mean([r[f"f1@{k}"] for r in results]) #the harmonic mean of precision and recall
    }

# Carica la variabile d'ambiente per la chiave API
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

# Set the paths to your vector store and gold standard file
vector_store_path = os.path.join("db", "index.faiss")
gold_standard_path = os.path.join("data", "gold_standard.csv")

# Call the evaluate_retriever function with the paths
metrics = evaluate_retriever(vector_store_path, gold_standard_path, api_key)
print(f"Retrieval Metrics: {metrics}")