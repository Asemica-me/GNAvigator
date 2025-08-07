# ablation/experiments.py
import os
from pathlib import Path
from typing import List, Dict, Any, Set
import json
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vector_store import VectorDatabaseManager
from dotenv import load_dotenv

load_dotenv()


class VectorDatabaseWrapper:
    def __init__(self, vector_db: VectorDatabaseManager):
        self.vector_db = vector_db
        self._cached_embeddings = {}
        
    def query(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # Check cache first
        if question in self._cached_embeddings:
            return self._cached_embeddings[question]
        
        # Query database and cache results
        results = self.vector_db.query(question, top_k) or []
        self._cached_embeddings[question] = results
        return results
                
    def query_batch(self, questions: List[str], top_k: int = 5) -> List[List[Dict[str, Any]]]:
        results = []
        # Process cached and uncached questions separately
        uncached_questions = []
        
        for q in questions:
            if q in self._cached_embeddings:
                results.append(self._cached_embeddings[q])
            else:
                uncached_questions.append(q)
                results.append(None)  # Placeholder
        
        # Batch query for uncached questions
        if uncached_questions:
            # Use existing query method if batch isn't available
            batch_results = [self.query(q, top_k) for q in uncached_questions]
            
            # Update results and cache
            for i, q in enumerate(uncached_questions):
                results[questions.index(q)] = batch_results[i]
                
        return results


class DenseRetriever:
    def __init__(self, device=None):
        self.vector_db = VectorDatabaseWrapper(VectorDatabaseManager(device=device))

    def _format_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Standardize result format"""
        return [{
            'id': item['id'],
            'chunk_id': item['id'],
            'score': item['score'],
            **item
        } for item in results]

    def retrieve(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.vector_db.query(question, top_k=top_k)
        return self._format_results(results)
    
    def query_with_scores(self, question: str, top_k: int = 20) -> List[Dict[str, Any]]:
        results = self.vector_db.query(question, top_k=top_k)
        return self._format_results(results)
    

    
# === EVALUATION LOGIC  ===

import numpy as np
import pandas as pd

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent  # ablation/
DATA_DIR = root_dir / "data"
METRICS_DIR = DATA_DIR / "metrics"
ABLATION_METRICS_DIR = METRICS_DIR / "ablation_metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
ABLATION_METRICS_DIR.mkdir(parents=True, exist_ok=True)

def compute_ndcg(relevant_ids: Set[str], retrieved_docs: List[Dict], k: int = 5) -> float:
    if k <= 0 or not retrieved_docs:
        return 0.0
    top_k = retrieved_docs[:k]
    y_true = [1.0 if doc['id'] in relevant_ids else 0.0 for doc in top_k]
    ideal_sorted = sorted(y_true, reverse=True)
    idcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_sorted))
    dcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(y_true))
    return dcg / idcg if idcg > 0 else 0.0

def compute_ap(relevant_ids: Set[str], retrieved_docs: List[Dict], k: int = 5) -> float:
    if not relevant_ids or not retrieved_docs:
        return 0.0
    top_k = retrieved_docs[:k]
    relevant_count = 0
    precision_sum = 0.0
    for i, doc in enumerate(top_k):
        if doc['id'] in relevant_ids:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)
    return precision_sum / min(len(relevant_ids), k)

class RetrieverEvaluator:
    def __init__(self, retriever: DenseRetriever, batch_size: int = 32):
        self.retriever = retriever
        self.batch_size = batch_size
        self.metrics = []

    def evaluate_retrieval(self, test_file: Path, top_k: int = 5) -> Dict[str, Any]:
        try:
            with open(test_file, "r", encoding="utf-8") as f:
                test_data = json.load(f)
        except Exception as e:
            print(f"Failed to load test file: {e}")
            return None

        results = []
        total_questions = len(test_data)

        for i in range(0, total_questions, self.batch_size):
            batch = test_data[i:i + self.batch_size]
            batch_results = self._evaluate_query_batch(
                questions=[item["question"] for item in batch],
                relevant_docs_batch=[set(item["relevant_docs"]) for item in batch],
                top_k=top_k
            )
            results.extend(batch_results)
            print(f"Processed {min(i + self.batch_size, total_questions)}/{total_questions} questions")

        self.metrics = results
        return self._aggregate_metrics()

    def _evaluate_query_batch(self, questions: List[str], relevant_docs_batch: List[Set[str]], top_k: int) -> List[Dict[str, Any]]:
        batch_results = []
        start_time = time.perf_counter()
        retrieved_docs_batch = [self.retriever.retrieve(q, top_k) for q in questions]
        retrieval_time = (time.perf_counter() - start_time) / len(questions)
        for i, (retrieved_docs, relevant_set) in enumerate(zip(retrieved_docs_batch, relevant_docs_batch)):
            retrieved_docs = retrieved_docs or []
            retrieved_ids = [doc["id"] for doc in retrieved_docs]
            relevant_retrieved = len(set(retrieved_ids) & relevant_set)
            precision = relevant_retrieved / top_k if top_k > 0 else 0
            recall = relevant_retrieved / len(relevant_set) if relevant_set else 0
            reciprocal_rank = 0
            for rank, doc in enumerate(retrieved_docs, 1):
                if doc["id"] in relevant_set:
                    reciprocal_rank = 1 / rank
                    break
            ndcg = compute_ndcg(relevant_set, retrieved_docs, k=top_k)
            ap = compute_ap(relevant_set, retrieved_docs, k=top_k)
            batch_results.append({
                "question": questions[i],
                "precision@k": precision,
                "recall@k": recall,
                "mrr": reciprocal_rank,
                "ndcg@k": ndcg,
                "ap@k": ap,
                "retrieval_time": retrieval_time,
                "retrieved_docs": retrieved_ids,
                "relevant_docs": list(relevant_set),
            })
        return batch_results

    def _aggregate_metrics(self) -> Dict[str, Any]:
        if not self.metrics:
            return {
                "num_queries": 0,
                "avg_precision@k": 0,
                "avg_recall@k": 0,
                "avg_mrr": 0,
                "avg_ndcg@k": 0,
                "avg_ap@k": 0,
                "avg_retrieval_time": 0
            }
        df = pd.DataFrame(self.metrics)
        return {
            "num_queries": int(len(df)),
            "avg_precision@k": float(df["precision@k"].mean()),
            "avg_recall@k": float(df["recall@k"].mean()),
            "avg_mrr": float(df["mrr"].mean()),
            "avg_ndcg@k": float(df["ndcg@k"].mean()),
            "avg_ap@k": float(df["ap@k"].mean()),
            "avg_retrieval_time": float(df["retrieval_time"].mean())
        }

    def save_reports(self, metrics: Dict[str, Any], prefix: str = "dense_eval") -> None:
        """Save only aggregated metrics to file"""
        if not metrics or metrics["num_queries"] == 0:
            print("No metrics to save")
            return
        metrics_file = ABLATION_METRICS_DIR / f"{prefix}_metrics.json"
        agg_metrics = {k: v for k, v in metrics.items() if k != "by_question"}
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(agg_metrics, f, indent=2, ensure_ascii=False)

import random

def load_and_shuffle_datasets(singlehop_path, multihop_path):
    with open(singlehop_path, encoding="utf-8") as f1:
        singlehop = json.load(f1)
    with open(multihop_path, encoding="utf-8") as f2:
        multihop = json.load(f2)
    combined = singlehop + multihop
    random.shuffle(combined)
    return combined

def main(top_k: int = 5):
    device = os.getenv("DEVICE", "cpu")
    retriever = DenseRetriever(device=device)
    evaluator = RetrieverEvaluator(retriever, batch_size=32)

    SINGLEHOP_FILE = DATA_DIR / "test" / "test_dataset_singlehop.json"
    MULTIHOP_FILE = DATA_DIR / "test" / "test_dataset_multihop.json"

    # 1. Evaluate on singlehop only
    print("\n=== Evaluating SINGLEHOP dataset ===")
    singlehop_metrics = evaluator.evaluate_retrieval(SINGLEHOP_FILE, top_k)
    if singlehop_metrics and singlehop_metrics["num_queries"] > 0:
        print(f"Singlehop: Queries evaluated: {singlehop_metrics['num_queries']}")
        print(f"Avg Precision@{top_k}: {singlehop_metrics['avg_precision@k']:.4f}")
        print(f"Avg Recall@{top_k}: {singlehop_metrics['avg_recall@k']:.4f}")
        print(f"Mean Reciprocal Rank: {singlehop_metrics['avg_mrr']:.4f}")
        print(f"Avg nDCG@{top_k}: {singlehop_metrics['avg_ndcg@k']:.4f}")
        print(f"Avg AP@{top_k}: {singlehop_metrics['avg_ap@k']:.4f}")
        print(f"Avg Retrieval Time: {singlehop_metrics['avg_retrieval_time']:.6f}s")
        evaluator.save_reports(singlehop_metrics, prefix="dense_eval_singlehop")
        print(f"Reports saved to {ABLATION_METRICS_DIR}")
    else:
        print("Singlehop evaluation failed.")

    # 2. Evaluate on singlehop + multihop (shuffled)
    print("\n=== Evaluating SINGLE+MULTIHOP dataset ===")
    # Load, combine, and shuffle datasets
    combined_data = load_and_shuffle_datasets(SINGLEHOP_FILE, MULTIHOP_FILE)
    temp_combined_file = DATA_DIR / "test" / "test_dataset_combined_temp.json"
    with open(temp_combined_file, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
    all_metrics = evaluator.evaluate_retrieval(temp_combined_file, top_k)
    if all_metrics and all_metrics["num_queries"] > 0:
        print(f"Combined: Queries evaluated: {all_metrics['num_queries']}")
        print(f"Avg Precision@{top_k}: {all_metrics['avg_precision@k']:.4f}")
        print(f"Avg Recall@{top_k}: {all_metrics['avg_recall@k']:.4f}")
        print(f"Mean Reciprocal Rank: {all_metrics['avg_mrr']:.4f}")
        print(f"Avg nDCG@{top_k}: {all_metrics['avg_ndcg@k']:.4f}")
        print(f"Avg AP@{top_k}: {all_metrics['avg_ap@k']:.4f}")
        print(f"Avg Retrieval Time: {all_metrics['avg_retrieval_time']:.6f}s")
        evaluator.save_reports(all_metrics, prefix="dense_eval_all")
        print(f"Reports saved to {ABLATION_METRICS_DIR}")
    else:
        print("Combined evaluation failed.")

if __name__ == "__main__":
    main(top_k=5)