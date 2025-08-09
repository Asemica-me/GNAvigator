import json
import os
import time
from sklearn.metrics import ndcg_score, average_precision_score
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rag_sys import RAGOrchestrator

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
METRICS_DIR = os.path.join(DATA_DIR, "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)

def compute_ndcg(relevant_ids, retrieved_ids, k=5):
    y_true = [1 if rid in relevant_ids else 0 for rid in retrieved_ids[:k]]
    y_score = [k - i for i in range(len(retrieved_ids[:k]))]
    return ndcg_score([y_true], [y_score]) if any(y_true) else 0.0

def compute_ap(relevant_ids, retrieved_ids, k=5):
    y_true = [1 if rid in relevant_ids else 0 for rid in retrieved_ids[:k]]
    return average_precision_score([1 if i in relevant_ids else 0 for i in retrieved_ids[:k]], y_true) if any(y_true) else 0.0

class RAGEvaluator:
    def __init__(self, mistral_api_key: str, batch_size: int = 32, device: str = None):
        self.orchestrator = RAGOrchestrator(
            mistral_api_key=mistral_api_key, device=device
        )
        self.metrics = []
        self.batch_size = batch_size

    def evaluate_retrieval(self, test_file: str, top_k: int = 5):
        try:
            with open(test_file, encoding="utf-8") as f:
                test_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Test file not found at {test_file}")
            return None
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {test_file}")
            return None

        results = []
        batch_size = min(self.batch_size, len(test_data))
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i : i + batch_size]
            questions_batch = [item["question"] for item in batch]
            relevant_docs_batch = [item["relevant_docs"] for item in batch]
            results.extend(
                self._evaluate_query_batch(questions_batch, relevant_docs_batch, top_k)
            )

        self.metrics = results
        return self._aggregate_metrics()

    def _evaluate_query_batch(
        self, questions: list[str], relevant_docs_batch: list[list[str]], top_k: int
    ):
        results = []
        try:
            start_time = time.time()
            retrieved_docs_batch = self.orchestrator.retrieve_docs_batch(
                questions, top_k
            )
            retrieval_time = (time.time() - start_time) / len(questions)

            for retrieved_docs, relevant_docs, question in zip(
                retrieved_docs_batch, relevant_docs_batch, questions
            ):
                retrieved_ids = [doc["id"] for doc in retrieved_docs] if retrieved_docs else []
                relevant_set = set(relevant_docs)

                relevant_retrieved = len(set(retrieved_ids) & relevant_set)
                precision = relevant_retrieved / top_k if top_k > 0 else 0
                recall = relevant_retrieved / len(relevant_set) if relevant_set else 0

                reciprocal_rank = 0
                if retrieved_docs:
                    for rank, doc in enumerate(retrieved_docs, 1):
                        if doc["id"] in relevant_set:
                            reciprocal_rank = 1 / rank
                            break

                ndcg = compute_ndcg(relevant_docs, retrieved_ids, k=top_k)
                ap = compute_ap(relevant_docs, retrieved_ids, k=top_k)

                input_tokens = (
                    len(self.orchestrator.tokenize(question))
                    if hasattr(self.orchestrator, "tokenize")
                    else 0
                )

                results.append(
                    {
                        "question": question,
                        "precision@k": precision,
                        "recall@k": recall,
                        "mrr": reciprocal_rank,
                        "ndcg@k": ndcg,
                        "ap@k": ap,
                        "retrieval_time": retrieval_time,
                        "input_tokens": input_tokens,
                        "retrieved_docs": retrieved_ids,
                        "relevant_docs": relevant_docs,
                    }
                )

            return results
        except Exception as e:
            print(f"Batch evaluation error: {str(e)}")
            return []

    def _aggregate_metrics(self):
        if not self.metrics:
            return {
                "num_queries": 0,
                "avg_precision@k": 0,
                "avg_recall@k": 0,
                "avg_mrr": 0,
                "avg_ndcg@k": 0,
                "avg_ap@k": 0,
                "avg_retrieval_time": 0,
                "total_input_tokens": 0,
                "by_question": [],
            }

        df = pd.DataFrame(self.metrics)
        metrics_df = df.fillna(0)
        return {
            "num_queries": len(metrics_df),
            "avg_precision@k": metrics_df["precision@k"].mean(),
            "avg_recall@k": metrics_df["recall@k"].mean(),
            "avg_mrr": metrics_df["mrr"].mean(),
            "avg_ndcg@k": metrics_df["ndcg@k"].mean(),
            "avg_ap@k": metrics_df["ap@k"].mean(),
            "avg_retrieval_time": metrics_df["retrieval_time"].mean(),
            "total_input_tokens": metrics_df["input_tokens"].sum(),
            "by_question": metrics_df.to_dict(orient="records"),
        }

    def save_reports(self, metrics, prefix="BASELINE_eval"):
        if metrics is None or metrics["num_queries"] == 0:
            print("No metrics to save")
            return

        def convert(o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            return o

        metrics_file = os.path.join(METRICS_DIR, f"{prefix}_singlehop.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(
                {k: convert(v) for k, v in metrics.items() if k != "by_question"},
                f,
                indent=2,
                ensure_ascii=False,
            )
        # details_file = os.path.join(METRICS_DIR, f"{prefix}_details_by_query.csv")
        # df = pd.DataFrame(metrics["by_question"])
        # df.to_csv(details_file, index=False)

def main(test_file, top_k=5):
    mistral_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        print("Error: MISTRAL_API_KEY not found in environment variables")
        return

    evaluator = RAGEvaluator(mistral_api_key=mistral_key, batch_size=32, device="cpu")

    print("Starting evaluation...")
    metrics = evaluator.evaluate_retrieval(test_file, top_k)

    if not metrics or metrics["num_queries"] == 0:
        print("Evaluation failed or no queries processed")
        return

    print("\n=== Evaluation Results (w/ additional metrics)===")
    print(f"Queries evaluated: {metrics['num_queries']}")
    print(f"Avg Precision@{top_k}: {metrics['avg_precision@k']:.2%}")
    print(f"Avg Recall@{top_k}: {metrics['avg_recall@k']:.2%}")
    print(f"Mean Reciprocal Rank: {metrics['avg_mrr']:.4f}")
    print(f"Avg nDCG@{top_k}: {metrics['avg_ndcg@k']:.4f}")
    print(f"Avg AP@{top_k}: {metrics['avg_ap@k']:.4f}")
    print(f"Avg Retrieval Time: {metrics['avg_retrieval_time']:.4f}s")
    print(f"Total Input Tokens: {metrics['total_input_tokens']}")

    evaluator.save_reports(metrics)

if __name__ == "__main__":
    test_file = os.path.join(DATA_DIR, "test", "test_dataset_singlehop.json")
    main(test_file, top_k=5)
