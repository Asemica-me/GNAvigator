import json
import os
import time
import random
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv

from rag_sys import RAGOrchestrator

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GEN_MODEL = os.getenv("GEN_MODEL")

OUTPUT_FILE = os.path.join(DATA_DIR, "test", "test_dataset_multihop.json")

METRICS_DIR = os.path.join("data", "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)

def load_and_shuffle_datasets(singlehop_path, multihop_path):
    with open(singlehop_path, encoding="utf-8") as f1:
        singlehop = json.load(f1)
    with open(multihop_path, encoding="utf-8") as f2:
        multihop = json.load(f2)
    combined = singlehop + multihop
    random.shuffle(combined)
    return combined


class RAGEvaluator:
    def __init__(self, mistral_api_key: str, batch_size: int = 32, device: str = None):
        self.orchestrator = RAGOrchestrator(
            mistral_api_key=mistral_api_key, device=device
        )
        self.metrics = []
        self.batch_size = batch_size

    def evaluate_retrieval(self, test_file: str, top_k: int = 5):
        """Run evaluation on test dataset"""
        try:
            with open(test_file, encoding="utf-8") as f:
                test_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Test file not found at {test_file}")
            return None
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {test_file}")
            return None

        # Run evaluations concurrently
        results = []
        # Iterate over batches of test data
        batch_size = min(self.batch_size, len(test_data))
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i : i + batch_size]
            questions_batch = [item["question"] for item in batch]
            relevant_docs_batch = [set(item["relevant_docs"]) for item in batch]
            results.extend(
                self._evaluate_query_batch(questions_batch, relevant_docs_batch, top_k)
            )

        self.metrics = results
        return self._aggregate_metrics()

    def _evaluate_query(self, question: str, relevant_docs: set, top_k: int):
        """Evaluate a single query"""
        try:
            # Time the retrieval
            start_time = time.time()
            retrieved_docs = self.orchestrator.retrieve_docs(question, top_k)
            retrieval_time = time.time() - start_time

            # Get document IDs
            retrieved_ids = (
                {doc["id"] for doc in retrieved_docs} if retrieved_docs else set()
            )

            # Calculate metrics
            relevant_retrieved = len(retrieved_ids & relevant_docs)
            precision = relevant_retrieved / top_k if top_k > 0 else 0
            recall = relevant_retrieved / len(relevant_docs) if relevant_docs else 0

            # Calculate MRR
            reciprocal_rank = 0
            if retrieved_docs:
                for rank, doc in enumerate(retrieved_docs, 1):
                    if doc["id"] in relevant_docs:
                        reciprocal_rank = 1 / rank
                        break

            # Token usage
            input_tokens = (
                len(self.orchestrator.tokenize(question))
                if hasattr(self.orchestrator, "tokenize")
                else 0
            )

            return {
                "question": question,
                "precision@k": precision,
                "recall@k": recall,
                "mrr": reciprocal_rank,
                "retrieval_time": retrieval_time,
                "input_tokens": input_tokens,
                "retrieved_docs": list(retrieved_ids),
                "relevant_docs": list(relevant_docs),
            }

        except Exception as e:
            print(f"Error processing query '{question}': {str(e)}")
            return {
                "question": question,
                "error": str(e),
                "precision@k": 0,
                "recall@k": 0,
                "mrr": 0,
                "retrieval_time": 0,
                "input_tokens": 0,
                "retrieved_docs": [],
                "relevant_docs": list(relevant_docs),
            }

    def _evaluate_query_batch(
        self, questions: list[str], relevant_docs_batch: list[set], top_k: int
    ):
        """Evaluate a single query"""
        results = []
        try:
            # Time the retrieval
            start_time = time.time()
            retrieved_docs_batch = self.orchestrator.retrieve_docs_batch(
                questions, top_k
            )
            retrieval_time = (time.time() - start_time) / len(questions)

            for retrieved_docs, relevant_docs, question in zip(
                retrieved_docs_batch, relevant_docs_batch, questions
            ):
                # Get document IDs
                retrieved_ids = (
                    {doc["id"] for doc in retrieved_docs} if retrieved_docs else set()
                )

                # Calculate metrics
                relevant_retrieved = len(retrieved_ids & relevant_docs)
                precision = relevant_retrieved / top_k if top_k > 0 else 0
                recall = relevant_retrieved / len(relevant_docs) if relevant_docs else 0

                # Calculate MRR
                reciprocal_rank = 0
                if retrieved_docs:
                    for rank, doc in enumerate(retrieved_docs, 1):
                        if doc["id"] in relevant_docs:
                            reciprocal_rank = 1 / rank
                            break

                # Token usage
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
                        "retrieval_time": retrieval_time,
                        "input_tokens": input_tokens,
                        "retrieved_docs": list(retrieved_ids),
                        "relevant_docs": list(relevant_docs),
                    }
                )

            return results

        except Exception:
            return []

    def _aggregate_metrics(self):
        """Calculate aggregate statistics"""
        if not self.metrics:
            return {
                "num_queries": 0,
                "avg_precision@k": 0,
                "avg_recall@k": 0,
                "avg_mrr": 0,
                "avg_retrieval_time": 0,
                "total_input_tokens": 0,
                "by_question": [],
            }

        df = pd.DataFrame(self.metrics)

        # Handle cases with NaN values
        metrics_df = df.fillna(
            {
                "precision@k": 0,
                "recall@k": 0,
                "mrr": 0,
                "retrieval_time": 0,
                "input_tokens": 0,
            }
        )

        return {
            "num_queries": len(metrics_df),
            "avg_precision@k": metrics_df["precision@k"].mean(),
            "avg_recall@k": metrics_df["recall@k"].mean(),
            "avg_mrr": metrics_df["mrr"].mean(),
            "avg_retrieval_time": metrics_df["retrieval_time"].mean(),
            "total_input_tokens": metrics_df["input_tokens"].sum(),
            "by_question": metrics_df.to_dict(orient="records"),
        }

    def save_reports(self, metrics, prefix="BASELINE_eval"):
        """Save all reports to metrics directory"""
        if metrics is None or metrics["num_queries"] == 0:
            print("No metrics to save")
            return

        # Convert numpy types to native Python types
        def convert(o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            return o

        # Save aggregate metrics
        metrics_file = os.path.join(METRICS_DIR, f"{prefix}_single_multi.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(
                {k: convert(v) for k, v in metrics.items() if k != "by_question"},
                f,
                indent=2,
                ensure_ascii=False,
            )

        # Save detailed results as CSV
        # details_file = os.path.join(METRICS_DIR, f"{prefix}_details.csv")
        # df = pd.DataFrame(metrics["by_question"])
        # df.to_csv(details_file, index=False)

        # # Save error analysis
        # errors_file = os.path.join(METRICS_DIR, f"{prefix}_errors.csv")
        # error_df = df[(df["precision@k"] < 0.3) | (df["recall@k"] < 0.3)]
        # if not error_df.empty:
        #     error_df.to_csv(errors_file, index=False)

        # print(f"\nReports saved to {METRICS_DIR}:")
        # print(f"- {prefix}_metrics.json (aggregate metrics)")
        # print(f"- {prefix}_details.csv (per-query results)")
        # if not error_df.empty:
        #     print(f"- {prefix}_errors.csv ({len(error_df)} problematic queries)")

    def evaluate_retrieval_from_data(self, test_data, top_k: int = 5):
        """Run evaluation on test dataset passed as list of dicts"""
        if not test_data:
            print("Error: No data provided.")
            return None

        # Run evaluations concurrently
        results = []
        batch_size = min(self.batch_size, len(test_data))
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i : i + batch_size]
            questions_batch = [item["question"] for item in batch]
            relevant_docs_batch = [set(item["relevant_docs"]) for item in batch]
            results.extend(
                self._evaluate_query_batch(questions_batch, relevant_docs_batch, top_k)
            )

        self.metrics = results
        return self._aggregate_metrics()


def main(test_file, top_k=5):
 
    SINGLEHOP_FILE = os.path.join(DATA_DIR, "test", "test_dataset_singlehop.json")
    MULTIHOP_FILE = os.path.join(DATA_DIR, "test", "test_dataset_multihop.json")

    # 1. Load and shuffle
    test_data = load_and_shuffle_datasets(SINGLEHOP_FILE, MULTIHOP_FILE)
    # 2. Evaluate
    mistral_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        print("Error: MISTRAL_API_KEY not found in environment variables")
        exit(1)

    evaluator = RAGEvaluator(mistral_api_key=mistral_key, batch_size=32, device="cpu")

    print("Starting evaluation on combined, shuffled dataset...")
    metrics = evaluator.evaluate_retrieval_from_data(test_data, top_k=5)

    if not metrics or metrics["num_queries"] == 0:
        print("Evaluation failed or no queries processed")
        exit(1)

    print("\n=== Evaluation Results (w/ multihop data) ===")
    print(f"Queries evaluated: {metrics['num_queries']}")
    print(f"Avg Precision@5: {metrics['avg_precision@k']:.2%}")
    print(f"Avg Recall@5: {metrics['avg_recall@k']:.2%}")
    print(f"Mean Reciprocal Rank: {metrics['avg_mrr']:.4f}")
    print(f"Avg Retrieval Time: {metrics['avg_retrieval_time']:.4f}s")
    print(f"Total Input Tokens: {metrics['total_input_tokens']}")

    evaluator.save_reports(metrics, prefix="BASELINE_eval")


if __name__ == "__main__":
    test_file = OUTPUT_FILE
    main(test_file, top_k=5)