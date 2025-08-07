from ablation import all_experiments
import os
import random
import pandas as pd
import json
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

def load_and_shuffle_datasets(singlehop_path, multihop_path):
    with open(singlehop_path, encoding="utf-8") as f1:
        singlehop = json.load(f1)
    with open(multihop_path, encoding="utf-8") as f2:
        multihop = json.load(f2)
    combined = singlehop + multihop
    random.shuffle(combined)
    return combined

def compute_ndcg(relevant_ids, retrieved_ids, k=5):
    y_true = [1 if rid in relevant_ids else 0 for rid in retrieved_ids[:k]]
    y_score = [k - i for i in range(len(retrieved_ids[:k]))]
    return ndcg_score([y_true], [y_score]) if any(y_true) else 0.0

def compute_ap(relevant_ids, retrieved_ids, k=5):
    y_true = [1 if rid in relevant_ids else 0 for rid in retrieved_ids[:k]]
    return average_precision_score([1 if i in relevant_ids else 0 for i in retrieved_ids[:k]], y_true) if any(y_true) else 0.0


class AblationOrchestrator(RAGOrchestrator):
    def __init__(self, mistral_api_key: str, retriever, device: str = None):
        super().__init__(mistral_api_key, device=device)
        self.custom_retriever = retriever

    async def retrieve_docs(self, question: str, top_k: int = 5):
        return self.custom_retriever.retrieve(question, top_k=top_k)

    def retrieve_docs_batch(self, questions: list, top_k: int = 5):
        return [self.custom_retriever.retrieve(q, top_k=top_k) for q in questions]


class RAGEvaluator:
    def __init__(self, mistral_api_key: str, batch_size: int = 32, device: str = None, orchestrator_class=RAGOrchestrator, retriever=None):
        if retriever:
            self.orchestrator = orchestrator_class(mistral_api_key, retriever, device=device)
        else:
            self.orchestrator = RAGOrchestrator(mistral_api_key=mistral_api_key, device=device)
        self.batch_size = batch_size

    def evaluate_retrieval(self, test_file: str, top_k: int = 5):
        with open(test_file, encoding="utf-8") as f:
            test_data = json.load(f)

        results = []
        batch_size = min(self.batch_size, len(test_data))
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i: i + batch_size]
            questions_batch = [item["question"] for item in batch]
            relevant_docs_batch = [set(item["relevant_docs"]) for item in batch]
            results.extend(self._evaluate_query_batch(questions_batch, relevant_docs_batch, top_k))

        return self._aggregate_metrics(results)

    def _evaluate_query_batch(self, questions, relevant_docs_batch, top_k):
        results = []
        start_time = time.time()
        retrieved_docs_batch = self.orchestrator.retrieve_docs_batch(questions, top_k)
        retrieval_time = (time.time() - start_time) / len(questions)

        for retrieved_docs, relevant_docs in zip(retrieved_docs_batch, relevant_docs_batch):
            retrieved_ids = set()
            for doc in retrieved_docs:
                # Handle all possible ID field names
                if 'id' in doc:
                    retrieved_ids.add(str(doc['id']))
                elif 'chunk_id' in doc:
                    retrieved_ids.add(str(doc['chunk_id']))

            relevant_docs = {str(x) for x in relevant_docs} 
            
            relevant_retrieved = len(retrieved_ids & relevant_docs)
            precision = relevant_retrieved / top_k if top_k > 0 else 0
            recall = relevant_retrieved / len(relevant_docs) if relevant_docs else 0

            reciprocal_rank = 0
            for rank, doc in enumerate(retrieved_docs, 1):
                doc_id = None
                if 'id' in doc:
                    doc_id = doc['id']
                elif 'chunk_id' in doc:
                    doc_id = doc['chunk_id']
                
                if doc_id and doc_id in relevant_docs:
                    reciprocal_rank = 1 / rank
                    break

            ndcg = compute_ndcg(relevant_docs, retrieved_ids, k=top_k)
            ap = compute_ap(relevant_docs, retrieved_ids, k=top_k)


            results.append({
                "precision@k": precision,
                "recall@k": recall,
                "mrr": reciprocal_rank,
                "ndcg@k": ndcg,
                "ap@k": ap,
                "retrieval_time": retrieval_time,
                "input_tokens": len(self.orchestrator.tokenize(questions[0])) if hasattr(self.orchestrator, "tokenize") else 0,
            })

        return results

    def _aggregate_metrics(self, metrics):
        if not metrics:
            return None
        df = pd.DataFrame(metrics)
        return {
            "avg_precision@k": df["precision@k"].mean(),
            "avg_recall@k": df["recall@k"].mean(),
            "avg_mrr": df["mrr"].mean(),
            "avg_ndcg@k": df["ndcg@k"].mean(),
            "avg_ap@k": df["ap@k"].mean(),
            "avg_retrieval_time": df["retrieval_time"].mean(),
            "total_input_tokens": df["input_tokens"].sum(),
        }


def ablation_main(test_file, top_k=5):
    SINGLEHOP_FILE = os.path.join(DATA_DIR, "test", "test_dataset_singlehop.json")
    MULTIHOP_FILE = os.path.join(DATA_DIR, "test", "test_dataset_multihop.json")

    # 1. Load and shuffle
    test_data = load_and_shuffle_datasets(SINGLEHOP_FILE, MULTIHOP_FILE)

    mistral_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        print("MISTRAL_API_KEY not set.")
        return

    retriever_confs = []

    # 1. DenseRetriever
    retriever_confs.append(("dense", all_experiments.DenseRetriever()))

    # 2. BM25Retriever
    retriever_confs.append(("bm25", all_experiments.BM25Retriever()))

    # 3. HybridRetriever: Try different rrf_k values
    for rrf_k in [30, 60]:
        retriever_confs.append((f"hybrid_rrf{rrf_k}", all_experiments.HybridRetriever(rrf_k=rrf_k)))

    # 4. RerankRetriever: with both Dense and BM25 as base
    for base_name, base in [("dense", all_experiments.DenseRetriever()), ("bm25", all_experiments.BM25Retriever())]:
        retriever_confs.append((f"rerank_{base_name}", all_experiments.RerankRetriever(base_retriever=base)))

    # 5. QueryRewriteRetriever: with both Dense and BM25 as base, and different expansion_terms
    for base_name, base in [("dense", all_experiments.DenseRetriever()), ("bm25", all_experiments.BM25Retriever())]:
        for expansion_terms in [1, 3, 5]:
            retriever_confs.append(
                (f"qrewrite_{base_name}_exp{expansion_terms}",
                 all_experiments.QueryRewriteRetriever(base_retriever=base, expansion_terms=expansion_terms))
            )

    all_metrics = []

    for name, retriever in retriever_confs:
        print(f"\nRunning ablation for: {name}")
        evaluator = RAGEvaluator(
            mistral_api_key=mistral_key,
            batch_size=32,
            device="cpu",
            orchestrator_class=AblationOrchestrator,
            retriever=retriever
        )

        metrics = evaluator.evaluate_retrieval(test_data, top_k)
        if not metrics:
            print(f"No metrics for {name}")
            continue

        metrics["method"] = name
        print(f"Method: {name}, Metrics: {metrics}")
        all_metrics.append(metrics)

    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df = metrics_df[["method", "avg_precision@k", "avg_recall@k", "avg_mrr", "avg_ndcg@k", "avg_ap@k", "avg_retrieval_time", "total_input_tokens"]]
        metrics_df.to_csv(os.path.join(METRICS_DIR, "ablation_results.csv"), index=False)
        print("\nAll metrics saved to 'ablation_results.csv'.")


if __name__ == "__main__":
    test_file = os.path.join(DATA_DIR, "test", "test_dataset.json")
    ablation_main(test_file, top_k=5)
