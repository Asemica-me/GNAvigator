# ablation/experiments.py

import os
import sys
import json
import time
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple

from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from vector_store import VectorDatabaseManager

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent 
os.chdir(root_dir)
DATA_DIR = root_dir / "data"
METRICS_DIR = DATA_DIR / "metrics"
ABLATION_METRICS_DIR = METRICS_DIR / "ablation_metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
ABLATION_METRICS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Helpers (metadata, time)
# -------------------------------
def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _describe_dense(retriever: "DenseRetriever") -> Dict[str, Any]:
    # Try to pull device from VectorDatabaseManager if exposed; otherwise env/CPU
    device = os.getenv("DEVICE", "cpu")
    try:
        mgr = retriever.vector_db.vector_db  # underlying VectorDatabaseManager
        if hasattr(mgr, "device") and mgr.device:
            device = mgr.device
    except Exception:
        pass
    return {"type": "dense", "device": device}

def _check_ready(mgr: "VectorDatabaseManager") -> bool:
    """
    Best-effort sanity checks for common backends.
    - FAISS: ensure .index exists
    - ES/OpenSearch: ensure .client exists
    """
    ok = True
    if hasattr(mgr, "index"):
        if getattr(mgr, "index", None) is None:
            ok = False
            logger.error("FAISS index is None. Did you build/load it correctly?")
    if hasattr(mgr, "client"):
        if getattr(mgr, "client", None) is None:
            ok = False
            logger.error("Elasticsearch/OpenSearch client is None. Check URL/creds/index.")
    if getattr(mgr, "metadata_db", None) in (None, []):
        logger.warning("metadata_db is empty or None. ID→doc mapping may be missing.")
    return ok

# -------------------------------
# Vector wrapper + retriever
# -------------------------------
class VectorDatabaseWrapper:
    def __init__(self, vector_db: VectorDatabaseManager):
        self.vector_db = vector_db
        self._cached_embeddings: Dict[str, List[Dict[str, Any]]] = {}

    def query(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # Cache
        if question in self._cached_embeddings:
            return self._cached_embeddings[question]

        try:
            results = self.vector_db.query(question, top_k) or []
        except AttributeError:
            logger.exception(
                "Vector DB backend not initialized (AttributeError). "
                "Check FAISS/ES setup and index/client initialization."
            )
            results = []
        except Exception as e:
            logger.exception("Query failed: %s", e)
            results = []

        self._cached_embeddings[question] = results
        return results

    def query_batch(self, questions: List[str], top_k: int = 5) -> List[List[Dict[str, Any]]]:
        """
        Robust to duplicate questions and preserves order.
        """
        results: List[Optional[List[Dict[str, Any]]]] = [None] * len(questions)
        uncached: List[Tuple[int, str]] = []

        for i, q in enumerate(questions):
            if q in self._cached_embeddings:
                results[i] = self._cached_embeddings[q]
            else:
                uncached.append((i, q))

        for i, q in uncached:
            res = self.query(q, top_k)
            results[i] = res

        return results 

class DenseRetriever:
    def __init__(self, device: Optional[str] = None):
        self.vector_db = VectorDatabaseWrapper(VectorDatabaseManager(device=device))

    def _format_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Standardize result format."""
        formatted: List[Dict[str, Any]] = []
        for item in results or []:
            # tolerate missing keys
            _id = item.get("id")
            if _id is None:
                # skip malformed entries
                continue
            formatted.append({
                "id": _id,
                "chunk_id": _id,
                "score": item.get("score", 0.0),
                **item
            })
        return formatted

    def retrieve(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.vector_db.query(question, top_k=top_k)
        return self._format_results(results)

    def query_with_scores(self, question: str, top_k: int = 20) -> List[Dict[str, Any]]:
        results = self.vector_db.query(question, top_k=top_k)
        return self._format_results(results)

# -------------------------------
# Metrics
# -------------------------------
import numpy as np
import pandas as pd

def compute_ndcg(relevant_ids: Set[str], retrieved_docs: List[Dict[str, Any]], k: int = 5) -> float:
    if k <= 0 or not retrieved_docs:
        return 0.0
    top_k = retrieved_docs[:k]
    y_true = [1.0 if doc.get("id") in relevant_ids else 0.0 for doc in top_k]
    ideal_sorted = sorted(y_true, reverse=True)
    idcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_sorted))
    dcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(y_true))
    return dcg / idcg if idcg > 0 else 0.0

def compute_ap(relevant_ids: Set[str], retrieved_docs: List[Dict[str, Any]], k: int = 5) -> float:
    if not relevant_ids or not retrieved_docs:
        return 0.0
    top_k = retrieved_docs[:k]
    relevant_count = 0
    precision_sum = 0.0
    for i, doc in enumerate(top_k):
        if doc.get("id") in relevant_ids:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)
    return precision_sum / min(len(relevant_ids), k)

# -------------------------------
# Evaluator
# -------------------------------
class RetrieverEvaluator:
    def __init__(self, retriever: DenseRetriever, batch_size: int = 32):
        self.retriever = retriever
        self.batch_size = batch_size
        self.metrics: List[Dict[str, Any]] = []

    def evaluate_retrieval(self, test_file: Path, top_k: int = 5) -> Optional[Dict[str, Any]]:
        try:
            with open(test_file, "r", encoding="utf-8") as f:
                test_data = json.load(f)
        except Exception as e:
            logger.error("Failed to load test file %s: %s", test_file, e)
            return None

        results: List[Dict[str, Any]] = []
        total_questions = len(test_data)

        for i in range(0, total_questions, self.batch_size):
            batch = test_data[i:i + self.batch_size]
            batch_results = self._evaluate_query_batch(
                questions=[item["question"] for item in batch],
                relevant_docs_batch=[set(item["relevant_docs"]) for item in batch],
                top_k=top_k
            )
            results.extend(batch_results)
            logger.info("Processed %d/%d questions", min(i + self.batch_size, total_questions), total_questions)

        self.metrics = results
        return self._aggregate_metrics()

    def _evaluate_query_batch(self, questions: List[str], relevant_docs_batch: List[Set[str]], top_k: int) -> List[Dict[str, Any]]:
        batch_results: List[Dict[str, Any]] = []
        start_time = time.perf_counter()
        retrieved_docs_batch = [self.retriever.retrieve(q, top_k) for q in questions]
        retrieval_time = (time.perf_counter() - start_time) / max(1, len(questions))

        for i, (retrieved_docs, relevant_set) in enumerate(zip(retrieved_docs_batch, relevant_docs_batch)):
            retrieved_docs = retrieved_docs or []
            retrieved_ids = [doc.get("id") for doc in retrieved_docs if doc.get("id") is not None]

            relevant_retrieved = len(set(retrieved_ids) & relevant_set)
            precision = relevant_retrieved / top_k if top_k > 0 else 0.0
            recall = relevant_retrieved / len(relevant_set) if relevant_set else 0.0

            reciprocal_rank = 0.0
            for rank, doc in enumerate(retrieved_docs, 1):
                if doc.get("id") in relevant_set:
                    reciprocal_rank = 1.0 / rank
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
                "avg_precision@k": 0.0,
                "avg_recall@k": 0.0,
                "avg_mrr": 0.0,
                "avg_ndcg@k": 0.0,
                "avg_ap@k": 0.0,
                "avg_retrieval_time": 0.0
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

    def save_reports(self, metrics: Dict[str, Any], prefix: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save aggregated metrics with rich metadata."""
        if not metrics or metrics.get("num_queries", 0) == 0:
            print("No metrics to save")
            return

        meta = {
            "timestamp_utc": _now_iso(),
            "batch_size_eval": self.batch_size,
        }
        if metadata:
            meta.update(metadata)

        payload = {
            "metadata": meta,
            "metrics": metrics
        }

        metrics_file = ABLATION_METRICS_DIR / f"{prefix}_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

# -------------------------------
# Dataset utils
# -------------------------------
def load_and_shuffle_datasets(singlehop_path: Path, multihop_path: Path) -> List[Dict[str, Any]]:
    with open(singlehop_path, encoding="utf-8") as f1:
        singlehop = json.load(f1)
    with open(multihop_path, encoding="utf-8") as f2:
        multihop = json.load(f2)
    combined = singlehop + multihop
    random.shuffle(combined)
    return combined

# -------------------------------
# Main
# -------------------------------
def main(top_k: int = 5) -> None:
    device = os.getenv("DEVICE", "cpu")
    retriever = DenseRetriever(device=device)

    # Preflight checks for backend readiness
    try:
        mgr = retriever.vector_db.vector_db  # underlying VectorDatabaseManager
        if not _check_ready(mgr):
            raise SystemExit("Vector DB not ready. See error logs above.")
    except Exception as e:
        logger.exception("Failed to access underlying VectorDatabaseManager: %s", e)
        raise SystemExit(1)

    evaluator = RetrieverEvaluator(retriever, batch_size=32)

    SINGLEHOP_FILE = DATA_DIR / "test" / "test_dataset_singlehop.json"
    MULTIHOP_FILE = DATA_DIR / "test" / "test_dataset_multihop.json"

    run_meta_common = {
        "experiment_prefix": "dense_eval",
        "retrieval_top_k": top_k,
        "retriever": _describe_dense(retriever),
    }

    # 1) SINGLEHOP
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
        evaluator.save_reports(
            singlehop_metrics,
            prefix="DENSE_eval_singlehop",
            metadata={**run_meta_common, "dataset": "singlehop"}
        )
        print(f"Reports saved to {ABLATION_METRICS_DIR}")
    else:
        print("Singlehop evaluation failed.")

    # 2) COMBINED (singlehop + multihop, shuffled)
    print("\n=== Evaluating SINGLE+MULTIHOP dataset ===")
    combined_data = load_and_shuffle_datasets(SINGLEHOP_FILE, MULTIHOP_FILE)
    temp_combined_file = DATA_DIR / "test" / "test_dataset_combined_temp.json"

    try:
        with open(temp_combined_file, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)

        all_metrics = evaluator.evaluate_retrieval(temp_combined_file, top_k)
    finally:
        try:
            # Python 3.8+: missing_ok
            temp_combined_file.unlink(missing_ok=True)
        except Exception as e:
            logger.warning("Could not delete temporary file: %s", e)

    if all_metrics and all_metrics["num_queries"] > 0:
        print(f"Combined: Queries evaluated: {all_metrics['num_queries']}")
        print(f"Avg Precision@{top_k}: {all_metrics['avg_precision@k']:.4f}")
        print(f"Avg Recall@{top_k}: {all_metrics['avg_recall@k']:.4f}")
        print(f"Mean Reciprocal Rank: {all_metrics['avg_mrr']:.4f}")
        print(f"Avg nDCG@{top_k}: {all_metrics['avg_ndcg@k']:.4f}")
        print(f"Avg AP@{top_k}: {all_metrics['avg_ap@k']:.4f}")
        print(f"Avg Retrieval Time: {all_metrics['avg_retrieval_time']:.6f}s")
        evaluator.save_reports(
            all_metrics,
            prefix="DENSE_eval_combined",
            metadata={**run_meta_common, "dataset": "combined", "combined_sources": ["singlehop", "multihop"], "shuffle": True}
        )
        print(f"Reports saved to {ABLATION_METRICS_DIR}")
    else:
        print("Combined evaluation failed.")

if __name__ == "__main__":
    main(top_k=5)
