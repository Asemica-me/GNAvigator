import numpy as np
import nltk
from rank_bm25 import BM25Okapi
import re
import logging
from collections import defaultdict
from functools import lru_cache
import time
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Set, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vector_store import VectorDatabaseManager
from dotenv import load_dotenv

import pandas as pd
from datetime import datetime

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# -------------------------------
# Paths
# -------------------------------
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent  # ablation/
DATA_DIR = root_dir / "data"
METRICS_DIR = DATA_DIR / "metrics"
ABLATION_METRICS_DIR = METRICS_DIR / "ablation_metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
ABLATION_METRICS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Helpers (metadata)
# -------------------------------
def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _describe_bm25(bm: "BM25Retriever") -> Dict[str, Any]:
    return {
        "type": "bm25",
        "use_stopwords": getattr(bm, "use_stopwords", None),
        "use_stemming": getattr(bm, "use_stemming", None),
        "k1": getattr(bm, "k1", None),
        "b": getattr(bm, "b", None),
        "corpus": getattr(bm, "corpus_stats", None),
    }

# -------------------------------
# Vector wrapper
# -------------------------------
class VectorDatabaseWrapper:
    def __init__(self, vector_db: VectorDatabaseManager):
        self.vector_db = vector_db
        self._cached_embeddings = {}
        # Expose metadata_db for access
        self.metadata_db = self.vector_db.metadata_db

    def query(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # Check cache first
        if question in self._cached_embeddings:
            return self._cached_embeddings[question]
        # Query database and cache results
        results = self.vector_db.query(question, top_k) or []
        self._cached_embeddings[question] = results
        return results

# -------------------------------
# BM25 retriever
# -------------------------------
class BM25Retriever:
    def __init__(self, use_stopwords=True, use_stemming=True, k1=1.5, b=0.75):
        """
        BM25Retriever with configurable parameters
        """
        self.vector_db = VectorDatabaseWrapper(VectorDatabaseManager())
        # Concatenate selected fields from each chunk for the BM25 corpus
        self.chunks = [
            self._concat_fields(meta)
            for meta in self.vector_db.metadata_db
        ]
        self.metadata_list = [meta for meta in self.vector_db.metadata_db]
        self.use_stopwords = use_stopwords
        self.use_stemming = use_stemming
        self.k1 = k1
        self.b = b

        # Log + store corpus statistics
        if self.chunks:
            chunk_lengths = [len(c.split()) for c in self.chunks]
            self.corpus_stats = {
                "num_chunks": len(self.chunks),
                "min_len": int(min(chunk_lengths)),
                "max_len": int(max(chunk_lengths)),
                "avg_len": float(np.mean(chunk_lengths)),
            }
            logger.info(
                "Loaded %d chunks | len min=%d max=%d avg=%.1f words",
                self.corpus_stats["num_chunks"],
                self.corpus_stats["min_len"],
                self.corpus_stats["max_len"],
                self.corpus_stats["avg_len"],
            )
        else:
            self.corpus_stats = {"num_chunks": 0, "min_len": 0, "max_len": 0, "avg_len": 0.0}
            logger.warning("Loaded 0 chunks — check metadata_db.")

        # Initialize Italian linguistic resources
        self.stop_words = set(nltk.corpus.stopwords.words('italian')) if use_stopwords else set()
        self.stemmer = nltk.stem.SnowballStemmer("italian") if use_stemming else None

        # Preprocess chunks
        self.tokenized_chunks = [self._preprocess(chunk) for chunk in self.chunks]
        self.bm25 = BM25Okapi(
            self.tokenized_chunks,
            k1=self.k1,
            b=self.b
        )
        self.chunk_map = defaultdict(list)
        for idx, tokens in enumerate(self.tokenized_chunks):
            self.chunk_map[tuple(tokens)].append(idx)

    def _concat_fields(self, meta: Dict[str, Any]) -> str:
        """
        Concatenate title, keywords, headers_context, content.
        """
        title = meta.get('title', '')
        keywords = ' '.join(meta.get('keywords', []))
        headers = ' '.join(meta.get('headers_context', []))
        content = meta.get('document', '')
        return f"{title} {keywords} {headers} {content}"

    def _preprocess(self, text: str) -> List[str]:
        """Text preprocessing pipeline"""
        text = re.sub(r"[^\w\s']", "", text)
        text = re.sub(r"\b(l'|un'|all'|d'|dell'|quest'|nell')\b", "", text)
        text = text.lower()

        tokens = nltk.word_tokenize(text, language='italian')
        processed = []

        for token in tokens:
            if len(token) < 2:
                continue
            if self.use_stopwords and token in self.stop_words:
                continue
            if self.use_stemming:
                token = self.stemmer.stem(token)
            processed.append(token)
        return processed

    @lru_cache(maxsize=1000)
    def _preprocess_query(self, query: str) -> List[str]:
        return self._preprocess(query)

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        tokenized_query = self._preprocess_query(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])][::-1]
        results = []
        for idx in top_indices:
            result = {
                "score": float(scores[idx]),
                "text": self.chunks[idx],
                **self.metadata_list[idx]
            }
            results.append(result)
        return results

    def batch_retrieve(self, queries: List[str], k: int = 5) -> List[List[Dict[str, Any]]]:
        tokenized_queries = [self._preprocess_query(q) for q in queries]
        batch_scores = np.array([self.bm25.get_scores(q) for q in tokenized_queries])
        results = []
        for scores in batch_scores:
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])][::-1]
            batch_result = []
            for idx in top_indices:
                result = {
                    "score": float(scores[idx]),
                    "text": self.chunks[idx],
                    **self.metadata_list[idx]
                }
                batch_result.append(result)
            results.append(batch_result)
        return results

    def query_with_scores(self, question: str, top_k: int = 20) -> List[Dict[str, Any]]:
        return self.retrieve(question, k=top_k)

    def batch_query_with_scores(self, questions: List[str], top_k: int = 20) -> List[List[Dict[str, Any]]]:
        return self.batch_retrieve(questions, k=top_k)

# === EVALUATION LOGIC  ===

def compute_ndcg(relevant_ids: Set[str], retrieved_docs: List[Dict], k: int = 5) -> float:
    if k <= 0 or not retrieved_docs:
        return 0.0
    top_k = retrieved_docs[:k]
    y_true = [1.0 if doc.get('id', None) in relevant_ids else 0.0 for doc in top_k]
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
        if doc.get('id', None) in relevant_ids:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)
    return precision_sum / min(len(relevant_ids), k)

class RetrieverEvaluator:
    def __init__(self, retriever: BM25Retriever, batch_size: int = 32):
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
            retrieved_ids = [doc.get("id", None) for doc in retrieved_docs]
            relevant_retrieved = len(set(retrieved_ids) & relevant_set)
            precision = relevant_retrieved / top_k if top_k > 0 else 0
            recall = relevant_retrieved / len(relevant_set) if relevant_set else 0
            reciprocal_rank = 0
            for rank, doc in enumerate(retrieved_docs, 1):
                if doc.get("id", None) in relevant_set:
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
                "retrieval_time": retrieval_time
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

def load_and_shuffle_datasets(singlehop_path, multihop_path):
    with open(singlehop_path, encoding="utf-8") as f1:
        singlehop = json.load(f1)
    with open(multihop_path, encoding="utf-8") as f2:
        multihop = json.load(f2)
    combined = singlehop + multihop
    random.shuffle(combined)
    return combined

def main(top_k: int = 5):
    retriever = BM25Retriever()
    evaluator = RetrieverEvaluator(retriever, batch_size=32)

    SINGLEHOP_FILE = DATA_DIR / "test" / "test_dataset_singlehop.json"
    MULTIHOP_FILE = DATA_DIR / "test" / "test_dataset_multihop.json"

    # common run metadata
    run_meta_common = {
        "experiment_prefix": "BM25_eval",
        "retrieval_top_k": top_k,
        "bm25": _describe_bm25(retriever),
    }

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
            prefix="BM25_eval_singlehop",
            metadata={**run_meta_common, "dataset": "singlehop"}
        )
        print(f"Reports saved to {ABLATION_METRICS_DIR}")
    else:
        print("Singlehop evaluation failed.")

    print("\n=== Evaluating SINGLE+MULTIHOP dataset ===")
    combined_data = load_and_shuffle_datasets(SINGLEHOP_FILE, MULTIHOP_FILE)
    temp_combined_file = DATA_DIR / "test" / "test_dataset_combined_temp.json"
    try:
        with open(temp_combined_file, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)
        all_metrics = evaluator.evaluate_retrieval(temp_combined_file, top_k)
    finally:
        try:
            temp_combined_file.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Could not delete temporary file: {str(e)}")

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
            prefix="BM25_eval_combined",
            metadata={**run_meta_common, "dataset": "combined", "combined_sources": ["singlehop", "multihop"], "shuffle": True}
        )
        print(f"Reports saved to {ABLATION_METRICS_DIR}")
    else:
        print("Combined evaluation failed.")

if __name__ == "__main__":
    main(top_k=5)
