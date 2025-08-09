# ablation/experiments.py
import sys
import os
from pathlib import Path
import numpy as np
from collections import defaultdict
import logging
import hashlib
from typing import List, Dict, Any, Set, Optional
import nltk
from rank_bm25 import BM25Okapi
import re
from functools import lru_cache
import json
import time
import random
from datetime import datetime
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vector_store import VectorDatabaseManager
from dotenv import load_dotenv

load_dotenv()
current_dir = Path(__file__).resolve().parent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# -------------------------------
# Paths
# -------------------------------

DATA_DIR = PROJECT_ROOT / "data"  
METRICS_DIR = DATA_DIR / "metrics"
ABLATION_METRICS_DIR = METRICS_DIR / "ablation_metrics_alpha_hybrid"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
ABLATION_METRICS_DIR.mkdir(parents=True, exist_ok=True)

_probe = VectorDatabaseManager()
print("CWD:", Path.cwd())
print("Expecting data under:", DATA_DIR)
print("Probe metadata entries:", len(_probe.metadata_db or []))

# -------------------------------
# Helpers (metadata/time)
# -------------------------------
def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _describe_bm25(bm: "BM25Retriever") -> Dict[str, Any]:
    # include basic BM25 knobs + corpus stats if you want
    try:
        chunk_lengths = [len(c.split()) for c in getattr(bm, "chunks", [])]
        corpus = {
            "num_chunks": len(bm.chunks),
            "min_len": int(min(chunk_lengths)) if chunk_lengths else 0,
            "max_len": int(max(chunk_lengths)) if chunk_lengths else 0,
            "avg_len": float(np.mean(chunk_lengths)) if chunk_lengths else 0.0,
        }
    except Exception:
        corpus = None
    return {
        "type": "bm25",
        "use_stopwords": getattr(bm, "use_stopwords", None),
        "use_stemming": getattr(bm, "use_stemming", None),
        "k1": getattr(bm, "k1", None),
        "b": getattr(bm, "b", None),
        "corpus": corpus,
    }

def _describe_dense(dense: "DenseRetriever") -> Dict[str, Any]:
    device = os.getenv("DEVICE", "cpu")
    try:
        mgr = dense.vector_db.vector_db  # underlying VectorDatabaseManager
        if hasattr(mgr, "device") and mgr.device:
            device = mgr.device
    except Exception:
        pass
    return {"type": "dense", "device": device}

def _describe_hybrid(hr: "HybridRetriever") -> Dict[str, Any]:
    return {
        "type": "hybrid",
        "rrf_k": getattr(hr, "rrf_k", None),
        "dense_weight": getattr(hr, "dense_weight", None),
        "sparse_weight": getattr(hr, "sparse_weight", None),
        "dense": _describe_dense(getattr(hr, "dense")),
        "bm25": _describe_bm25(getattr(hr, "bm25")),
    }

# -------------------------------
# Vector wrapper + retrievers
# -------------------------------
class VectorDatabaseWrapper:
    def __init__(self, vector_db: VectorDatabaseManager):
        self.vector_db = vector_db
        self._cached_embeddings = {}
        
    def query(self, question: str, top_k: int = 5) -> list:
        """Query with caching using question hash to handle similar queries"""
        query_hash = self._hash_query(question)
        if query_hash in self._cached_embeddings:
            return self._cached_embeddings[query_hash]
        
        results = self.vector_db.query(question, top_k) or []
        self._cached_embeddings[query_hash] = results
        return results

    @property
    def metadata_db(self):
        return self.vector_db.metadata_db
    
    def get_document_ids(self):
        """Return a list of all document IDs from metadata_db."""
        if not hasattr(self.vector_db, "metadata_db") or self.vector_db.metadata_db is None:
            return []
        return [item.get("id") for item in self.vector_db.metadata_db if "id" in item]
    
    def query_batch(self, questions: List[str], top_k: int = 5) -> List[List]:
        """Batch query with caching and parallelization support"""
        query_hashes = [self._hash_query(q) for q in questions]
        results = []
        to_fetch = []
        
        # Check cache first
        for idx, q_hash in enumerate(query_hashes):
            if q_hash in self._cached_embeddings:
                results.append((idx, self._cached_embeddings[q_hash]))
            else:
                to_fetch.append((idx, questions[idx]))
        
        # Fetch missing queries in batch
        if to_fetch:
            batch_questions = [q for _, q in to_fetch]
            batch_results = self.vector_db.query_batch(batch_questions, top_k) or [[] for _ in to_fetch]
            
            for (idx, _), batch_res in zip(to_fetch, batch_results):
                q_hash = query_hashes[idx]
                self._cached_embeddings[q_hash] = batch_res
                results.append((idx, batch_res))
        
        # Sort results by original index
        results.sort(key=lambda x: x[0])
        return [res for _, res in results]

    def _hash_query(self, query: str) -> str:
        """Create consistent hash for similar queries"""
        normalized = " ".join(query.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()
    
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
    
class BM25Retriever:
    def __init__(self, use_stopwords=True, use_stemming=True, k1=1.5, b=0.75):
        """
        BM25Retriever with configurable parameters
        """
        self.vector_db = VectorDatabaseWrapper(VectorDatabaseManager())
        # Concatenate selected fields from each chunk for the BM25 corpus
        self.chunks = [self._concat_fields(meta) for meta in self.vector_db.metadata_db]
        self.metadata_list = [meta for meta in self.vector_db.metadata_db]
        self.use_stopwords = use_stopwords
        self.use_stemming = use_stemming
        self.k1 = k1
        self.b = b
        
        # Log corpus statistics
        if self.chunks:
            chunk_lengths = [len(c.split()) for c in self.chunks]
            logger.info(
                "Loaded %d chunks | len min=%d max=%d avg=%.1f words",
                len(self.chunks), min(chunk_lengths), max(chunk_lengths), float(np.mean(chunk_lengths))
            )
        else:
            logger.warning("Loaded 0 chunks for BM25")

        # Initialize Italian linguistic resources
        self.stop_words = set(nltk.corpus.stopwords.words('italian')) if use_stopwords else set()
        self.stemmer = nltk.stem.SnowballStemmer("italian") if use_stemming else None
        
        # Preprocess chunks
        self.tokenized_chunks = [self._preprocess(chunk) for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks, k1=self.k1, b=self.b)
        self.chunk_map = defaultdict(list)
        for idx, tokens in enumerate(self.tokenized_chunks):
            self.chunk_map[tuple(tokens)].append(idx)

    def _concat_fields(self, meta: Dict[str, Any]) -> str:
        title = meta.get('title', '')
        keywords = ' '.join(meta.get('keywords', []))
        headers = ' '.join(meta.get('headers_context', []))
        content = meta.get('document', '')
        return f"{title} {keywords} {headers} {content}"

    def _preprocess(self, text: str) -> List[str]:
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
            results.append({
                "score": float(scores[idx]),
                "text": self.chunks[idx],
                **self.metadata_list[idx]
            })
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
                batch_result.append({
                    "score": float(scores[idx]),
                    "text": self.chunks[idx],
                    **self.metadata_list[idx]
                })
            results.append(batch_result)
        return results

    def query_with_scores(self, question: str, top_k: int = 20) -> List[Dict[str, Any]]:
        return self.retrieve(question, k=top_k)

    def batch_query_with_scores(self, questions: List[str], top_k: int = 20) -> List[List[Dict[str, Any]]]:
        return self.batch_retrieve(questions, k=top_k)


class HybridRetriever:
    def __init__(self, rrf_k=60, dense_weight=1.0, sparse_weight=1.0, device=None, alpha=0.3):
        self.dense = DenseRetriever(device=device)
        self.bm25 = BM25Retriever()
        self.rrf_k = rrf_k
        self.alpha = alpha
        self._validate_retriever_consistency()

    def _minmax_norm(self, scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        vmin, vmax = min(scores.values()), max(scores.values())
        if vmax == vmin:
            return {k: 1.0 for k in scores}  # all equal → 1.0
        return {k: (v - vmin) / (vmax - vmin) for k, v in scores.items()}
        
    def _validate_retriever_consistency(self):
        dense_ids = set(self.dense.vector_db.get_document_ids())
        bm25_ids = set(self.bm25.vector_db.get_document_ids())
        if dense_ids != bm25_ids:
            missing_in_dense = bm25_ids - dense_ids
            missing_in_bm25 = dense_ids - bm25_ids
            logger.warning(
                "Document ID mismatch: %d in BM25 not in Dense, %d in Dense not in BM25",
                len(missing_in_dense), len(missing_in_bm25)
            )
    
    def _standardize_results(self, results: List[Dict], retriever_type: str) -> Dict[str, Dict]:
        standardized = {}
        for rank, doc in enumerate(results, 1):
            doc_id = doc.get('id', str(doc.get('text', ''))[:100])
            standardized[doc_id] = {
                'id': doc_id,
                'text': doc.get('text', doc.get('content', '')),
                'score': doc.get('score', 0.0),
                'retriever': retriever_type,
                'rank': rank
            }
        return standardized

    def _fuse_results(self, dense_results: Dict[str, Dict], sparse_results: Dict[str, Dict], top_k: int) -> List[Dict]:
        # collect raw scores per retriever
        d_raw = {doc_id: d['score'] for doc_id, d in dense_results.items()}
        s_raw = {doc_id: s['score'] for doc_id, s in sparse_results.items()}
        # normalize within each list (per query)
        d = self._minmax_norm(d_raw)
        s = self._minmax_norm(s_raw)
        # union of docs
        all_ids = set(dense_results) | set(sparse_results)
        fused = []
        for doc_id in all_ids:
            Sd = d.get(doc_id, 0.0)
            Ss = s.get(doc_id, 0.0)
            Sh = self.alpha * Ss + Sd  # Sh = α·Ss + Sd (Wang 2023 paper)
            base = dense_results.get(doc_id) or sparse_results.get(doc_id)
            fused.append((Sh, {**base, "hybrid_score": Sh}))
        fused.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in fused[:top_k]]

    def retrieve(self, question: str, top_k: int = 5, candidate_k: int = 50) -> List[Dict]:
        dense_raw = self.dense.query_with_scores(question, top_k=candidate_k)
        sparse_raw = self.bm25.query_with_scores(question, top_k=candidate_k)
        dense_std = self._standardize_results(dense_raw, 'dense')
        sparse_std = self._standardize_results(sparse_raw, 'sparse')
        return self._fuse_results(dense_std, sparse_std, top_k)
    
    def batch_retrieve(self, questions: List[str], top_k: int = 5, candidate_k: int = 50) -> List[List[Dict]]:
        dense_batch = self.dense.batch_query_with_scores(questions, top_k=candidate_k)
        sparse_batch = self.bm25.batch_query_with_scores(questions, top_k=candidate_k)
        results = []
        for dense_raw, sparse_raw in zip(dense_batch, sparse_batch):
            dense_std = self._standardize_results(dense_raw, 'dense')
            sparse_std = self._standardize_results(sparse_raw, 'sparse')
            results.append(self._fuse_results(dense_std, sparse_std, top_k))
        return results
    

# === EVALUATION LOGIC  ===

def compute_ndcg(relevant_ids: Set[str], retrieved_docs: List[Dict], k: int = 5) -> float:
    if k <= 0 or not retrieved_docs:
        return 0.0
    top_k_docs = retrieved_docs[:k]
    y_true = [1.0 if doc.get('id') in relevant_ids else 0.0 for doc in top_k_docs]
    ideal_sorted = sorted(y_true, reverse=True)
    idcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_sorted))
    dcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(y_true))
    return dcg / idcg if idcg > 0 else 0.0

def compute_ap(relevant_ids: Set[str], retrieved_docs: List[Dict], k: int = 5) -> float:
    if not relevant_ids or not retrieved_docs:
        return 0.0
    top_k_docs = retrieved_docs[:k]
    relevant_count = 0
    precision_sum = 0.0
    for i, doc in enumerate(top_k_docs):
        if doc.get('id') in relevant_ids:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)
    return precision_sum / min(len(relevant_ids), k)

class RetrieverEvaluator:
    def __init__(self, retriever: HybridRetriever, batch_size: int = 32):
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

    # ---- UPDATED: accepts metadata and wraps payload ----
    def save_reports(self, metrics: Dict[str, Any], prefix: str = "HYBRID_eval", metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save aggregated metrics with rich metadata to file"""
        if not metrics or metrics["num_queries"] == 0:
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

def main(top_k: int = 5, candidate_k: int = 50):
    retriever = HybridRetriever(rrf_k=60, dense_weight=1.0, sparse_weight=1.0)
    evaluator = RetrieverEvaluator(retriever, batch_size=32)

    SINGLEHOP_FILE = DATA_DIR / "test" / "test_dataset_singlehop.json"
    MULTIHOP_FILE = DATA_DIR / "test" / "test_dataset_multihop.json"

    # Common run metadata injected into the saved JSON
    run_meta_common = {
        "experiment_prefix": "HYBRID_eval",
        "retrieval_top_k": top_k,
        "candidate_k": candidate_k,
        "hybrid": _describe_hybrid(retriever),
    }

    print("\n=== Evaluating SINGLEHOP dataset (HYBRID) ===")
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
            prefix="HYBRID_eval_singlehop",
            metadata={**run_meta_common, "dataset": "singlehop"}
        )
        print(f"Reports saved to {ABLATION_METRICS_DIR}")
    else:
        print("Singlehop evaluation failed.")

    print("\n=== Evaluating SINGLE+MULTIHOP dataset (HYBRID) ===")
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
            prefix="HYBRID_eval_combined",
            metadata={**run_meta_common, "dataset": "combined", "combined_sources": ["singlehop", "multihop"], "shuffle": True}
        )
        print(f"Reports saved to {ABLATION_METRICS_DIR}")
    else:
        print("Combined evaluation failed.")

if __name__ == "__main__":
    main(top_k=5, candidate_k=50)
