# ablation/rerank/experiments.py
import sys
from pathlib import Path
import numpy as np
import nltk
from rank_bm25 import BM25Okapi
import os
from collections import defaultdict
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import logging
from concurrent.futures import ThreadPoolExecutor
import hashlib
from typing import List, Dict, Tuple, Optional, Any, Set
from functools import lru_cache
import json
import time
import random
import pandas as pd
from contextlib import nullcontext
from datetime import datetime

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

os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
os.environ.setdefault("DATA_DIR", str(PROJECT_ROOT / "data"))

# -------------------------------
# Time helper
# -------------------------------
def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

# -------------------------------
# Vector wrapper
# -------------------------------
class VectorDatabaseWrapper:
    def __init__(self, vector_db: VectorDatabaseManager):
        self.vector_db = vector_db
        self._cached_embeddings = {}

    def query(self, question: str, top_k: int = 5) -> list:
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
        if not hasattr(self.vector_db, "metadata_db") or self.vector_db.metadata_db is None:
            return []
        return [item.get("id") for item in self.vector_db.metadata_db if "id" in item]

    def query_batch(self, questions: List[str], top_k: int = 5) -> List[List]:
        query_hashes = [self._hash_query(q) for q in questions]
        results = []
        to_fetch = []
        for idx, q_hash in enumerate(query_hashes):
            if q_hash in self._cached_embeddings:
                results.append((idx, self._cached_embeddings[q_hash]))
            else:
                to_fetch.append((idx, questions[idx]))
        if to_fetch:
            batch_questions = [q for _, q in to_fetch]
            batch_results = self.vector_db.query_batch(batch_questions, top_k) or [[] for _ in to_fetch]
            for (idx, _), batch_res in zip(to_fetch, batch_results):
                q_hash = query_hashes[idx]
                self._cached_embeddings[q_hash] = batch_res
                results.append((idx, batch_res))
        results.sort(key=lambda x: x[0])
        return [res for _, res in results]

    def _hash_query(self, query: str) -> str:
        normalized = " ".join(query.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()

# -------------------------------
# Simple retrievers
# -------------------------------
class DenseRetriever:
    def __init__(self, device=None):
        self.vector_db = VectorDatabaseWrapper(VectorDatabaseManager(device=device))

    def _format_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{
            'id': item.get('id'),
            'chunk_id': item.get('id'),
            'score': item.get('score', 0.0),
            **item
        } for item in (results or []) if item.get('id') is not None]

    def retrieve(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self._format_results(self.vector_db.query(question, top_k=top_k))

    def query_with_scores(self, question: str, top_k: int = 20) -> List[Dict[str, Any]]:
        return self._format_results(self.vector_db.query(question, top_k=top_k))

    def batch_query_with_scores(self, questions: List[str], top_k: int = 20) -> List[List[Dict[str, Any]]]:
        batch = self.vector_db.query_batch(questions, top_k=top_k)
        return [self._format_results(res) for res in batch]

class BM25Retriever:
    def __init__(self, use_stopwords=True, use_stemming=True, k1=1.5, b=0.75):
        self.vector_db = VectorDatabaseWrapper(VectorDatabaseManager())
        self.chunks = [self._concat_fields(meta) for meta in self.vector_db.metadata_db]
        self.metadata_list = [meta for meta in self.vector_db.metadata_db]
        self.use_stopwords = use_stopwords
        self.use_stemming = use_stemming
        self.k1 = k1
        self.b = b

        chunk_lengths = [len(c.split()) for c in self.chunks] if self.chunks else []
        if chunk_lengths:
            logger.info(
                "Loaded %d chunks | len min=%d max=%d avg=%.1f words",
                len(self.chunks), min(chunk_lengths), max(chunk_lengths), float(np.mean(chunk_lengths))
            )
        else:
            logger.warning("Loaded 0 chunks for BM25")

        self.stop_words = set(nltk.corpus.stopwords.words('italian')) if use_stopwords else set()
        self.stemmer = nltk.stem.SnowballStemmer("italian") if use_stemming else None

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

    def retrieve(self, query: str, top_k: int = 5, candidates_k: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        k = top_k if top_k is not None else candidates_k if candidates_k is not None else 5
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

    def batch_retrieve(self, queries: List[str], top_k: int = 5, candidates_k: Optional[int] = None, **kwargs) -> List[List[Dict[str, Any]]]:
        k = top_k if top_k is not None else candidates_k if candidates_k is not None else 5
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
        return self.retrieve(question, top_k=top_k)

    def batch_query_with_scores(self, questions: List[str], top_k: int = 20) -> List[List[Dict[str, Any]]]:
        return self.batch_retrieve(questions, top_k=top_k)

class HybridRetriever:
    def __init__(self, rrf_k=60, dense_weight=1.0, sparse_weight=1.0, device=None):
        self.dense = DenseRetriever(device=device)
        self.bm25 = BM25Retriever()
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self._validate_retriever_consistency()

    def _validate_retriever_consistency(self):
        dense_ids = set(self.dense.vector_db.get_document_ids())
        bm25_ids = set(self.bm25.vector_db.get_document_ids())
        if dense_ids != bm25_ids:
            missing_in_dense = bm25_ids - dense_ids
            missing_in_bm25 = dense_ids - bm25_ids
            logger.warning(f"Document ID mismatch: {len(missing_in_dense)} in BM25 not in Dense, "
                           f"{len(missing_in_bm25)} in Dense not in BM25")

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
        all_docs = set(dense_results.keys()) | set(sparse_results.keys())
        fused_scores = []
        for doc_id in all_docs:
            dense_rank = dense_results.get(doc_id, {}).get('rank', float('inf'))
            sparse_rank = sparse_results.get(doc_id, {}).get('rank', float('inf'))
            rrf_dense = self.dense_weight / (self.rrf_k + dense_rank) if dense_rank != float('inf') else 0
            rrf_sparse = self.sparse_weight / (self.rrf_k + sparse_rank) if sparse_rank != float('inf') else 0
            rrf_score = rrf_dense + rrf_sparse
            doc_source = dense_results.get(doc_id) or sparse_results.get(doc_id)
            fused_scores.append((rrf_score, doc_source))
        fused_scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in fused_scores[:top_k]]

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

# -------------------------------
# Rerank retriever
# -------------------------------
class RerankRetriever:
    def __init__(
        self,
        base_retriever=None,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-2-v2",
        device: Optional[str] = None,
        max_length: int = 256,
        batch_size: int = 64,
        cache_size: int = 5000,
        max_rerank_candidates: int = 20
    ):
        self.base = base_retriever if base_retriever else DenseRetriever(device=device)
        self.max_rerank_candidates = max_rerank_candidates
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.reranker_name = reranker_model
        self.max_length = max_length
        self.batch_size = batch_size
        self._init_reranker()
        self.score_cache = lru_cache(maxsize=cache_size)(self._compute_scores_uncached)

    def _init_reranker(self):
        try:
            logger.info(f"Loading reranker: {self.reranker_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.reranker_name, truncation_side='left')
            self.reranker = AutoModelForSequenceClassification.from_pretrained(
                self.reranker_name,
                torch_dtype=torch.float16 if 'cuda' in self.device else torch.float32
            ).to(self.device).eval()
            if hasattr(torch, 'compile') and 'cuda' in self.device:
                self.reranker = torch.compile(self.reranker)
            logger.info(f"Reranker loaded on {self.device} with AMP")
        except Exception as e:
            logger.error(f"Failed to load reranker: {str(e)}")
            raise RuntimeError("Reranker initialization failed") from e

    def _compute_scores_uncached(self, query: str, documents: Tuple[str, ...]) -> List[float]:
        pairs = [(query, doc) for doc in documents]
        scores = []
        amp_ctx = torch.cuda.amp.autocast if (self.device.startswith("cuda") and torch.cuda.is_available()) else nullcontext
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i+self.batch_size]
            features = self.tokenizer(
                batch,
                padding='longest',
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_token_type_ids=False
            ).to(self.device)
            with torch.no_grad(), amp_ctx():
                outputs = self.reranker(**features)
                logits = outputs.logits
                if logits.shape[1] == 1:
                    batch_scores = logits.squeeze(-1).float().cpu().numpy()
                else:
                    batch_scores = logits[:, 1].float().cpu().numpy()
            scores.extend(batch_scores.tolist())
        return scores

    def _get_scores(self, query: str, documents: List[str]) -> List[float]:
        return self.score_cache(query, tuple(documents))

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
        strategy: str = "cross_encoder",
        diversity_penalty: float = 0.5,
        fast_mode: bool = True
    ) -> List[Dict]:
        if len(candidates) > self.max_rerank_candidates:
            candidates = sorted(candidates, key=lambda x: x.get('score', 0), reverse=True)[:self.max_rerank_candidates]
        documents = [c.get('content', '') or c.get('text', '') or c.get('document', '') for c in candidates]
        if strategy == "cross_encoder":
            scores = self._get_scores(query, documents)
            sorted_indices = np.argsort(scores)[::-1]
            return [candidates[i] for i in sorted_indices[:top_k]]
        elif strategy == "mmr":
            return self._mmr_rerank(query, candidates, top_k, diversity_penalty, fast_mode)
        elif strategy == "diversity":
            return self._optimized_diversity_rerank(candidates, top_k, fast_mode)
        else:
            raise ValueError(f"Unknown reranking strategy: {strategy}")

    # (MMR/diversity implementations omitted for brevity)

    def _retrieve_candidates(self, question: str, candidates_k: int):
        try:
            return self.base.retrieve(question, top_k=candidates_k, candidate_k=candidates_k)
        except TypeError:
            return self.base.retrieve(question, top_k=candidates_k)

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        candidates_k: int = 50,
        strategy: str = "cross_encoder",
        fast_mode: bool = True,
        **kwargs
    ) -> List[Dict]:
        candidates = self._retrieve_candidates(question, candidates_k)
        return self.rerank(question, candidates, top_k, strategy, fast_mode=fast_mode, **kwargs)

    def batch_retrieve(
        self,
        questions: List[str],
        top_k: int = 5,
        candidates_k: int = 50,
        strategy: str = "cross_encoder",
        fast_mode: bool = True,
        **kwargs
    ) -> List[List[Dict]]:
        all_candidates = [self._retrieve_candidates(q, candidates_k) for q in questions]
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.rerank,
                    q,
                    cands,
                    top_k,
                    strategy,
                    fast_mode=fast_mode,
                    **kwargs
                )
                for q, cands in zip(questions, all_candidates)
            ]
            results = [f.result() for f in futures]
        return results

# -------------------------------
# Experiment runner (adds METADATA)
# -------------------------------
def run_rerank_experiment(
    retriever_type: str,
    cross_encoder_model: str,
    top_k: int = 5,
    candidate_k: int = 50,
    batch_size: int = 32,
    prefix: str = None
):
    # --- build base + reranker ---
    if retriever_type == "dense":
        base_retriever = DenseRetriever(device="cpu")
    elif retriever_type == "bm25":
        base_retriever = BM25Retriever()
    elif retriever_type == "hybrid":
        base_retriever = HybridRetriever(rrf_k=60, dense_weight=1.0, sparse_weight=1.0, device="cpu")
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

    retriever = RerankRetriever(
        base_retriever=base_retriever,
        reranker_model=cross_encoder_model,
        device="cpu"
    )
    evaluator = RetrieverEvaluator(retriever, batch_size=batch_size, candidate_k=candidate_k)

    # Describe everything used in this run (metadata goes into the JSON)
    run_meta_common = {
        "experiment_prefix": prefix or f"RERANK_{retriever_type}",
        "retrieval_top_k": top_k,
        "candidate_k": candidate_k,
        "base_retriever": _describe_base_retriever(base_retriever),
        "reranker": _describe_reranker(retriever),
        "rerank_strategy": "cross_encoder",  # evaluator uses default strategy
    }

    SINGLEHOP_FILE = DATA_DIR / "test" / "test_dataset_singlehop.json"
    MULTIHOP_FILE = DATA_DIR / "test" / "test_dataset_multihop.json"

    base_prefix = prefix or f"RERANK_{retriever_type}"

    # --- SINGLEHOP ---
    print(f"\n=== Evaluating SINGLEHOP dataset ({retriever_type.upper()} + Rerank) ===")
    singlehop_metrics = evaluator.evaluate_retrieval(SINGLEHOP_FILE, top_k)
    if singlehop_metrics and singlehop_metrics["num_queries"] > 0:
        print(f"{retriever_type.upper()}+Rerank Singlehop: {singlehop_metrics}")
        evaluator.save_reports(
            singlehop_metrics,
            prefix=f"{base_prefix}_singlehop",
            metadata={**run_meta_common, "dataset": "singlehop"}
        )
    else:
        print(f"{retriever_type.upper()}+Rerank Singlehop evaluation failed.")

    # --- COMBINED (singlehop + multihop) ---
    print(f"\n=== Evaluating COMBINED dataset ({retriever_type.upper()} + Rerank) ===")
    combined_data = load_and_shuffle_datasets(SINGLEHOP_FILE, MULTIHOP_FILE)
    temp_combined_file = DATA_DIR / "test" / "test_dataset_combined_temp.json"
    try:
        with open(temp_combined_file, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)

        combined_metrics = evaluator.evaluate_retrieval(temp_combined_file, top_k)
        if combined_metrics and combined_metrics["num_queries"] > 0:
            print(f"{retriever_type.upper()}+Rerank Combined: {combined_metrics}")
            evaluator.save_reports(
                combined_metrics,
                prefix=f"{base_prefix}_combined",
                metadata={**run_meta_common, "dataset": "combined", "combined_sources": ["singlehop", "multihop"], "shuffle": True}
            )
        else:
            print(f"{retriever_type.upper()}+Rerank Combined evaluation failed.")
    finally:
        try:
            temp_combined_file.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Could not delete temporary file: {str(e)}")

# -------------------------------
# Evaluation plumbing
# -------------------------------
current_dir = Path(__file__).resolve().parent
root_dir = PROJECT_ROOT
DATA_DIR = root_dir / "data"
METRICS_DIR = DATA_DIR / "metrics" / "ablation_metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

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

def _describe_base_retriever(base):
    """
    Return a dict with the retriever type and its key parameters.
    Works for DenseRetriever, BM25Retriever, and HybridRetriever.
    """
    info: Dict[str, Any] = {}

    # Hybrid
    if hasattr(base, "bm25") and hasattr(base, "dense"):
        info["type"] = "hybrid"
        info["hybrid"] = {
            "rrf_k": getattr(base, "rrf_k", None),
            "dense_weight": getattr(base, "dense_weight", None),
            "sparse_weight": getattr(base, "sparse_weight", None),
        }
        # Describe dense
        device = os.getenv("DEVICE", "cpu")
        try:
            mgr = base.dense.vector_db.vector_db
            if hasattr(mgr, "device") and mgr.device:
                device = mgr.device
        except Exception:
            pass
        info["dense"] = {"impl": "DenseRetriever", "device": device}
        # Describe BM25
        bm = getattr(base, "bm25", None)
        if bm:
            try:
                lengths = [len(c.split()) for c in getattr(bm, "chunks", [])]
                corpus = {
                    "num_chunks": len(bm.chunks),
                    "min_len": int(min(lengths)) if lengths else 0,
                    "max_len": int(max(lengths)) if lengths else 0,
                    "avg_len": float(np.mean(lengths)) if lengths else 0.0,
                }
            except Exception:
                corpus = None
            info["bm25"] = {
                "use_stopwords": getattr(bm, "use_stopwords", None),
                "use_stemming": getattr(bm, "use_stemming", None),
                "k1": getattr(bm, "k1", None),
                "b": getattr(bm, "b", None),
                "corpus": corpus,
            }
        return info

    # BM25
    if base.__class__.__name__ == "BM25Retriever":
        try:
            lengths = [len(c.split()) for c in getattr(base, "chunks", [])]
            corpus = {
                "num_chunks": len(base.chunks),
                "min_len": int(min(lengths)) if lengths else 0,
                "max_len": int(max(lengths)) if lengths else 0,
                "avg_len": float(np.mean(lengths)) if lengths else 0.0,
            }
        except Exception:
            corpus = None
        return {
            "type": "bm25",
            "bm25": {
                "use_stopwords": getattr(base, "use_stopwords", None),
                "use_stemming": getattr(base, "use_stemming", None),
                "k1": getattr(base, "k1", None),
                "b": getattr(base, "b", None),
                "corpus": corpus,
            },
        }

    # Dense
    device = os.getenv("DEVICE", "cpu")
    try:
        mgr = base.vector_db.vector_db
        if hasattr(mgr, "device") and mgr.device:
            device = mgr.device
    except Exception:
        pass
    return {"type": "dense", "dense": {"impl": "DenseRetriever", "device": device}}

def _describe_reranker(rerank_retriever):
    """Return a dict with reranker model + inference settings."""
    return {
        "reranker_model": getattr(rerank_retriever, "reranker_name", None),
        "device": getattr(rerank_retriever, "device", None),
        "max_length": getattr(rerank_retriever, "max_length", None),
        "batch_size": getattr(rerank_retriever, "batch_size", None),
        "max_rerank_candidates": getattr(rerank_retriever, "max_rerank_candidates", None),
        "default_strategy": "cross_encoder",
        "fast_mode_default": True,
    }

class RetrieverEvaluator:
    def __init__(self, retriever: RerankRetriever, batch_size: int = 32, candidate_k: int = 50):
        self.retriever = retriever
        self.batch_size = batch_size
        self.candidate_k = candidate_k
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
        retrieved_docs_batch = [self.retriever.retrieve(q, top_k=top_k, candidates_k=getattr(self, "candidate_k", 50)) for q in questions]
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

    def save_reports(self, metrics: Dict[str, Any], prefix: str = "RERANK_eval", metadata: Dict[str, Any] = None) -> None:
        if not metrics or metrics.get("num_queries", 0) == 0:
            print("No metrics to save")
            return

        meta = {
            "timestamp_utc": _now_iso(),
            "batch_size_eval": self.batch_size,
            "candidate_k_eval": self.candidate_k,
        }
        if metadata:
            meta.update(metadata)

        payload = {
            "metadata": meta,
            "metrics": metrics
        }

        metrics_file = METRICS_DIR / f"{prefix}_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

# -------------------------------
# Dataset utils
# -------------------------------
def load_and_shuffle_datasets(singlehop_path, multihop_path):
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
def main(top_k: int = 5, candidate_k: int = 50):
    ce_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    for retriever_type in ["dense", "bm25", "hybrid"]:
        run_rerank_experiment(
            retriever_type=retriever_type,
            cross_encoder_model=ce_model,
            top_k=top_k,
            candidate_k=candidate_k,
            batch_size=32,
            prefix=f"RERANK_{retriever_type}"
        )

if __name__ == "__main__":
    main(top_k=5, candidate_k=50)
