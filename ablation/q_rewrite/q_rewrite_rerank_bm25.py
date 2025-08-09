# ablation/experiments.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import re
import gc
import json
import time
import math
import torch
import spacy
import nltk
import numpy as np
import pandas as pd
import logging
import random
import hashlib
import inspect

from typing import List, Dict, Tuple, Union, Optional, Any, Set
from functools import lru_cache
from collections import defaultdict

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import CountVectorizer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModel,
)
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

from vector_store import VectorDatabaseManager

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data"
METRICS_DIR = DATA_DIR / "metrics"
ABLATION_METRICS_DIR = METRICS_DIR / "ablation_metrics_alpha_hybrid"
ABLATION_METRICS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# helpers
# -------------------------------
def _now_iso() -> str:
    from datetime import datetime
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _safe_len(arr) -> int:
    try:
        return len(arr)
    except Exception:
        return 0


# ===============================
# Vector wrapper
# ===============================
class VectorDatabaseWrapper:
    def __init__(self, vector_db: VectorDatabaseManager):
        self.vector_db = vector_db
        self._cached_embeddings: Dict[str, List[Dict[str, Any]]] = {}

    @property
    def metadata_db(self):
        return getattr(self.vector_db, "metadata_db", None) or []

    def _hash_query(self, query: str) -> str:
        normalized = " ".join(query.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()

    def query(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        qh = self._hash_query(question)
        if qh in self._cached_embeddings:
            return self._cached_embeddings[qh]
        try:
            results = self.vector_db.query(question, top_k) or []
        except Exception as e:
            logger.exception("Vector query failed: %s", e)
            results = []
        self._cached_embeddings[qh] = results
        return results

    def get_document_ids(self) -> List[str]:
        return [m.get("id") for m in self.metadata_db if m.get("id") is not None]


# ===============================
# BM25
# ===============================
class BM25Retriever:
    def __init__(self, use_stopwords=True, use_stemming=True, k1=1.5, b=0.75):
        """
        BM25Retriever with configurable parameters.
        If the corpus is empty, works as a no-op retriever.
        """
        self.vector_db = VectorDatabaseWrapper(VectorDatabaseManager())
        self.use_stopwords = use_stopwords
        self.use_stemming = use_stemming
        self.k1 = k1
        self.b = b

        # Build corpus from metadata
        self.chunks = [self._concat_fields(meta) for meta in (self.vector_db.metadata_db or [])]
        self.metadata_list = [meta for meta in (self.vector_db.metadata_db or [])]

        if not self.chunks:
            logger.warning("BM25 corpus is empty. BM25 retriever will return no results.")
            # mark as disabled
            self.stop_words = set()
            self.stemmer = None
            self.tokenized_chunks = []
            self.bm25 = None
            return

        # Log stats
        chunk_lengths = [len(c.split()) for c in self.chunks]
        logger.info(
            "BM25 corpus: %d chunks | len min=%d max=%d avg=%.1f",
            len(self.chunks), min(chunk_lengths), max(chunk_lengths), float(np.mean(chunk_lengths))
        )

        # Linguistic resources
        self.stop_words = set(nltk.corpus.stopwords.words('italian')) if use_stopwords else set()
        self.stemmer = nltk.stem.SnowballStemmer("italian") if use_stemming else None

        # Build index
        self.tokenized_chunks = [self._preprocess(chunk) for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks, k1=self.k1, b=self.b)

    def _concat_fields(self, meta: Dict[str, Any]) -> str:
        title = meta.get('title', '')
        keywords = ' '.join(meta.get('keywords', []))
        headers = ' '.join(meta.get('headers_context', []))
        content = meta.get('document', '')
        return f"{title} {keywords} {headers} {content}".strip()

    def _preprocess(self, text: str) -> List[str]:
        text = re.sub(r"[^\w\s']", "", text)
        text = re.sub(r"\b(l'|un'|all'|d'|dell'|quest'|nell')\b", "", text, flags=re.IGNORECASE)
        text = text.lower()
        tokens = nltk.word_tokenize(text, language='italian')
        out = []
        for token in tokens:
            if len(token) < 2:
                continue
            if self.use_stopwords and token in self.stop_words:
                continue
            if self.use_stemming and self.stemmer is not None:
                token = self.stemmer.stem(token)
            out.append(token)
        return out

    @lru_cache(maxsize=1000)
    def _preprocess_query(self, query: str) -> List[str]:
        return self._preprocess(query or "")

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.tokenized_chunks:
            return []
        tokenized_query = self._preprocess_query(query)
        scores = self.bm25.get_scores(tokenized_query)
        if len(scores) == 0:
            return []
        k = max(0, min(k, len(scores)))
        if k == 0:
            return []
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

    def query_with_scores(self, question: str, top_k: int = 20) -> List[Dict[str, Any]]:
        return self.retrieve(question, k=top_k)


# ===============================
# Reranker (cross-encoder over BM25 results)
# ===============================
class RerankRetriever:
    def __init__(
        self,
        base_retriever=None,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        max_length: int = 256,
        batch_size: int = 64,
        cache_size: int = 5000,
        max_rerank_candidates: int = 20,
    ):
        self.base = base_retriever if base_retriever else BM25Retriever()
        self.max_rerank_candidates = max_rerank_candidates
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.reranker_name = reranker_model
        self.max_length = max_length
        self.batch_size = batch_size
        self._init_reranker()
        self.score_cache = lru_cache(maxsize=cache_size)(self._compute_scores_uncached)

    def _init_reranker(self):
        try:
            logger.info("Loading reranker: %s", self.reranker_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.reranker_name, truncation_side="left")
            self.reranker = AutoModelForSequenceClassification.from_pretrained(
                self.reranker_name,
                torch_dtype=(torch.float16 if ("cuda" in self.device) else torch.float32),
            ).to(self.device).eval()
        except Exception as e:
            logger.error("Failed to load reranker: %s", e)
            raise

    def _compute_scores_uncached(self, query: str, documents: Tuple[str, ...]) -> List[float]:
        scores: List[float] = []
        use_amp = self.device.startswith("cuda") and torch.cuda.is_available()
        for i in range(0, len(documents), self.batch_size):
            pairs = [(query, d) for d in documents[i:i + self.batch_size]]
            feats = self.tokenizer(
                pairs,
                padding="longest",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_token_type_ids=False,
            ).to(self.device)
            with torch.no_grad(), (torch.cuda.amp.autocast() if use_amp else torch.autocast("cpu", enabled=False)):
                logits = self.reranker(**feats).logits
                if logits.shape[1] == 1:
                    s = logits.squeeze(-1).float().cpu().numpy()
                else:
                    s = logits[:, 1].float().cpu().numpy()
            scores.extend(s.tolist())
        return scores

    def _get_scores(self, query: str, documents: List[str]) -> List[float]:
        return self.score_cache(query, tuple(documents))

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if len(candidates) > self.max_rerank_candidates:
            candidates = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)[: self.max_rerank_candidates]
        docs = [c.get("content") or c.get("text") or c.get("document") or "" for c in candidates]
        scores = self._get_scores(query, docs)
        order = np.argsort(scores)[::-1]
        return [candidates[i] for i in order[:top_k]]

    def retrieve(self, question: str, top_k: int = 5, candidates_k: int = 50, **kwargs) -> List[Dict[str, Any]]:
        """
        Normalize args to whatever the base retriever expects.
        Supports BM25 (k=...), Dense/Hybrid (top_k / candidate_k), etc.
        """
        try:
            sig = inspect.signature(self.base.retrieve)
        except (ValueError, TypeError):
            sig = None

        base_kwargs = {}
        if sig:
            if "top_k" in sig.parameters:
                base_kwargs["top_k"] = candidates_k
            elif "k" in sig.parameters:
                base_kwargs["k"] = candidates_k
            if "candidates_k" in sig.parameters:
                base_kwargs["candidates_k"] = candidates_k
            elif "candidate_k" in sig.parameters:
                base_kwargs["candidate_k"] = candidates_k

        try:
            cands = self.base.retrieve(question, **base_kwargs)
        except TypeError:
            try:
                cands = self.base.retrieve(question, candidates_k)
            except Exception:
                cands = self.base.retrieve(question)

        return self.rerank(question, cands, top_k)

    def batch_retrieve(self, questions: List[str], top_k: int = 5, candidates_k: int = 50, **kwargs) -> List[List[Dict[str, Any]]]:
        return [self.retrieve(q, top_k=top_k, candidates_k=candidates_k) for q in questions]


# ===============================
# Query rewriting
# ===============================
class QueryRewriter:
    """
    Best-effort, fail-graceful initializations for optional components.
    """
    def __init__(self, device: Optional[str] = None, lang: str = "italian"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lang = lang

        # spaCy
        self.nlp = None
        if self.lang == "italian":
            try:
                self.nlp = spacy.load("it_core_news_md")
            except Exception:
                try:
                    import spacy.cli
                    spacy.cli.download("it_core_news_md")
                    self.nlp = spacy.load("it_core_news_md")
                except Exception:
                    logger.warning("spaCy it_core_news_md not available; falling back to rule-based.")
                    self.nlp = None

        # CCE (seq2seq) – optional
        self.cce_tokenizer = None
        self.cce_model = None
        try:
            self.cce_tokenizer = AutoTokenizer.from_pretrained("gsarti/it5-small")
            self.cce_model = AutoModelForSeq2SeqLM.from_pretrained("gsarti/it5-small").to(self.device).eval()
        except Exception:
            logger.warning("CCE (it5-small) unavailable; CCE will be skipped.")
            self.cce_tokenizer = None
            self.cce_model = None

        # Embeddings – optional
        self.embed_tokenizer = None
        self.embed_model = None
        try:
            self.embed_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
            self.embed_model = AutoModel.from_pretrained("intfloat/multilingual-e5-small").to(self.device).eval()
        except Exception:
            logger.warning("Embedding model unavailable; synonyms/diversity degrade gracefully.")
            self.embed_tokenizer = None
            self.embed_model = None

        # KeyBERT – optional
        try:
            self.keyword_model = KeyBERT(model=SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device=str(self.device)))
        except Exception:
            logger.warning("KeyBERT unavailable; keyword features will be limited.")
            self.keyword_model = None


class QueryRewriteRetriever:
    def __init__(
        self,
        base_retriever,
        device: Optional[str] = None,
        expansion_terms: int = 3,
        lang: str = "italian",
        enable_cce: bool = True,
        enable_kwr: bool = True,
        enable_gqr: bool = True,
        enable_prf: bool = True,
        enable_decompose: bool = True,
    ):
        self.base = base_retriever
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.expansion_terms = expansion_terms
        self.lang = lang

        self.enable_cce = enable_cce
        self.enable_kwr = enable_kwr
        self.enable_gqr = enable_gqr
        self.enable_prf = enable_prf
        self.enable_decompose = enable_decompose

        self.qr = QueryRewriter(self.device, lang)
        self.rewrite_cache: Dict[str, List[str]] = {}
        self.embed_cache: Dict[str, np.ndarray] = {}

        # BM25 for PRF – build if we can
        self._bm25_for_prf = self._init_bm25_for_prf()

    # ---------- rewrite strategies ----------
    def core_content_extraction(self, question: str) -> str:
        if not (self.enable_cce and self.qr.cce_model and self.qr.cce_tokenizer):
            return question
        try:
            enc = self.qr.cce_tokenizer(
                f"estrai il contenuto principale: {question}",
                return_tensors="pt",
                truncation=True,
                max_length=128,
            ).to(self.device)
            out = self.qr.cce_model.generate(**enc, max_new_tokens=64, num_beams=4, early_stopping=True)
            return self.qr.cce_tokenizer.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            logger.debug("CCE inference failed: %s", e)
            return question

    def _sklearn_clean_stopwords(self) -> List[str]:
        if not self.qr.nlp:
            return list(nltk.corpus.stopwords.words("italian"))
        cleaned = set()
        for s in self.qr.nlp.Defaults.stop_words:
            t = re.sub(r"[^\w\s]", "", s.lower())
            if len(t) >= 2:
                cleaned.add(t)
        return list(cleaned)

    def keyword_rewriting(self, question: str) -> str:
        if not (self.enable_kwr and self.qr.keyword_model):
            return question
        try:
            vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w\w+\b", stop_words=self._sklearn_clean_stopwords())
            kws = self.qr.keyword_model.extract_keywords(
                question, keyphrase_ngram_range=(1, 2), vectorizer=vectorizer, top_n=self.expansion_terms
            )
            return " ".join([k for k, _ in kws])
        except Exception as e:
            logger.debug("KeyBERT failed: %s", e)
            return question

    def _embed_text(self, texts: List[str]) -> np.ndarray:
        if not (self.qr.embed_model and self.qr.embed_tokenizer):
            arr = []
            for t in texts:
                h = hashlib.md5(t.encode()).digest()
                arr.append(np.frombuffer(h, dtype=np.uint8).astype(np.float32))
            return np.vstack(arr)
        enc = self.qr.embed_tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.qr.embed_model(**enc).last_hidden_state.mean(dim=1).cpu().numpy()
        return out

    def _get_synonyms(self, word: str) -> List[str]:
        if not (self.qr.embed_model and self.qr.embed_tokenizer):
            return []
        # Placeholder for a real synonym lookup
        return []

    def keyword_expansion(self, question: str) -> List[str]:
        kw = self.keyword_rewriting(question)
        kws = kw.split()
        out = []
        try:
            if len(kws) >= 2:
                for i in range(len(kws)):
                    for j in range(i + 1, len(kws)):
                        out.append(f"{kws[i]} {kws[j]}")
            for k in kws:
                for syn in self._get_synonyms(k)[:1]:
                    out.append(question.replace(k, syn))
        except Exception:
            pass
        return out

    def general_query_rewriting(self, question: str) -> str:
        if not (self.enable_gqr and self.qr.nlp):
            return question
        try:
            doc = self.qr.nlp(question)
            toks = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct]
            return " ".join(toks) or question
        except Exception:
            return question

    def query_decomposition(self, question: str) -> List[str]:
        if not (self.enable_decompose and self.qr.nlp):
            return [question]
        try:
            if (" e " not in question) and (" o " not in question):
                return [question]
            doc = self.qr.nlp(question)
            conj = [t for t in doc if t.dep_ == "cc"]
            if not conj:
                return [question]
            subs, cur = [], []
            for t in doc:
                if t in conj:
                    if cur:
                        subs.append(" ".join(cur))
                        cur = []
                else:
                    cur.append(t.text)
            if cur:
                subs.append(" ".join(cur))
            return [s.strip() for s in subs if s.strip()]
        except Exception:
            return [question]

    # ---------- PRF ----------
    def _init_bm25_for_prf(self) -> Optional[BM25Okapi]:
        try:
            if hasattr(self.base, "chunks") and self.base.chunks:
                corpus = self.base.chunks
            else:
                md = getattr(getattr(self.base, "vector_db", None), "metadata_db", None) or []
                corpus = [m.get("document", "") for m in md if m.get("document")]
            if not corpus:
                return None
            tokenized = [self._bm25_prep(c) for c in corpus]
            if len(tokenized) == 0:
                return None
            return BM25Okapi(tokenized)
        except Exception:
            return None

    def _bm25_prep(self, text: str) -> List[str]:
        if self.qr.nlp:
            doc = self.qr.nlp(text.lower())
            return [t.lemma_ for t in doc if (not t.is_stop and not t.is_punct and len(t.text) > 1)]
        text = re.sub(r"[^\w\s]", " ", text.lower())
        toks = text.split()
        stops = set(nltk.corpus.stopwords.words("italian"))
        return [t for t in toks if t not in stops and len(t) > 1]

    def pseudo_relevance_feedback(self, question: str) -> str:
        if not (self.enable_prf and self._bm25_for_prf):
            return question
        try:
            # initial docs
            try:
                docs = self._call_base_retrieve(question, n=5)
            except Exception:
                docs = []
            texts = [(d.get("content") or d.get("text") or d.get("document") or "") for d in docs]
            terms = []
            q_toks = set(self._bm25_prep(question))
            for t in texts:
                for term in self._bm25_prep(t):
                    if term not in q_toks:
                        terms.append(term)
            freq = defaultdict(int)
            for t in terms:
                freq[t] += 1
            top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[: self.expansion_terms]
            out = question
            for t, _ in top:
                out += f" {t}"
            return out
        except Exception:
            return question

    # ---------- glue ----------
    def generate_rewrites(self, question: str, strategy: str = "all") -> List[str]:
        key = f"{strategy}:{hashlib.md5(question.encode()).hexdigest()}"
        if key in self.rewrite_cache:
            return self.rewrite_cache[key]

        rewrites = [question]
        if strategy in ("all", "core"):
            rewrites.append(self.core_content_extraction(question))
        if strategy in ("all", "keywords"):
            rewrites.append(self.keyword_rewriting(question))
            rewrites.extend(self.keyword_expansion(question))
        if strategy in ("all", "general"):
            rewrites.append(self.general_query_rewriting(question))
        if strategy in ("all", "prf"):
            rewrites.append(self.pseudo_relevance_feedback(question))
        if strategy in ("all", "decompose"):
            rewrites.extend(self.query_decomposition(question))
        if strategy == "hybrid":
            rewrites.append(f"{self.core_content_extraction(question)} {self.keyword_rewriting(question)}")

        seen, out = set(), []
        for q in rewrites:
            qn = (q or "").strip()
            if qn and (qn not in seen):
                seen.add(qn)
                out.append(qn)
        self.rewrite_cache[key] = out
        return out

    def _call_base_retrieve(self, query: str, n: int, candidates_k: Optional[int] = None):
        sig = inspect.signature(self.base.retrieve)
        kwargs = {}
        if "top_k" in sig.parameters:
            kwargs["top_k"] = n
        elif "k" in sig.parameters:
            kwargs["k"] = n
        else:
            try:
                return self.base.retrieve(query, n)
            except TypeError:
                pass
        if candidates_k is not None:
            if "candidates_k" in sig.parameters:
                kwargs["candidates_k"] = candidates_k
            elif "candidate_k" in sig.parameters:
                kwargs["candidate_k"] = candidates_k
        return self.base.retrieve(query, **kwargs)

    def retrieve(self, question: str, top_k: int = 5, strategy: str = "all", fusion_method: str = "rrf", **kwargs) -> List[Dict[str, Any]]:
        candidates_k = kwargs.pop("candidates_k", None)
        q_list = self.generate_rewrites(question, strategy=strategy)
        all_results: List[Dict[str, Any]] = []
        for q in q_list:
            try:
                docs = self._call_base_retrieve(q, n=top_k * 3, candidates_k=candidates_k) or []
                for d in docs:
                    d["source_query"] = q
                all_results.extend(docs)
            except Exception as e:
                logger.debug("base retrieve failed for '%s': %s", q, e)
        return self._fuse(all_results, top_k, fusion_method)

    def batch_retrieve(self, questions: List[str], top_k: int = 5, strategy: str = "all", fusion_method: str = "rrf", **kwargs) -> List[List[Dict[str, Any]]]:
        candidates_k = kwargs.pop("candidates_k", None)
        return [self.retrieve(q, top_k=top_k, strategy=strategy, fusion_method=fusion_method, candidates_k=candidates_k) for q in questions]

    def _fuse(self, results: List[Dict[str, Any]], top_k: int, method: str = "rrf") -> List[Dict[str, Any]]:
        if not results:
            return []
        bucket: Dict[str, Dict[str, Any]] = {}
        for d in results:
            did = d.get("id") or d.get("chunk_id") or hashlib.md5((d.get("text", "")[:200]).encode()).hexdigest()
            if did not in bucket:
                bucket[did] = {"doc": d, "ranks": [], "scores": [], "sources": set()}
            b = bucket[did]
            b["scores"].append(float(d.get("score", 0.0)))
            b["sources"].add(d.get("source_query", ""))
            b["ranks"].append(len(b["ranks"]) + 1)
        if method == "rrf":
            fused = {did: sum(1.0 / (60 + r) for r in b["ranks"]) for did, b in bucket.items()}
        else:
            fused = {did: (1 + 0.2 * (len(b["sources"]) - 1)) * (np.mean(b["scores"]) if b["scores"] else 0.0) for did, b in bucket.items()}
        order = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return [bucket[did]["doc"] for did, _ in order[:top_k]]


# ===============================
# Evaluation
# ===============================
def compute_ndcg(relevant_ids: Set[str], retrieved_docs: List[Dict[str, Any]], k: int = 5) -> float:
    if k <= 0 or not retrieved_docs:
        return 0.0
    top_k = retrieved_docs[:k]
    y_true = [1.0 if (doc.get("id") in relevant_ids) else 0.0 for doc in top_k]
    ideal_sorted = sorted(y_true, reverse=True)
    idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_sorted))
    dcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(y_true))
    return float(dcg / idcg) if idcg > 0 else 0.0

def compute_ap(relevant_ids: Set[str], retrieved_docs: List[Dict[str, Any]], k: int = 5) -> float:
    if not relevant_ids or not retrieved_docs:
        return 0.0
    top_k = retrieved_docs[:k]
    rel = 0
    psum = 0.0
    for i, d in enumerate(top_k):
        if d.get("id") in relevant_ids:
            rel += 1
            psum += rel / (i + 1)
    return float(psum / min(len(relevant_ids), k)) if relevant_ids else 0.0


class RetrieverEvaluator:
    def __init__(self, retriever, batch_size: int = 32, candidate_k: int = 50):
        self.retriever = retriever
        self.batch_size = batch_size
        self.candidate_k = candidate_k
        self.rows: List[Dict[str, Any]] = []

    def evaluate_retrieval(self, test_file: Path, top_k: int = 5) -> Optional[Dict[str, Any]]:
        try:
            with open(test_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error("Failed to load test file %s: %s", test_file, e)
            return None

        self.rows.clear()
        n = len(data)
        for i in range(0, n, self.batch_size):
            batch = data[i : i + self.batch_size]
            qs = [x["question"] for x in batch]
            rels = [set(x["relevant_docs"]) for x in batch]
            t0 = time.perf_counter()
            if hasattr(self.retriever, "batch_retrieve"):
                outs = self.retriever.batch_retrieve(qs, top_k=top_k, candidates_k=self.candidate_k)
            else:
                outs = [self.retriever.retrieve(q, top_k=top_k) for q in qs]
            dt = (time.perf_counter() - t0) / max(1, len(qs))
            for q, rel, docs in zip(qs, rels, outs):
                retrieved_ids = [d.get("id") for d in (docs or [])]
                rr = 0.0
                for rank, d in enumerate(docs or [], 1):
                    if d.get("id") in rel:
                        rr = 1.0 / rank
                        break
                relevant_retrieved = len(set(retrieved_ids) & rel)
                prec = relevant_retrieved / top_k if top_k > 0 else 0.0
                rec = relevant_retrieved / len(rel) if rel else 0.0
                nd = compute_ndcg(rel, docs or [], k=top_k)
                ap = compute_ap(rel, docs or [], k=top_k)
                self.rows.append({
                    "question": q,
                    "precision@k": prec,
                    "recall@k": rec,
                    "mrr": rr,
                    "ndcg@k": nd,
                    "ap@k": ap,
                    "retrieval_time": dt,
                })
            logger.info("Processed %d/%d", min(i + self.batch_size, n), n)

        return self._aggregate()

    def _aggregate(self) -> Dict[str, Any]:
        if not self.rows:
            return {
                "num_queries": 0,
                "avg_precision@k": 0.0,
                "avg_recall@k": 0.0,
                "avg_mrr": 0.0,
                "avg_ndcg@k": 0.0,
                "avg_ap@k": 0.0,
                "avg_retrieval_time": 0.0,
            }
        df = pd.DataFrame(self.rows)
        return {
            "num_queries": int(len(df)),
            "avg_precision@k": float(df["precision@k"].mean()),
            "avg_recall@k": float(df["recall@k"].mean()),
            "avg_mrr": float(df["mrr"].mean()),
            "avg_ndcg@k": float(df["ndcg@k"].mean()),
            "avg_ap@k": float(df["ap@k"].mean()),
            "avg_retrieval_time": float(df["retrieval_time"].mean()),
        }

    def save_reports(self, metrics: Dict[str, Any], prefix: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not metrics or metrics.get("num_queries", 0) == 0:
            logger.warning("No metrics to save for %s", prefix)
            return
        payload = {
            "metadata": {
                "timestamp_utc": _now_iso(),
                "batch_size_eval": self.batch_size,
                "candidate_k_eval": self.candidate_k,
                **(metadata or {}),
            },
            "metrics": metrics,
        }
        out = ABLATION_METRICS_DIR / f"{prefix}_metrics.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("Saved: %s", out)


# ===============================
# Metadata descriptors
# ===============================
def _describe_base_retriever(base) -> Dict[str, Any]:
    # BM25
    if base.__class__.__name__ == "BM25Retriever":
        lengths = [len(c.split()) for c in getattr(base, "chunks", [])] if getattr(base, "chunks", None) else []
        return {
            "type": "bm25",
            "bm25": {
                "use_stopwords": base.use_stopwords,
                "use_stemming": base.use_stemming,
                "k1": base.k1,
                "b": base.b,
                "corpus": {
                    "num_chunks": len(getattr(base, "chunks", [])),
                    "min_len": int(min(lengths)) if lengths else 0,
                    "max_len": int(max(lengths)) if lengths else 0,
                    "avg_len": float(np.mean(lengths)) if lengths else 0.0,
                },
            },
        }
    # Rerank(BM25)
    if isinstance(base, RerankRetriever) and isinstance(base.base, BM25Retriever):
        bm = base.base
        lengths = [len(c.split()) for c in getattr(bm, "chunks", [])] if getattr(bm, "chunks", None) else []
        return {
            "type": "rerank_bm25",
            "bm25": {
                "use_stopwords": bm.use_stopwords,
                "use_stemming": bm.use_stemming,
                "k1": bm.k1,
                "b": bm.b,
                "corpus": {
                    "num_chunks": len(getattr(bm, "chunks", [])),
                    "min_len": int(min(lengths)) if lengths else 0,
                    "max_len": int(max(lengths)) if lengths else 0,
                    "avg_len": float(np.mean(lengths)) if lengths else 0.0,
                },
            },
            "reranker": _describe_reranker(base),
        }
    return {"type": "unknown"}

def _describe_reranker(rerank: Optional[RerankRetriever]) -> Optional[Dict[str, Any]]:
    if rerank is None:
        return None
    return {
        "reranker_model": rerank.reranker_name,
        "device": rerank.device,
        "max_length": rerank.max_length,
        "batch_size": rerank.batch_size,
        "max_rerank_candidates": rerank.max_rerank_candidates,
    }

def _describe_query_rewriter(qr: QueryRewriteRetriever) -> Dict[str, Any]:
    return {
        "strategy_flags": {
            "enable_cce": qr.enable_cce and (qr.qr.cce_model is not None),
            "enable_kwr": qr.enable_kwr and (qr.qr.keyword_model is not None),
            "enable_gqr": qr.enable_gqr and (qr.qr.nlp is not None),
            "enable_prf": qr.enable_prf and (qr._bm25_for_prf is not None),
            "enable_decompose": qr.enable_decompose and (qr.qr.nlp is not None),
        },
        "expansion_terms": qr.expansion_terms,
        "lang": qr.lang,
        "device": qr.device,
        "base_retriever": _describe_base_retriever(qr.base),
        "reranker": _describe_reranker(qr.base if isinstance(qr.base, RerankRetriever) else None),
    }


# ===============================
# Data utils
# ===============================
def load_and_shuffle_datasets(singlehop_path: Path, multihop_path: Path) -> List[Dict[str, Any]]:
    with open(singlehop_path, "r", encoding="utf-8") as f1:
        single = json.load(f1)
    with open(multihop_path, "r", encoding="utf-8") as f2:
        multi = json.load(f2)
    combined = single + multi
    random.shuffle(combined)
    return combined


# ===============================
# Runner: ONLY QueryRewrite over Rerank(BM25)
# ===============================
def run_rerank_bm25_only(top_k: int = 5, candidate_k: int = 50):
    ce_model = os.getenv("CE_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    bm25 = BM25Retriever()
    rerank_bm25 = RerankRetriever(base_retriever=bm25, reranker_model=ce_model, device="cpu")

    qrr = QueryRewriteRetriever(
        base_retriever=rerank_bm25,
        device="cpu",
        expansion_terms=3,
        lang="italian",
        enable_cce=True,
        enable_kwr=True,
        enable_gqr=True,
        enable_prf=True,
        enable_decompose=True,
    )
    evaluator = RetrieverEvaluator(qrr, batch_size=32, candidate_k=candidate_k)

    SINGLEHOP_FILE = DATA_DIR / "test" / "test_dataset_singlehop.json"
    MULTIHOP_FILE = DATA_DIR / "test" / "test_dataset_multihop.json"

    common_meta = {
        "experiment_prefix": "QUERY-REWRITE_rerank_bm25",
        "retrieval_top_k": top_k,
        "candidate_k": candidate_k,
        "query_rewriter": _describe_query_rewriter(qrr),
    }

    # SINGLEHOP
    print("\n--- SINGLEHOP (rerank_bm25) ---")
    single = evaluator.evaluate_retrieval(SINGLEHOP_FILE, top_k)
    if single and single["num_queries"] > 0:
        evaluator.save_reports(single, prefix="QUERY-REWRITE_rerank_bm25_singlehop", metadata={**common_meta, "dataset": "singlehop"})
        print("Saved singlehop.")

    # COMBINED (single + multihop)
    print("\n--- SINGLE+MULTIHOP (rerank_bm25) ---")
    combined = load_and_shuffle_datasets(SINGLEHOP_FILE, MULTIHOP_FILE)
    tmp = DATA_DIR / "test" / "test_dataset_combined_temp.json"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        allm = evaluator.evaluate_retrieval(tmp, top_k)
        if allm and allm["num_queries"] > 0:
            evaluator.save_reports(
                allm,
                prefix="QUERY-REWRITE_rerank_bm25_combined",
                metadata={**common_meta, "dataset": "combined", "combined_sources": ["singlehop", "multihop"], "shuffle": True},
            )
            print("Saved combined.")
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    run_rerank_bm25_only(top_k=5, candidate_k=50)
