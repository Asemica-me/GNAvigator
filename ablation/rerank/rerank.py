# ablation/experiments.py
import sys
from pathlib import Path
import numpy as np
import nltk
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
import os
from dotenv import load_dotenv
import torch
from collections import defaultdict  
import re

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vector_store import VectorDatabaseManager
from dotenv import load_dotenv

class VectorDatabaseWrapper:
    def __init__(self, vector_db: VectorDatabaseManager):
        self.vector_db = vector_db
        self._cached_embeddings = {}
        
    def query(self, question: str, top_k: int = 5) -> list:
        # Implement proper caching here
        if question in self._cached_embeddings:
            return self._cached_embeddings[question]
        else:
            results = self.vector_db.query(question, top_k)
            if results is None:
                results = []
            self._cached_embeddings[question] = results
            return results
                
    def query_batch(self, questions: list[str], top_k: int = 5) -> list:
                # Similar batching implementation with proper caching
                pass

# ablation/experiments.py
import numpy as np
from sentence_transformers import CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from collections import defaultdict
import logging
import hashlib
from typing import List, Dict, Tuple, Union, Optional
import os
from dotenv import load_dotenv
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class RerankRetriever:
    def __init__(
        self,
        base_retriever=None,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
        cache_size: int = 1000
    ):
        """
        Enhanced reranking retriever with multiple strategies
        
        Args:
            base_retriever: Retriever with .retrieve() method (default: DenseRetriever)
            reranker_model: Cross-encoder model path
            device: Torch device (auto-detected if None)
            max_length: Max token length for reranker
            batch_size: Batch size for reranking
            cache_size: Size of query-doc score cache
        """
        self.base = base_retriever if base_retriever else DenseRetriever(device=device)
        
        # Auto device detection
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize reranker
        self.reranker_name = reranker_model
        self.max_length = max_length
        self.batch_size = batch_size
        self._init_reranker()
        
        # Initialize cache
        self.score_cache = lru_cache(maxsize=cache_size)(self._compute_scores_uncached)
        
    def _init_reranker(self):
        """Initialize reranker model with optimized settings"""
        try:
            logger.info(f"Loading reranker: {self.reranker_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.reranker_name)
            self.reranker = AutoModelForSequenceClassification.from_pretrained(
                self.reranker_name
            ).to(self.device).eval()
            logger.info(f"Reranker loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load reranker: {str(e)}")
            raise RuntimeError("Reranker initialization failed") from e

    def _compute_scores_uncached(
        self, 
        query: str, 
        documents: List[str]
    ) -> List[float]:
        """Compute reranker scores without caching"""
        # Prepare input pairs
        pairs = [(query, doc) for doc in documents]
        
        # Batch processing
        scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i:i+self.batch_size]
            
            # Tokenize with truncation and padding
            features = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Compute scores
            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = self.reranker(**features)
                batch_scores = outputs.logits[:, 1].float().cpu().numpy()
            
            scores.extend(batch_scores.tolist())
        
        return scores
    
    def _get_scores(
        self, 
        query: str, 
        documents: List[str]
    ) -> List[float]:
        """Get scores with caching using query-doc hash"""
        # Generate unique cache keys
        query_hash = hashlib.md5(query.encode()).hexdigest()
        doc_hashes = [hashlib.md5(doc.encode()).hexdigest() for doc in documents]
        cache_keys = [f"{query_hash}_{doc_hash}" for doc_hash in doc_hashes]
        
        # Check cache and compute missing scores
        cached_scores = []
        to_compute = []
        compute_indices = []
        
        for idx, key in enumerate(cache_keys):
            if key in self.score_cache.cache:
                cached_scores.append((idx, self.score_cache.cache[key]))
            else:
                to_compute.append(documents[idx])
                compute_indices.append(idx)
        
        # Compute missing scores
        if to_compute:
            computed_scores = self._compute_scores_uncached(query, to_compute)
            for idx, score in zip(compute_indices, computed_scores):
                self.score_cache(cache_keys[idx], score)  # Add to cache
                cached_scores.append((idx, score))
        
        # Sort scores by original index
        cached_scores.sort(key=lambda x: x[0])
        return [score for _, score in cached_scores]

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
        strategy: str = "cross_encoder",
        diversity_penalty: float = 0.5
    ) -> List[Dict]:
        """
        Rerank candidates using specified strategy
        
        Args:
            query: Input query
            candidates: List of candidate documents (dicts with 'content' or 'text')
            top_k: Number of results to return
            strategy: Reranking strategy (cross_encoder|mmr|reciprocal_rank|diversity)
            diversity_penalty: Penalty for similar documents (0-1)
            
        Returns:
            List of reranked documents
        """
        # Extract document texts
        documents = []
        for cand in candidates:
            text = cand.get('content') or cand.get('text') or cand.get('document', '')
            documents.append(text)
        
        if strategy == "cross_encoder":
            scores = self._get_scores(query, documents)
            sorted_indices = np.argsort(scores)[::-1]
            return [candidates[i] for i in sorted_indices[:top_k]]
        
        elif strategy == "mmr":
            return self._mmr_rerank(query, candidates, top_k, diversity_penalty)
        
        elif strategy == "reciprocal_rank":
            return self._reciprocal_rank_fusion(candidates, top_k)
        
        elif strategy == "diversity":
            return self._diversity_rerank(candidates, top_k, diversity_penalty)
        
        else:
            raise ValueError(f"Unknown reranking strategy: {strategy}")

    def _mmr_rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int,
        lambda_param: float = 0.5
    ) -> List[Dict]:
        """
        Maximal Marginal Relevance reranking
        
        Args:
            query: Input question
            candidates: List of candidate documents
            top_k: Number of results to return
            lambda_param: 0-1 balancing relevance vs diversity
        
        Returns:
            MMR-reranked documents
        """
        # Get relevance scores
        documents = [c.get('content', '') for c in candidates]
        rel_scores = np.array(self._get_scores(query, documents))
        
        # Compute document-document similarity matrix
        doc_embeddings = [self._embed_text(doc) for doc in documents]
        sim_matrix = np.zeros((len(documents), len(documents)))
        for i in range(len(documents)):
            for j in range(i+1, len(documents)):
                sim = np.dot(doc_embeddings[i], doc_embeddings[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
        
        # MMR selection
        selected = []
        remaining = set(range(len(candidates)))
        
        # Start with most relevant document
        first_idx = np.argmax(rel_scores)
        selected.append(first_idx)
        remaining.remove(first_idx)
        
        while len(selected) < min(top_k, len(candidates)):
            mmr_scores = []
            for idx in remaining:
                # Max similarity to already selected docs
                max_sim = max(sim_matrix[idx, s] for s in selected) if selected else 0
                # MMR score calculation
                mmr_score = lambda_param * rel_scores[idx] - (1 - lambda_param) * max_sim
                mmr_scores.append((idx, mmr_score))
            
            # Select document with highest MMR score
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        return [candidates[i] for i in selected]

    def _reciprocal_rank_fusion(
        self,
        candidates: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """Combine multiple rankings using reciprocal rank fusion"""
        # Assume candidates have 'rank' from different systems
        # For demo: simulate multiple rankings
        rankings = [
            sorted(candidates, key=lambda x: x.get('score', 0), reverse=True),
            sorted(candidates, key=lambda x: x.get('bm25_score', 0), reverse=True)
        ]
        
        rrf_scores = defaultdict(float)
        k = 60  # RRF constant
        
        for rank_list in rankings:
            for rank, doc in enumerate(rank_list, 1):
                doc_id = doc.get('id', id(doc))
                rrf_scores[doc_id] += 1 / (k + rank)
        
        # Sort by RRF score
        sorted_docs = sorted(
            candidates,
            key=lambda x: rrf_scores.get(x.get('id', id(x)), 0),
            reverse=True
        )
        return sorted_docs[:top_k]

    def _diversity_rerank(
        self,
        candidates: List[Dict],
        top_k: int,
        penalty: float = 0.5
    ) -> List[Dict]:
        """Promote diverse documents using embedding clustering"""
        from sklearn.cluster import KMeans
        import numpy as np
        
        # Extract embeddings from candidates or compute
        if 'embedding' in candidates[0]:
            embeddings = np.array([c['embedding'] for c in candidates])
        else:
            texts = [c.get('content', '') for c in candidates]
            embeddings = np.array([self._embed_text(t) for t in texts])
        
        # Cluster documents
        n_clusters = min(top_k, len(candidates))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
        
        # Select top document from each cluster
        selected = []
        cluster_labels = kmeans.labels_
        
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) > 0:
                # Select highest-scoring document in cluster
                best_idx = cluster_indices[0]
                if 'score' in candidates[0]:
                    scores = [candidates[i].get('score', 0) for i in cluster_indices]
                    best_idx = cluster_indices[np.argmax(scores)]
                selected.append(candidates[best_idx])
        
        return selected[:top_k]

    @lru_cache(maxsize=1000)
    def _embed_text(self, text: str) -> np.ndarray:
        """Create embedding for text (cached)"""
        # Using mean pooling for simplicity
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.reranker(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            return last_hidden.mean(dim=1).cpu().numpy().flatten()

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        candidates_k: int = 50,
        strategy: str = "cross_encoder",
        **kwargs
    ) -> List[Dict]:
        """
        Retrieve and rerank documents
        
        Args:
            question: Input query
            top_k: Final number of results
            candidates_k: Number of candidates to rerank
            strategy: Reranking strategy
            kwargs: Strategy-specific parameters
            
        Returns:
            Reranked documents
        """
        # Retrieve candidates
        candidates = self.base.retrieve(question, top_k=candidates_k)
        
        # Rerank top candidates
        return self.rerank(question, candidates, top_k, strategy, **kwargs)

    def batch_retrieve(
        self,
        questions: List[str],
        top_k: int = 5,
        candidates_k: int = 50,
        strategy: str = "cross_encoder",
        **kwargs
    ) -> List[List[Dict]]:
        """
        Batch retrieve and rerank documents
        
        Args:
            questions: List of input queries
            top_k: Final number of results per query
            candidates_k: Number of candidates to rerank per query
            strategy: Reranking strategy
            
        Returns:
            List of reranked documents for each query
        """
        # Batch retrieve candidates
        all_candidates = []
        for q in questions:
            candidates = self.base.retrieve(q, top_k=candidates_k)
            all_candidates.append(candidates)
        
        # Batch rerank
        results = []
        for q, cands in zip(questions, all_candidates):
            results.append(self.rerank(q, cands, top_k, strategy, **kwargs))
        
        return results