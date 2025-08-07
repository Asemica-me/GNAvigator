# ablation/experiments.py
import numpy as np
import nltk
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
import torch
import re
import logging
import spacy
import hashlib
from typing import List, Dict, Tuple, Union
from functools import lru_cache
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryRewriteRetriever:
    def __init__(
        self,
        base_retriever=None,
        device=None,
        expansion_terms=3,
        lang="italian",
        enable_cce=True,
        enable_kwr=True,
        enable_gqr=True,
        enable_prf=True,
        enable_decompose=True
    ):
        """
        Enhanced query rewriting retriever with multiple strategies
        
        Args:
            base_retriever: Base retriever instance
            device: Torch device
            expansion_terms: Number of expansion terms for KWR
            lang: Language for NLP processing
            enable_*: Flags to enable/disable specific strategies
        """
        self.base = base_retriever or DenseRetriever(device=device)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.expansion_terms = expansion_terms
        self.lang = lang
        self.enable_cce = enable_cce
        self.enable_kwr = enable_kwr
        self.enable_gqr = enable_gqr
        self.enable_prf = enable_prf
        self.enable_decompose = enable_decompose
        
        # Initialize NLP components
        self._init_nlp_models()
        
        # Initialize caches
        self.rewrite_cache = {}
        self.embed_cache = {}
        
        # Initialize keyword model
        self.keyword_model = KeyBERT(model=SentenceTransformer(
            "paraphrase-multilingual-mpnet-base-v2", 
            device=str(self.device)
        ))
        
        # Initialize pseudo-relevance feedback model
        self.bm25 = self._init_bm25_index()

    def _init_nlp_models(self):
        """Initialize NLP models with error handling"""
        try:
            # Core Content Extraction (CCE) model
            self.cce_tokenizer = AutoTokenizer.from_pretrained("gsarti/bart-base-it")
            self.cce_model = AutoModelForSeq2SeqLM.from_pretrained(
                "gsarti/bart-base-it"
            ).to(self.device).eval()
            
            # Embedding model for Keyword Expansion (KWR)
            self.embed_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
            self.embed_model = AutoModel.from_pretrained(
                "intfloat/multilingual-e5-large"
            ).to(self.device).eval()
            
            # Spacy for linguistic features
            if self.lang == "italian":
                try:
                    self.nlp = spacy.load("it_core_news_sm")
                except:
                    spacy.cli.download("it_core_news_sm")
                    self.nlp = spacy.load("it_core_news_sm")
            else:
                self.nlp = spacy.load("en_core_web_sm")
                
            logger.info("NLP models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize NLP models: {str(e)}")
            raise RuntimeError("Model initialization failed") from e

    def _init_bm25_index(self):
        """Initialize BM25 index for pseudo-relevance feedback"""
        if not hasattr(self.base, 'chunks'):
            return None
            
        tokenized_chunks = [
            self._preprocess_text(chunk) 
            for chunk in self.base.chunks
        ]
        return BM25Okapi(tokenized_chunks)

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25 indexing"""
        doc = self.nlp(text.lower())
        return [
            token.lemma_ for token in doc
            if not token.is_stop 
            and not token.is_punct 
            and len(token.text) > 1
        ]

    def retrieve(
        self, 
        question: str, 
        top_k: int = 5,
        strategy: str = "all",
        fusion_method: str = "rrf"
    ) -> List[Dict]:
        """
        Retrieve documents using query rewriting strategies
        
        Args:
            question: Original query
            top_k: Number of results to return
            strategy: Rewriting strategy to use (all, single, hybrid)
            fusion_method: Result fusion method (rrf, score_fusion)
            
        Returns:
            List of retrieved documents
        """
        # Generate rewritten queries
        queries = self.generate_rewrites(question, strategy)
        
        # Retrieve documents for each query
        all_results = []
        for q in queries:
            try:
                docs = self.base.retrieve(q, top_k=top_k*3)
                if docs:
                    # Add query context to results
                    for doc in docs:
                        doc['source_query'] = q
                    all_results.extend(docs)
            except Exception as e:
                logger.error(f"Retrieval failed for query '{q}': {str(e)}")
        
        # Fuse and deduplicate results
        return self.fuse_results(all_results, top_k, fusion_method)

    def generate_rewrites(
        self, 
        question: str, 
        strategy: str = "all"
    ) -> List[str]:
        """
        Generate rewritten queries using multiple strategies
        
        Args:
            question: Original query
            strategy: Rewriting strategy (all, core, keywords, etc.)
            
        Returns:
            List of rewritten queries
        """
        # Check cache first
        cache_key = f"{strategy}_{hashlib.md5(question.encode()).hexdigest()}"
        if cache_key in self.rewrite_cache:
            return self.rewrite_cache[cache_key]
        
        rewrites = []
        
        # Always include original query
        rewrites.append(question)
        
        # Apply selected strategies
        if strategy == "all" or strategy == "core" and self.enable_cce:
            rewrites.append(self.core_content_extraction(question))
            
        if strategy == "all" or strategy == "keywords" and self.enable_kwr:
            rewrites.append(self.keyword_rewriting(question))
            rewrites.extend(self.keyword_expansion(question))
            
        if strategy == "all" or strategy == "general" and self.enable_gqr:
            rewrites.append(self.general_query_rewriting(question))
            
        if strategy == "all" or strategy == "prf" and self.enable_prf:
            rewrites.append(self.pseudo_relevance_feedback(question))
            
        if strategy == "all" or strategy == "decompose" and self.enable_decompose:
            rewrites.extend(self.query_decomposition(question))
        
        # Add hybrid rewrites
        if strategy == "hybrid":
            core = self.core_content_extraction(question)
            keywords = self.keyword_rewriting(question)
            rewrites.append(f"{core} {keywords}")
            
        # Deduplicate while preserving order
        unique_queries = []
        seen = set()
        for q in rewrites:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)
        
        # Cache and return
        self.rewrite_cache[cache_key] = unique_queries
        return unique_queries

    def core_content_extraction(self, question: str) -> str:
        """Extract core content using seq2seq model (CCE)"""
        try:
            inputs = self.cce_tokenizer(
                f"estrai il contenuto principale: {question}",
                return_tensors="pt",
                max_length=64,
                truncation=True
            ).to(self.device)

            outputs = self.cce_model.generate(
                inputs.input_ids,
                max_new_tokens=64,
                num_beams=5,
                early_stopping=True
            )
            return self.cce_tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
        except Exception as e:
            logger.error(f"CCE failed: {str(e)}")
            return question

    def keyword_rewriting(self, question: str) -> str:
        """Extract important keywords (KWR)"""
        try:
            # Use KeyBERT for multilingual keyword extraction
            keywords = self.keyword_model.extract_keywords(
                question,
                keyphrase_ngram_range=(1, 2),
                stop_words=list(self.nlp.Defaults.stop_words),
                top_n=self.expansion_terms
            )
            return " ".join([kw[0] for kw in keywords])
        except Exception as e:
            logger.error(f"Keyword extraction failed: {str(e)}")
            return question

    def keyword_expansion(self, question: str) -> List[str]:
        """Generate query variations using keyword expansion"""
        variations = []
        try:
            # Extract keywords
            keywords = self.keyword_rewriting(question).split()
            
            # Generate combinations
            if len(keywords) >= 2:
                # Pairwise combinations
                for i in range(len(keywords)):
                    for j in range(i+1, len(keywords)):
                        variations.append(f"{keywords[i]} {keywords[j]}")
            
            # Synonym expansion
            for keyword in keywords:
                # This would be replaced with actual synonym lookup
                synonyms = self._get_synonyms(keyword)
                for syn in synonyms[:2]:
                    variations.append(question.replace(keyword, syn))
                    
            return variations
        except Exception as e:
            logger.error(f"Keyword expansion failed: {str(e)}")
            return []

    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms using word embeddings (simplified)"""
        try:
            # Get embedding for target word
            word_embed = self._embed_text([word])[0]
            
            # Find similar words in vocabulary
            vocab = list(self.nlp.vocab.strings)
            vocab_embeds = self._embed_text(vocab[:5000])  # Limit for demo
            
            # Calculate similarities
            similarities = np.dot(vocab_embeds, word_embed) / (
                np.linalg.norm(vocab_embeds, axis=1) * np.linalg.norm(word_embed) + 1e-9
            )
            
            # Get top similar words
            top_indices = np.argsort(similarities)[::-1][1:4]  # Skip self
            return [vocab[i] for i in top_indices]
        except:
            return []

    def general_query_rewriting(self, question: str) -> str:
        """General query rewriting (GQR) with linguistic normalization"""
        try:
            doc = self.nlp(question)
            
            # Handle questions
            if any(token.text.lower() in ["chi", "cosa", "dove", "quando", "perché"] for token in doc):
                return self._rewrite_question(doc)
            
            # Handle imperative statements
            if any(token.dep_ == "ROOT" and token.tag_ == "VERB" for token in doc):
                return self._rewrite_imperative(doc)
            
            # Default normalization
            cleaned = []
            for token in doc:
                if not token.is_stop and not token.is_punct:
                    cleaned.append(token.lemma_)
            return " ".join(cleaned)
        except Exception as e:
            logger.error(f"GQR failed: {str(e)}")
            return question

    def _rewrite_question(self, doc) -> str:
        """Rewrite questions into declarative form"""
        root = [token for token in doc if token.dep_ == "ROOT"][0]
        subject = next((token for token in root.children if token.dep_ == "nsubj"), None)
        
        if subject:
            return f"{subject.text} {root.lemma_} {' '.join([t.text for t in root.children if t.dep_ not in ['nsubj', 'punct']])}"
        return doc.text

    def _rewrite_imperative(self, doc) -> str:
        """Rewrite imperative statements into queries"""
        root = [token for token in doc if token.dep_ == "ROOT"][0]
        objects = [token.text for token in root.children if token.dep_ in ['dobj', 'attr']]
        return f"{root.lemma_} {' '.join(objects)}"

    def pseudo_relevance_feedback(self, question: str) -> str:
        """Pseudo-relevance feedback (PRF) query expansion"""
        if not self.bm25:
            return question
            
        try:
            # Get initial results
            initial_results = self.base.retrieve(question, top_k=5)
            if not initial_results:
                return question
                
            # Extract text from results
            texts = [doc.get('content', '') for doc in initial_results]
            
            # Tokenize query
            tokenized_query = self._preprocess_text(question)
            
            # Get expansion terms from top documents
            expansion_terms = []
            for text in texts:
                doc_terms = self._preprocess_text(text)
                for term in doc_terms:
                    if term not in tokenized_query:
                        expansion_terms.append(term)
            
            # Get most frequent expansion terms
            term_counts = defaultdict(int)
            for term in expansion_terms:
                term_counts[term] += 1
            sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Add top expansion terms
            expanded_query = question
            for term, _ in sorted_terms[:self.expansion_terms]:
                expanded_query += f" {term}"
                
            return expanded_query
        except Exception as e:
            logger.error(f"PRF failed: {str(e)}")
            return question

    def query_decomposition(self, question: str) -> List[str]:
        """Decompose complex queries into sub-questions"""
        try:
            if " e " not in question and " o " not in question:
                return [question]
                
            doc = self.nlp(question)
            conjunctions = [token for token in doc if token.dep_ == "cc"]
            
            if not conjunctions:
                return [question]
                
            sub_queries = []
            current = []
            for token in doc:
                if token in conjunctions:
                    sub_queries.append(" ".join(current))
                    current = []
                else:
                    current.append(token.text)
            sub_queries.append(" ".join(current))
            
            return [q.strip() for q in sub_queries if q.strip()]
        except Exception as e:
            logger.error(f"Query decomposition failed: {str(e)}")
            return [question]

    @lru_cache(maxsize=1000)
    def _embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings with caching"""
        if isinstance(text, str):
            text = [text]
            
        # Check cache
        cache_keys = [hashlib.md5(t.encode()).hexdigest() for t in text]
        cached = [self.embed_cache.get(key) for key in cache_keys]
        
        if all(c is not None for c in cached):
            return np.array(cached)
            
        # Compute missing embeddings
        to_compute = []
        compute_indices = []
        for i, (t, key) in enumerate(zip(text, cache_keys)):
            if key not in self.embed_cache:
                to_compute.append(t)
                compute_indices.append(i)
        
        if to_compute:
            inputs = self.embed_tokenizer(
                to_compute,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.embed_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                
            for idx, emb in zip(compute_indices, embeddings):
                key = cache_keys[idx]
                self.embed_cache[key] = emb
        
        # Return in original order
        return np.array([self.embed_cache[key] for key in cache_keys])

    def fuse_results(
        self, 
        results: List[Dict], 
        top_k: int,
        method: str = "rrf"
    ) -> List[Dict]:
        """
        Fuse results from multiple queries
        
        Args:
            results: List of document results
            top_k: Number of results to return
            method: Fusion method (rrf, score_fusion)
            
        Returns:
            Fused and deduplicated results
        """
        # Deduplicate documents
        doc_map = {}
        for doc in results:
            doc_id = doc.get('id') or doc.get('chunk_id') or hash(doc.get('content', ''))
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    'doc': doc,
                    'scores': [],
                    'sources': set(),
                    'ranks': []
                }
            doc_map[doc_id]['scores'].append(doc.get('score', 0))
            doc_map[doc_id]['sources'].add(doc.get('source_query', 'unknown'))
            doc_map[doc_id]['ranks'].append(len(doc_map[doc_id]['ranks']) + 1)
        
        # Apply fusion method
        if method == "rrf":
            return self._rrf_fusion(doc_map, top_k)
        else:  # score_fusion
            return self._score_fusion(doc_map, top_k)

    def _rrf_fusion(
        self, 
        doc_map: Dict, 
        top_k: int,
        k: int = 60
    ) -> List[Dict]:
        """Reciprocal Rank Fusion"""
        fused_scores = {}
        for doc_id, data in doc_map.items():
            rrf_score = 0
            for rank in data['ranks']:
                rrf_score += 1 / (k + rank)
            fused_scores[doc_id] = rrf_score
        
        # Sort by RRF score
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        return [doc_map[doc_id]['doc'] for doc_id in sorted_ids[:top_k]]

    def _score_fusion(
        self, 
        doc_map: Dict, 
        top_k: int
    ) -> List[Dict]:
        """Score fusion with source weighting"""
        fused_scores = {}
        for doc_id, data in doc_map.items():
            # Weight scores by number of sources
            source_weight = 1 + 0.2 * (len(data['sources']) - 1)
            fused_scores[doc_id] = source_weight * np.mean(data['scores'])
        
        # Sort by fused score
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        return [doc_map[doc_id]['doc'] for doc_id in sorted_ids[:top_k]]

    def batch_retrieve(
        self, 
        questions: List[str], 
        top_k: int = 5,
        strategy: str = "all",
        fusion_method: str = "rrf"
    ) -> List[List[Dict]]:
        """
        Batch retrieve with query rewriting
        
        Args:
            questions: List of original queries
            top_k: Number of results per query
            strategy: Rewriting strategy
            fusion_method: Result fusion method
            
        Returns:
            List of results for each query
        """
        return [
            self.retrieve(q, top_k, strategy, fusion_method)
            for q in questions
        ]