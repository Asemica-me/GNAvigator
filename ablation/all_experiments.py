# ablation/experiments.py
import numpy as np
import nltk
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from vector_store import VectorDatabaseManager
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
import os
from dotenv import load_dotenv
import torch
from collections import defaultdict  
import re

load_dotenv()

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

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

class DenseRetriever:
    def __init__(self, device=None):
        self.vector_db = VectorDatabaseWrapper(VectorDatabaseManager(device=device))

    def retrieve(self, question, top_k=5):
        results = self.vector_db.query(question, top_k=top_k)
        if not results:
            results = []
        return [{
            'id': item['id'],
            'chunk_id': item['id'],
            'score': item['score'],
            **item
        } for item in (results or [])]
    
    def query_with_scores(self, question, top_k=20):
        results = self.vector_db.query(question, top_k=top_k)
        if not results:
            results = []
        return [{
            'id': item['id'],
            'chunk_id': item['id'],
            'score': item['score'],
            **item
        } for item in (results or [])]

class BM25Retriever:
    def __init__(self):
        self.vector_db = VectorDatabaseManager()
        self.chunks = [meta['document'] for meta in self.vector_db.metadata_db]

        print(f"\n[BM25] Number of chunks: {len(self.chunks)}")
        chunk_lengths = [len(c.split()) for c in self.chunks]
        print(f"[BM25] Chunk length (words): min={min(chunk_lengths)}, max={max(chunk_lengths)}, avg={np.mean(chunk_lengths):.1f}")
        
        # Load Italian linguistic resources
        self.stop_words = set(nltk.corpus.stopwords.words('italian'))
        self.stemmer = nltk.stem.SnowballStemmer("italian")
        
        # Preprocess all chunks during initialization
        self.tokenized_chunks = [self._italian_preprocess(chunk) for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def _italian_preprocess(self, text):
        """Comprehensive Italian text preprocessing for BM25"""
        # Preserve apostrophes in Italian contractions
        text = re.sub(r"[^\w\s']", "", text)
        
        # Handle common Italian contractions
        text = re.sub(r"\b(l'|un'|all'|d'|dell'|quest'|nell')\b", "", text)
        
        # Case normalization
        text = text.lower()
        
        # Tokenize with Italian-specific rules
        tokens = nltk.word_tokenize(text, language='italian')
        
        # Remove stopwords and stem
        processed = []
        for token in tokens:
            # Skip stopwords and short tokens
            if token in self.stop_words or len(token) < 2:
                continue
                
            # Apply stemming
            stemmed = self.stemmer.stem(token)
            if stemmed:
                processed.append(stemmed)
                
        return processed

    def retrieve(self, question, top_k=5):
        tokenized_query = self._italian_preprocess(question)
        scores = self.bm25.get_scores(tokenized_query)
        indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in indices:
            item = self.vector_db.metadata_db[i].copy()
            # Add id field matching VectorDatabaseManager.query
            if 'id' not in item:
                item['id'] = item.get('chunk_id', f"bm25_{i}")
            # Add chunk_id for compatibility
            item['chunk_id'] = item['id']
            results.append(item)
        return results
    
    def query_with_scores(self, question, top_k=20):
        tokenized_query = self._italian_preprocess(question)
        scores = self.bm25.get_scores(tokenized_query)
        indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in indices:
            item = self.vector_db.metadata_db[i].copy()
            item['score'] = float(scores[i])
            # Add id field matching VectorDatabaseManager.query
            if 'id' not in item:
                item['id'] = item.get('chunk_id', f"bm25_{i}")
            # Add chunk_id for compatibility
            item['chunk_id'] = item['id']
            results.append(item)
        return results

class HybridRetriever:
    def __init__(self, rrf_k=60, device=None):
        self.dense = DenseRetriever(device=device)
        self.bm25 = BM25Retriever()
        self.rrf_k = rrf_k

    def retrieve(self, question, top_k=5, candidate_k=50):
        dense_results = self.dense.query_with_scores(question, top_k=candidate_k)
        bm25_results = self.bm25.query_with_scores(question, top_k=candidate_k)
        
        id_to_doc = {}
        for doc in dense_results + bm25_results:
            # Use 'id' as primary identifier
            id_to_doc[doc['id']] = doc
        
        dense_ids = [doc['id'] for doc in dense_results]
        bm25_ids = [doc['id'] for doc in bm25_results]
        
        # Calculate RRF scores
        rrf_scores = defaultdict(float)
        for rank, doc_id in enumerate(dense_ids, start=1):
            rrf_scores[doc_id] += 1 / (self.rrf_k + rank)
        for rank, doc_id in enumerate(bm25_ids, start=1):
            rrf_scores[doc_id] += 1 / (self.rrf_k + rank)
        
        # Sort by RRF score and return top results
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [id_to_doc[doc_id] for doc_id, _ in sorted_docs]

class RerankRetriever:
    def __init__(self, base_retriever=None, device=None):
        self.base = base_retriever if base_retriever else DenseRetriever(device=device)
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device or "cpu")

    def retrieve(self, question, top_k=5, candidates_k=20):
        candidates = self.base.retrieve(question, top_k=candidates_k)
        pairs = [(question, c.get('content', c.get('document', ''))) for c in candidates]
        scores = self.cross_encoder.predict(pairs)
        best_idx = np.argsort(scores)[::-1][:top_k]
        return [candidates[i] for i in best_idx]

class QueryRewriteRetriever:
    def __init__(self, base_retriever=None, device=None, expansion_terms=3):
        self.base = base_retriever or DenseRetriever(device=device)
        self.device = device
        self.expansion_terms = expansion_terms
        
        # Core Content Extraction (CCE) model
        self.cce_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        self.cce_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base").to(device)
        
        # Embedding model for Keyword Expansion (KWR)
        self.embed_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
        self.embed_model = AutoModel.from_pretrained("intfloat/multilingual-e5-small").to(device)

    def retrieve(self, question, top_k=5):
        """Retrieve documents using parallel independent rewritten queries"""
        queries = [
            self.general_query_rewriting(question),
            self.core_content_extraction(question),
            self.keyword_rewriting(question),
            question  # original query
        ]
        
        all_results = []
        for q in queries:
            docs = self.base.retrieve(q, top_k=top_k)
            if docs:
                all_results.extend(docs)

        # Deduplicate and rank by original scores or by frequency
        unique_docs = {}
        for doc in all_results:
            doc_id = doc.get('id') or doc.get('chunk_id')
            if doc_id not in unique_docs:
                unique_docs[doc_id] = doc
            else:
                # Optionally enhance score for docs retrieved multiple times
                unique_docs[doc_id]['score'] += doc.get('score', 0)

        sorted_docs = sorted(unique_docs.values(), key=lambda x: x['score'], reverse=True)
        return sorted_docs[:top_k]

    def general_query_rewriting(self, question):
        """Simple general rewrite removing noise (GQR)"""
        # For simplicity, returning original or lightly processed query
        return question.strip().lower()

    def core_content_extraction(self, question):
        """Extract core content (CCE)"""
        inputs = self.cce_tokenizer(
            f"extract core content: {question}",
            return_tensors="pt",
            max_length=64,
            truncation=True
        ).to(self.device)

        outputs = self.cce_model.generate(inputs.input_ids, max_new_tokens=64, num_beams=2)
        return self.cce_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def keyword_rewriting(self, question):
        """Extract important keywords (KWR)"""
        tokens = nltk.word_tokenize(question.lower(), language='italian')
        tokens = [t for t in tokens if t.isalnum() and len(t) > 2]

        if not tokens:
            return question

        query_embed = self._embed_text(question)[0]
        term_embeds = self._embed_text(tokens)

        similarities = np.dot(term_embeds, query_embed) / (
            np.linalg.norm(term_embeds, axis=1) * np.linalg.norm(query_embed) + 1e-9
        )

        top_indices = np.argsort(similarities)[::-1][:self.expansion_terms]
        keywords = [tokens[i] for i in top_indices]

        return ' '.join(keywords)

    def _embed_text(self, text):
        if isinstance(text, str):
            text = [text]

        prefixed = [f"query: {t}" for t in text]

        inputs = self.embed_tokenizer(
            prefixed,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.embed_model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings