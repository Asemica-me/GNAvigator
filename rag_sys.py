import asyncio
import gc
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional
import hashlib

import backoff
import nltk
import numpy as np
from dotenv import load_dotenv
from mistralai import Mistral, SystemMessage, UserMessage
# from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from vector_store import VectorDatabaseManager

load_dotenv()
LLM = os.getenv("GEN_MODEL")
logger = logging.getLogger(__name__)


class MistralLLM:
    def __init__(
        self,
        api_key: str,
        model_name: str = LLM,
        max_retries: int = 5,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        max_concurrency: int = 5,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.citation_regex = re.compile(r"\[(\d+)\]")
        self.url_to_id = {}
        self.id_to_url = {}
        self._client = None  # Use lazy initialization
        self.last_used = time.time()

    @property
    def client(self):
        """Lazy initialization of Mistral client"""
        if self._client is None:
            self._client = Mistral(api_key=self.api_key)
        self.last_used = time.time()
        return self._client

    def clear_cache(self):
        """Release client resources and reset connection"""
        if self._client:
            try:
                # Try to close any existing connections
                if hasattr(self._client, "close"):
                    self._client.close()
            except Exception as e:
                logger.warning(f"Error closing Mistral client: {e}")
            finally:
                self._client = None
        gc.collect()
        logger.info("MistralLLM cache cleared")

    def _generate_url_id(self, url: str) -> str:
        """Generate unique ID from URL using hash"""
        return hashlib.md5(url.encode()).hexdigest()[:12]

    def _build_rag_prompt(
        self,
        question: str,
        context: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Construct RAG system prompt with context"""
        # RESET per query to prevent conflicts
        self.url_to_id = {}
        self.id_to_url = {}
        system_content = """
        Sei un assistente virtuale incaricato di rispondere a domande sul manuale operativo del Geoportale Nazionale Archeologia (GNA), disponibile all'indirizzo: https://gna.cultura.gov.it/wiki/index.php/Pagina_principale, e gestito dall'Istituto Centrale per il Catalogo e la Documentazione (ICCD).

        Segui sempre queste regole:
        1. Non rispondere a una domanda con un'altra domanda.
        2. Rispondi **sempre** in italiano, indipendentemente dalla lingua della domanda, a meno che l'utente non richieda esplicitamente un'altra lingua.
        3. Cita le fonti utilizzando la notazione [numero] dove:
            - Le fonti sono fornite nel contesto della domanda e sono numerate in ordine crescente;
            - Usa numeri diversi per fonti diverse;
            - Non includere mai l'URL nel corpo della risposta;
        4. Alla fine della risposta, aggiungi un elenco di riferimenti con il seguente formato, su righe separate:
            [ID] URL_completo
        5. Se non hai informazioni sufficienti per rispondere, rispondi "Non ho informazioni sufficienti".

        Le tue risposte devono essere sempre:
        - Disponibili, professionali e naturali
        - Grammaticalmente corrette e coerenti
        - Espresse con frasi semplici, evitando formulazioni complesse o frammentate
        - Complete e chiare, evitando di lasciare domande senza risposta
        """

        # Build context string with grouped sources
        context_parts = []
        for idx, source_group in enumerate(context, start=1):
            url = source_group["source"]
            # Generate unique ID for this URL
            url_id = self._generate_url_id(url)
            self.url_to_id[url] = url_id
            self.id_to_url[url_id] = url

            contents = source_group["contents"]
            contents_str = "\n".join(
                f"CONTENUTO {j}: {content}"
                for j, content in enumerate(contents, start=1)
            )
            context_parts.append(f"FONTE {url_id}: {url}\n{contents_str}")

        context_str = "\n\n".join(context_parts)

        # Build message list with conversation history
        messages = [SystemMessage(content=system_content)]

        # Add chat history if exists
        if chat_history:
            for msg in chat_history:
                if msg["role"] == "user":
                    messages.append(UserMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(SystemMessage(content=msg["content"]))

        # Add current context and question
        messages.append(
            UserMessage(content=f"CONTESTO:\n{context_str}\n\nDOMANDA: {question}")
        )

        return messages

    @backoff.on_exception(
        backoff.expo, Exception, max_tries=3, jitter=backoff.full_jitter
    )
    async def generate_async(
        self,
        question: str,
        context: List[Dict[str, Any]],
        temperature: float = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generate answer with RAG context using async streaming"""
        async with self.semaphore:
            messages = None
            stream = None
            try:
                messages = self._build_rag_prompt(question, context, chat_history)
                stream = await self.client.chat.stream_async(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature or self.temperature,
                    max_tokens=self.max_tokens,
                )

                collected_messages = []
                async for chunk in stream:
                    if chunk.data.choices[0].delta.content:
                        collected_messages.append(chunk.data.choices[0].delta.content)

                response = "".join(collected_messages).strip()
                # Validate citations against number of unique sources
                return self._format_references(response)

            except Exception as e:
                logger.error(f"Generation failed: {str(e)}")
                # Clear client on error to force reconnect
                self.clear_cache()
                raise
            finally:
                # Clean up intermediate resources
                if messages is not None:
                    del messages
                if stream is not None:
                    del stream
                self.url_to_id = {}
                self.id_to_url = {}
                gc.collect()

    def _format_references(self, response: str) -> str:
        """Format references section and ensure proper citation usage"""
        # Step 1: Extract all citation IDs from the response
        citation_ids = set()
        for match in re.finditer(r"\[([a-f0-9]{6})\]", response):
            citation_id = match.group(1)
            citation_ids.add(citation_id)

        # Step 2: Build references section
        references = []
        for cid in sorted(citation_ids):
            if cid in self.id_to_url:
                references.append(f"[{cid}] {self.id_to_url[cid]}")

        # Step 3: Append references if any exist
        if references:
            response = response.rstrip() + "\n\n" + "\n".join(references)

        return response


class RAGOrchestrator:
    """Orchestrates RAG workflow with enhanced memory management"""

    def __init__(self, mistral_api_key: str, device: str = None):
        self.mistral_api_key = mistral_api_key
        self._vector_db = None
        self._llm = None
        self.last_cleanup = time.time()
        self.query_count = 0
        self.tokenizer = self._initialize_tokenizer()
        self.device = device

    @property
    def vector_db(self):
        """Lazy initialization of vector database"""
        if self._vector_db is None:
            self._vector_db = VectorDatabaseManager(device=self.device)
        return self._vector_db

    @property
    def llm(self):
        """Lazy initialization of LLM"""
        if self._llm is None:
            self._llm = MistralLLM(api_key=self.mistral_api_key)
        return self._llm

    def clear_cache(self, full: bool = False):
        """Release memory resources with optional deep cleanup"""
        logger.info("Clearing RAG orchestrator cache")

        # Clear LLM resources
        if self._llm:
            self._llm.clear_cache()

        # Clear vector DB resources if possible
        if self._vector_db and hasattr(self._vector_db, "clear_cache"):
            self._vector_db.clear_cache()

        # Optional deep cleanup
        if full:
            self._vector_db = None
            self._llm = None
            logger.info("Full cache clearance completed")

        # Release any dangling references
        gc.collect()

        # Reset counters
        self.query_count = 0
        self.last_cleanup = time.time()

    async def query(
        self,
        question: str,
        top_k: int = 5,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """End-to-end RAG query execution with memory management"""
        try:
            context_chunks = self.vector_db.query(question, top_k=top_k)

            # Group chunks by URL
            url_to_contents = {}
            for chunk in context_chunks:
                url = chunk["source"]
                if url not in url_to_contents:
                    url_to_contents[url] = []
                url_to_contents[url].append(chunk["content"])

            # Create grouped context structure
            grouped_context = []
            for url, contents in url_to_contents.items():
                grouped_context.append({"source": url, "contents": contents})

            # Pass grouped context to LLM
            answer = await self.llm.generate_async(
                question=question, context=grouped_context, chat_history=chat_history
            )

            # Create source map {index: url}
            source_map = {}
            for group in grouped_context:
                url = group["source"]
                url_id = self.llm._generate_url_id(url)
                source_map[url_id] = url

            # Increment and check for cleanup
            self.query_count += 1
            if self.query_count >= 10 or time.time() - self.last_cleanup > 300:
                self.clear_cache(full=self.query_count >= 30)

            return {
                "question": question,
                "answer": answer,
                "sources": source_map,
                "context": context_chunks,
                "updated_history": self._update_history(chat_history, question, answer),
            }

        except Exception as e:
            logger.error(f"Query failed: {e}")
            # Force cleanup on error
            self.clear_cache(full=True)
            raise
        finally:
            # Clean up intermediate resources
            del context_chunks
            gc.collect()

    def _update_history(
        self, history: Optional[List[Dict[str, str]]], question: str, answer: str
    ) -> List[Dict[str, str]]:
        """Update chat history with new exchange"""
        new_history = history.copy() if history else []
        new_history.append({"role": "user", "content": question})
        new_history.append({"role": "assistant", "content": answer})

        # Optional: Implement history truncation
        # if len(new_history) > 10:  # Keep last 5 exchanges (10 messages)
        #     new_history = new_history[-10:]

        return new_history

    async def initialize_vector_store(self, sitemap_path: str, base_domain: str):
        """Initialize vector store with documents"""
        try:
            await self.vector_db.process_and_store_chunks(sitemap_path, base_domain)
        finally:
            # Clean up after initialization
            self.clear_cache(full=False)

    async def sample_documents(self, n: int) -> list:
        """Sample documents from the vector store"""
        return await asyncio.get_running_loop().run_in_executor(
            None, self.vector_db.sample_documents, n
        )

    async def get_documents_by_ids(self, doc_ids: list) -> list:
        """Retrieve documents by their IDs"""
        return await asyncio.get_running_loop().run_in_executor(
            None, self.vector_db.get_documents_by_ids, doc_ids
        )

    def _initialize_tokenizer(self):
        """Initialize tokenizer for approximate token counting"""
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(os.getenv("EMBEDDING_MODEL"))
        except ImportError:
            # Fallback to simple whitespace tokenizer
            return lambda text: text.split()

    async def retrieve_docs(self, question: str, top_k: int = 5) -> list:
        """Retrieve documents without generating full response"""
        # Run synchronous vector store query in executor
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.vector_db.query, question, top_k)

    def retrieve_docs_batch(self, questions: list[str], top_k: int = 5) -> list:
        """Retrieve documents without generating full response"""
        return self.vector_db.query_batch(questions, top_k)

    def tokenize(self, text: str) -> list:
        """Tokenize text for approximate token counting"""
        return self.tokenizer(text) if callable(self.tokenizer) else text.split()


# class HybridRetriever:
#     def __init__(self, chunks):
#         self.chunks = chunks
#         self.tokenized_chunks = [
#             nltk.word_tokenize(chunk["content"]) for chunk in chunks
#         ]
#         self.bm25 = BM25Okapi(self.tokenized_chunks)
#         self.cross_encoder = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L6-H384-v1")

#     def retrieve(self, query: str, top_k: int = 5):
#         # BM25 retrieval
#         tokenized_query = nltk.word_tokenize(query)
#         bm25_scores = self.bm25.get_scores(tokenized_query)
#         bm25_indices = np.argsort(bm25_scores)[::-1][: top_k * 3]  # Get 3x candidates

#         # Rerank with cross-encoder
#         pairs = [(query, self.chunks[i]["content"]) for i in bm25_indices]
#         ce_scores = self.cross_encoder.predict(pairs)

#         # Combine scores
#         combined_scores = [
#             0.7 * ce_scores[i] + 0.3 * bm25_scores[bm25_indices[i]]
#             for i in range(len(bm25_indices))
#         ]

#         # Get top k results
#         best_indices = np.argsort(combined_scores)[::-1][:top_k]
#         return [self.chunks[bm25_indices[i]] for i in best_indices]
