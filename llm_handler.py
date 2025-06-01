import asyncio
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from mistralai import Mistral, SystemMessage, UserMessage
import backoff
import logging
import os
import re
import gc
import time
import weakref
from vector_store import *

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
        max_concurrency: int = 5
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.citation_regex = re.compile(r'\[(\d+)\]')
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
                if hasattr(self._client, 'close'):
                    self._client.close()
            except Exception as e:
                logger.warning(f"Error closing Mistral client: {e}")
            finally:
                self._client = None
        gc.collect()
        logger.info("MistralLLM cache cleared")

    def _build_rag_prompt(self, question: str, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Construct RAG system prompt with context"""
        system_content = """
        Sei un assistente virtuale incaricato di rispondere a domande sul manuale operativo del Geoportale Nazionale Archeologia (GNA), disponibile all'indirizzo: https://gna.cultura.gov.it/wiki/index.php/Pagina_principale, e gestito dall'Istituto Centrale per il Catalogo e la Documentazione (ICCD).

        Segui sempre queste regole:
        Non rispondere a una domanda con un'altra domanda.
        Rispondi **sempre** in italiano, indipendentemente dalla lingua della domanda, a meno che l'utente non richieda esplicitamente un'altra lingua.
        Cita le fonti utilizzando la notazione [numero] dove:
           - "numero" corrisponde esattamente all'URL della fonte nel contesto;
           - Usa numeri separati per fonti diverse;
        Se non hai informazioni sufficienti per rispondere, rispondi "Non ho informazioni sufficienti".

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
            contents = source_group["contents"]
            contents_str = "\n".join(
                f"CONTENUTO {j}: {content}" 
                for j, content in enumerate(contents, start=1)
            )
            context_parts.append(f"FONTE {idx}: {url}\n{contents_str}")
        
        context_str = "\n\n".join(context_parts)
        
        return [
            SystemMessage(content=system_content),
            UserMessage(content=f"CONTESTO:\n{context_str}\n\nDOMANDA: {question}")
        ]

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        jitter=backoff.full_jitter 
    )
    async def generate_async(
        self,
        question: str,
        context: List[Dict[str, Any]],
        temperature: float = None
    ) -> str:
        """Generate answer with RAG context using async streaming"""
        async with self.semaphore:
            try:
                messages = self._build_rag_prompt(question, context)
                stream = await self.client.chat.stream_async(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature or self.temperature,
                    max_tokens=self.max_tokens
                )

                collected_messages = []
                async for chunk in stream:
                    if chunk.data.choices[0].delta.content:
                        collected_messages.append(chunk.data.choices[0].delta.content)
                
                response = "".join(collected_messages).strip()
                # Validate citations against number of unique sources
                return self._validate_citations(response, len(context))

            except Exception as e:
                logger.error(f"Generation failed: {str(e)}")
                # Clear client on error to force reconnect
                self.clear_cache()
                raise
            finally:
                # Clean up intermediate resources
                del messages, stream
                gc.collect()

    def _validate_citations(self, response: str, context_size: int) -> str:
        """Ensure citations reference valid sources"""
        citations = set()
        for match in self.citation_regex.finditer(response):
            citation = int(match.group(1))
            if 1 <= citation <= context_size:
                citations.add(citation)
            else:
                logger.warning(f"Invalid citation detected: [{citation}]")

        if not citations:
            logger.warning("No valid citations found in response")
            return response
            
        return response


class RAGOrchestrator:
    """Orchestrates RAG workflow with enhanced memory management"""
    
    def __init__(self, mistral_api_key: str):
        self.mistral_api_key = mistral_api_key
        self._vector_db = None
        self._llm = None
        self.last_cleanup = time.time()
        self.query_count = 0

    @property
    def vector_db(self):
        """Lazy initialization of vector database"""
        if self._vector_db is None:
            self._vector_db = VectorDatabaseManager()
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
        if self._vector_db and hasattr(self._vector_db, 'clear_cache'):
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
        
    async def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
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
                grouped_context.append({
                    "source": url,
                    "contents": contents
                })
            
            # Pass grouped context to LLM
            answer = await self.llm.generate_async(
                question=question, 
                context=grouped_context
            )
            
            # Create source map {index: url}
            source_map = {}
            for idx, group in enumerate(grouped_context, start=1):
                source_map[str(idx)] = group["source"]
            
            # Increment and check for cleanup
            self.query_count += 1
            if self.query_count >= 10 or time.time() - self.last_cleanup > 300:
                self.clear_cache(full=self.query_count >= 30)
            
            return {
                "question": question,
                "answer": answer,
                "sources": source_map,
                "context": context_chunks  # Return original chunks
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

    async def initialize_vector_store(self, sitemap_path: str, base_domain: str):
        """Initialize vector store with documents"""
        try:
            await self.vector_db.process_and_store_chunks(sitemap_path, base_domain)
        finally:
            # Clean up after initialization
            self.clear_cache(full=False)