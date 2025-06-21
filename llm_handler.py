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
from aiohttp import ClientSession
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
        self._create_client()
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

    def _build_rag_prompt(self, question: str, context: List[Dict[str, Any]], chat_history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, Any]]:
        """Construct RAG system prompt with context"""
        system_content = """
        Sei un assistente virtuale incaricato di rispondere a domande sul manuale operativo del Geoportale Nazionale Archeologia (GNA), disponibile all'indirizzo: https://gna.cultura.gov.it/wiki/index.php/Pagina_principale, e gestito dall'Istituto Centrale per il Catalogo e la Documentazione (ICCD).

        Segui sempre queste regole:
        Non rispondere a una domanda con un'altra domanda.
        Rispondi **sempre** in italiano, indipendentemente dalla lingua della domanda, a meno che l'utente non richieda esplicitamente un'altra lingua.
        Cita le fonti utilizzando la notazione [numero] dove:
           - "numero" corrisponde esattamente all'URL della fonte nel contesto;
           - usa numeri separati per fonti diverse;
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
            UserMessage(
                content=f"CONTESTO:\n{context_str}\n\nDOMANDA: {question}"
            )
        )
        
        return messages
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        jitter=backoff.full_jitter 
    )

    def _create_client(self):
        if not self.client or self.client.closed:
            self.client = ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"}
            )

    async def generate_async(
        self,
        question: str,
        context: List[Dict[str, Any]],
        temperature: float = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate answer with RAG context using async streaming"""
        self._create_client() 
        # Refresh client every 10 requests
        if not hasattr(self, 'request_count'):
            self.request_count = 0
        
        self.request_count += 1
        if self.request_count % 10 == 0:
            await self.client.close()
            self.client = ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
        async with self.semaphore:
            try:
                messages = self._build_rag_prompt(question, context, chat_history)
                
                # Convert messages to Mistral API format
                api_messages = []
                for msg in messages:
                    if isinstance(msg, SystemMessage):
                        api_messages.append({"role": "system", "content": msg.content})
                    else:
                        api_messages.append({"role": "user", "content": msg.content})
                
                # Using the Mistral API directly with aiohttp
                async with self.client.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    json={
                        "model": self.model_name,
                        "messages": api_messages,
                        "temperature": temperature or self.temperature,
                        "max_tokens": self.max_tokens,
                        "stream": True
                    }
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(f"API error {response.status}: {error_text}")
                    
                    collected_messages = []
                    async for chunk in response.content.iter_any():
                        if chunk:
                            decoded_chunk = chunk.decode('utf-8')
                            for line in decoded_chunk.split('\n'):
                                if line.startswith('data:'):
                                    data = line[5:].strip()
                                    if data != '[DONE]':
                                        try:
                                            json_data = json.loads(data)
                                            if 'choices' in json_data and len(json_data['choices']) > 0:
                                                content = json_data['choices'][0]['delta'].get('content', '')
                                                if content:
                                                    collected_messages.append(content)
                                        except json.JSONDecodeError:
                                            logger.warning("Invalid JSON chunk")
                    
                    response_text = "".join(collected_messages).strip()
                    return self._validate_citations(response_text, len(context))

            except Exception as e:
                logger.error(f"Generation failed: {str(e)}")
                # Reinitialize client on error
                await self.initialize_client()
                raise
            finally:
                # Clean up intermediate resources
                del messages
                gc.collect()

    def clear_cache(self):
        """Release client resources and reset connection"""
        if self.client:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.client.close())
                else:
                    loop.run_until_complete(self.client.close())
            except Exception as e:
                logger.warning(f"Error closing Mistral client: {str(e)}")
            finally:
                self.client = None
        logger.info("MistralLLM cache cleared")

    def _validate_citations(self, response: str, context_size: int) -> str:
        """Ensure citations reference valid sources"""
        citations = set()
        for match in self.citation_regex.finditer(response):
            citation = int(match.group(1))
            if 1 <= citation <= context_size:
                citations.add(citation)
            else:
                logger.warning(f"Invalid citation detected: [{citation}]")

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
        
    async def query(self, question: str, top_k: int=5, chat_history: Optional[List[Dict[str, str]]] = None,) -> Dict[str, Any]:
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
                context=grouped_context,
                chat_history=chat_history
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
                "context": context_chunks
            }
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            # Force cleanup on error
            self.clear_cache(full=True)
            return {
                "question": question,
                "answer": f"Error processing request: {str(e)}",
                "sources": {},
                "context": []
            }
        finally:
            # Clean up intermediate resources
            if 'context_chunks' in locals():
                del context_chunks
            gc.collect()

    def _update_history(
        self,
        history: Optional[List[Dict[str, str]]],
        question: str,
        answer: str
    ) -> List[Dict[str, str]]:
        """Update chat history with new exchange"""
        new_history = history.copy() if history else []
        new_history.append({"role": "user", "content": question})
        new_history.append({"role": "assistant", "content": answer})
        
        # Optional: Implement history truncation here
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

    async def close(self):
        """Clean up resources"""
        if self._llm:
            try:
                self._llm.clear_cache()
            except Exception as e:
                logger.error(f"Error closing LLM: {str(e)}")
        
        if self._vector_db and hasattr(self._vector_db, 'close'):
            try:
                await self._vector_db.close()
            except Exception as e:
                logger.error(f"Error closing vector DB: {str(e)}")