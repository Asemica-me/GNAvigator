import asyncio
import json
import logging
import os
import re
import time
import gc
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from aiohttp import ClientSession
from vector_store import *

load_dotenv()
logger = logging.getLogger(__name__)

class MistralLLM:
    def __init__(
        self,
        api_key: str,
        model_name: str = "mistral-small",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        max_concurrency: int = 5
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.citation_regex = re.compile(r'\[(\d+)\]')
        self.client = None
        self.request_count = 0
        
    async def get_client(self):
        """Get or create a client session"""
        if self.client is None or self.client.closed:
            self.client = ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            logger.info("Created new Mistral client")
        return self.client

    def _build_rag_prompt(self, question: str, context: List[Dict[str, Any]], chat_history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, Any]]:
        """Construct RAG system prompt with context"""
        system_content = """
        ... [your existing system content] ...
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
        
        # Build message list
        messages = [{"role": "system", "content": system_content}]
        
        # Add chat history
        if chat_history:
            for msg in chat_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add current context and question
        messages.append({
            "role": "user",
            "content": f"CONTESTO:\n{context_str}\n\nDOMANDA: {question}"
        })
        
        return messages

    async def generate_async(
        self,
        question: str,
        context: List[Dict[str, Any]],
        temperature: float = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate answer with RAG context using async streaming"""
        client = await self.get_client()
        self.request_count += 1
        
        async with self.semaphore:
            try:
                messages = self._build_rag_prompt(question, context, chat_history)
                
                async with client.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    json={
                        "model": self.model_name,
                        "messages": messages,
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
                if self.client:
                    await self.client.close()
                    self.client = None
                raise
            finally:
                # Clean up intermediate resources
                if 'messages' in locals():
                    del messages
                gc.collect()

    def clear_cache(self):
        """Release client resources"""
        async def close_client():
            if self.client:
                try:
                    await self.client.close()
                except Exception as e:
                    logger.warning(f"Error closing client: {str(e)}")
                finally:
                    self.client = None
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(close_client())
            else:
                loop.run_until_complete(close_client())
        except Exception:
            pass
        
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
        
    async def query(self, question: str, top_k: int=5, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
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
                "answer": f"Si è verificato un errore durante l'elaborazione: {str(e)}",
                "sources": {},
                "context": []
            }
        finally:
            # Clean up intermediate resources
            if 'context_chunks' in locals():
                del context_chunks
            gc.collect()

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