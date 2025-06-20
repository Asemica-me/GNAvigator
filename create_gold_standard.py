import json
import os
import asyncio
import logging
import random
import time
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from llm_handler import MistralLLM 

load_dotenv()
LLM = os.getenv("GEN_MODEL")
DATA_FOLDER = "data"
INPUT_CHUNKS_FILE = os.path.join(DATA_FOLDER, "chunks_memory.json")
OUTPUT_TEST_SET = os.path.join(DATA_FOLDER, "test_set.json")
MAX_QUESTIONS_PER_CHUNK = 3
MAX_CONCURRENCY = 3  # Reduced from 5 to 3
CONTENT_PREVIEW_LENGTH = 150

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QuestionGenerator:
    def __init__(self, api_key: str):
        self.llm = MistralLLM(
            api_key=api_key,
            model_name=LLM,
            temperature=0.3,
            max_tokens=500,
            max_concurrency=MAX_CONCURRENCY,  # Use the reduced concurrency
            max_retries=5  # Increase retry attempts
        )
        self.last_request_time = time.time()
    
    async def generate_questions(self, chunk_content: str) -> list:
        if not chunk_content.strip():
            return []

        # Create a fake context
        context = [{
            "source": "question-generation",
            "contents": [chunk_content]
        }]
        
        # Create the prompt
        question_prompt = """
        Genera 1-3 domande in italiano basate esclusivamente sul testo fornito.
        Le domande devono:
        1. Essere rispondibili direttamente dal testo
        2. Essere chiare e concise
        3. Riguardare le informazioni chiave nel testo
        4. Essere in italiano corretto
        5. Non contenere riferimenti a immagini o tabelle
        6. Essere formulate come interrogative complete

        Restituisci SOLO un JSON valido con la struttura:
        {"questions": ["domanda1", "domanda2"]}
        """

        try:
            # Rate limiting: add jittered delay
            elapsed = time.time() - self.last_request_time
            if elapsed < 1.0:  # Ensure at least 1 second between requests
                jitter = random.uniform(0.1, 0.5)
                await asyncio.sleep(1.0 - elapsed + jitter)
            
            # Use your existing generate_async method
            response = await self.llm.generate_async(
                question=question_prompt,
                context=context,
                temperature=0.3
            )
            
            # Update last request time
            self.last_request_time = time.time()
            
            # Parse the response
            if response.strip().startswith("{"):
                try:
                    result = json.loads(response)
                    return result.get("questions", [])
                except json.JSONDecodeError:
                    pass
            
            # Fallback: extract questions from text
            questions = []
            for line in response.split("\n"):
                if any(line.strip().startswith(c) for c in ['-', '*', '•', '1.', '2.', '3.']):
                    question = line.strip().lstrip('*-•1234567890. ').strip()
                    if question and question.endswith('?'):
                        questions.append(question)
            return questions[:MAX_QUESTIONS_PER_CHUNK]
            
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return []
    
    async def close(self):
        """Clean up resources"""
        self.llm.clear_cache()

async def main():
    # Ensure data folder exists
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    # Load chunks from memory file
    try:
        with open(INPUT_CHUNKS_FILE, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
            
        # Handle both list and dictionary formats
        if isinstance(chunks_data, dict):
            chunks_list = list(chunks_data.values())
            logger.info(f"Loaded {len(chunks_list)} chunks from dictionary in {INPUT_CHUNKS_FILE}")
        elif isinstance(chunks_data, list):
            chunks_list = chunks_data
            logger.info(f"Loaded {len(chunks_list)} chunks from list in {INPUT_CHUNKS_FILE}")
        else:
            logger.error(f"Unexpected data format in {INPUT_CHUNKS_FILE}")
            return
            
    except FileNotFoundError:
        logger.error(f"Chunks file not found: {INPUT_CHUNKS_FILE}")
        return
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in {INPUT_CHUNKS_FILE}")
        return
    
    # Initialize Mistral question generator
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        logger.error("MISTRAL_API_KEY environment variable not set")
        return
    
    generator = QuestionGenerator(api_key)
    
    # Generate test set
    test_set = []
    valid_chunks = [c for c in chunks_list if c.get('content', '').strip()]
    logger.info(f"Found {len(valid_chunks)} valid chunks with content")
    
    # Process chunks with rate limiting
    progress_bar = tqdm(total=len(valid_chunks), desc="Generating questions")
    
    for chunk in valid_chunks:
        try:
            questions = await generator.generate_questions(chunk['content'])
            
            for q in questions[:MAX_QUESTIONS_PER_CHUNK]:
                test_entry = {
                    "question": q,
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk["source"],
                    "content_preview": chunk["content"][:CONTENT_PREVIEW_LENGTH] + "..." 
                            if len(chunk["content"]) > CONTENT_PREVIEW_LENGTH 
                            else chunk["content"],
                    "content_type": chunk["content_type"]
                }
                
                test_set.append(test_entry)
        except Exception as e:
            logger.error(f"Error processing chunk {chunk['chunk_id']}: {str(e)}")
        
        progress_bar.update(1)
        
        # Add additional delay between chunks
        await asyncio.sleep(random.uniform(0.5, 1.5))
    
    progress_bar.close()
    
    # Save results
    with open(OUTPUT_TEST_SET, "w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ Generated {len(test_set)} questions in {OUTPUT_TEST_SET}")
    await generator.close()

if __name__ == "__main__":
    asyncio.run(main())