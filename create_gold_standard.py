import asyncio
import json
import csv
import os
import logging
import re
from pathlib import Path
from dotenv import load_dotenv
from mistralai import Mistral, SystemMessage, UserMessage
import backoff
import random
from typing import List, Dict, Any, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class QuestionGenerator:
    def __init__(
        self,
        api_key: str,
        model_name: str = os.getenv("GEN_MODEL"),
        max_retries: int = 6,
        max_concurrency: int = 5,
        temperature: float = 0.1
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.max_retries = max_retries
        self.client = Mistral(api_key=api_key)
        self.question_id = 0

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def generate_questions(self, text: str, context_type: str = "single") -> List[str]:
        """Generate questions based on text content and context type"""
        if context_type == "single":
            # prompt = (
            #     """
            #     Genera 1-3 domande in italiano che possono essere risposte esattamente utilizzando SOLO il seguente testo.
            #     Scrivi ogni domanda su una nuova riga senza numeri, punti elenco o caratteri aggiuntivi.\n\n"""
            #     f"TESTO:\n{text}"
            # )
            prompt = """
                Generate 1-3 different questions in ITALIAN that could be answered by the following text:\n
                {text}\n     
                Write each question on a new line without numbers, bullet points, or additional characters."""
        else:  # multi-chunk context
            # prompt = (
            #     """
            #     Genera 1-2 domande in italiano che richiedono la COMBINAZIONE di tutte le informazioni nel testo seguente.
            #     Scrivi ogni domanda su una nuova riga senza numeri, punti elenco o caratteri aggiuntivi.\n\n"""
            #     f"TESTO COMBINATO:\n{text}"
            # )
            prompt = """
                Generate 1-2 questions in Italian that require the COMBINATION of the information in the following text:\n
                {text}\n
                Write each question on a new line without numbers, bullet points, or additional characters."""

        messages = [UserMessage(role="user", content=prompt)]
        async with self.semaphore:
            try:
                response = self.client.chat.complete(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=1000
                )
                content = response.choices[0].message.content.strip()
                return [line.strip() for line in content.splitlines() if line.strip()]
            except Exception as e:
                logger.error(f"Question generation failed: {str(e)}")
                return []


def is_valid_question(question: str) -> bool:
    """Filter out low-quality questions"""
    if not question or len(question) < 15 or len(question) > 150:
        return False
    if not question.endswith('?'):
        return False
    if any(word in question.lower() for word in ["riassumi", "opinione", "pensi", "secondo te"]):
        return False
    return True


async def main():
    # Configure paths
    data_folder = Path("data")
    chunks_file = data_folder / "chunks_memory.json"
    output_file = data_folder / "gold_standard.csv"
    
    # Validate files
    if not chunks_file.exists():
        logger.error(f"Chunks file not found: {chunks_file}")
        return

    # Load chunks data
    try:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        logger.info(f"Loaded {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"Error loading chunks: {str(e)}")
        return

    # Initialize generator
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        logger.error("MISTRAL_API_KEY environment variable not set")
        return
    
    generator = QuestionGenerator(api_key=api_key)

    # Prepare output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    written_pairs = set()

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["question", "chunk_id"])
        
        # Process single-chunk questions
        single_count = 0
        for chunk in chunks:
            chunk_id = chunk.get('chunk_id')
            content = chunk.get('content', '')
            
            if not chunk_id or not content.strip():
                continue
            
            try:
                questions = await generator.generate_questions(content, "single")
                for q in questions:
                    if not is_valid_question(q):
                        continue
                    # Deduplicate question-chunk pairs
                    pair = (q.lower().strip(), chunk_id)
                    if pair not in written_pairs:
                        writer.writerow([q, chunk_id])
                        written_pairs.add(pair)
                        single_count += 1
            except Exception as e:
                logger.error(f"Failed chunk {chunk_id}: {str(e)}")
        
        logger.info(f"Generated {single_count} single-chunk questions")
        
        # Process multi-chunk questions
        multi_count = 0
        
        # Create groups of 2-3 random chunks from the same source
        chunks_by_source = {}
        for chunk in chunks:
            if source := chunk.get('source'):
                chunks_by_source.setdefault(source, []).append(chunk)
        
        for source, source_chunks in chunks_by_source.items():
            if len(source_chunks) < 2:
                continue
                
            # Create 2-3 groups per source
            for _ in range(min(3, len(source_chunks) // 2)):
                # Random group size (2-3 chunks)
                group_size = random.randint(2, min(3, len(source_chunks)))
                group = random.sample(source_chunks, group_size)
                
                combined_content = "\n\n".join(c['content'] for c in group)
                logger.info(f"Processing multi-chunk group with {group_size} chunks from {source}")
                
                try:
                    questions = await generator.generate_questions(combined_content, "multi")
                    for q in questions:
                        if not is_valid_question(q):
                            continue
                        for chunk in group:
                            pair = (q.lower().strip(), chunk['chunk_id'])
                            if pair not in written_pairs:
                                writer.writerow([q, chunk['chunk_id']])
                                written_pairs.add(pair)
                                multi_count += 1
                except Exception as e:
                    logger.error(f"Failed multi-chunk group: {str(e)}")
        
        logger.info(f"Generated {multi_count} multi-chunk associations")
        logger.info(f"Total questions: {len(written_pairs)}")
        logger.info(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())