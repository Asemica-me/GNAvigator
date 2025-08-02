import asyncio
import json
import os

from dotenv import load_dotenv

from rag_sys import RAGOrchestrator

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GEN_MODEL = os.getenv("GEN_MODEL")


async def generate_test_data(output_file=None, num_questions=50):
    """Generate test questions from document chunks"""
    if output_file is None:
        output_file = os.path.join("data", "test_dataset.json")
    orchestrator = RAGOrchestrator(mistral_api_key=MISTRAL_API_KEY)

    # Get sample documents
    sample_docs = await orchestrator.sample_documents(num_questions)

    # Filter out image chunks
    filtered_docs = [
        doc
        for doc in sample_docs
        if doc["metadata"].get("content_type", "") not in ["image_ocr", "image"]
    ]  # take only the first N after filtering

    test_data = []
    for doc in filtered_docs:
        question = await generate_question_for_doc(orchestrator, doc)
        test_data.append(
            {
                "question": question,
                "relevant_docs": [doc["id"]],  # Ground truth
                "document_content": doc["content"],  # For reference
            }
        )
        await asyncio.sleep(1)  # Avoid hitting rate limits

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(test_data)} test questions in {output_file}")


async def generate_question_for_doc(orchestrator, document):
    """Use LLM to generate question from document content"""
    prompt = f"""
    [INST] 
    Genera una domanda a cui il seguente testo risponde. 
    La domanda deve essere in italiano e deve riguardare esclusivamente le informazioni presenti nel testo.
    
    Testo:
    {document["content"][:2000]}
    
    Ritorna SOLO la domanda, senza alcun commento aggiuntivo.
    [/INST]
    """

    response = await orchestrator.llm.generate_async(question=prompt, context=[])
    return response.strip()


if __name__ == "__main__":
    asyncio.run(
        generate_test_data(output_file="data/test_dataset.json", num_questions=550)
    )
# This script generates a test dataset of questions based on document chunks.