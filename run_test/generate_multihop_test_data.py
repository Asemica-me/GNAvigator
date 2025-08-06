import asyncio
import json
import os
import sys
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv

from rag_sys import RAGOrchestrator

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEST_SUBDIR = os.path.join(DATA_DIR, "test")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

GROUP_SIZE_RANGE = (2, 4)
NUM_MULTI = 400

SINGLEHOP_FILE = os.path.join(TEST_SUBDIR, "test_dataset_singlehop.json")
MULTIHOP_FILE = os.path.join(TEST_SUBDIR, "test_dataset_multihop.json")

async def generate_multichunk_question(orchestrator, docs_subset):
    multi_chunks_content = "\n\n".join([doc["document_content"][:1000] for doc in docs_subset])
    prompt = f"""
    [INST]
    Leggi i seguenti testi. Genera una domanda in italiano che può essere risolta
    solo combinando informazioni provenienti da almeno {len(docs_subset)} di questi testi.
    La domanda deve essere specifica, e la risposta NON deve essere trovabile in un singolo testo.
    Testi:
    {multi_chunks_content}
    Ritorna SOLO la domanda, senza spiegazioni.
    [/INST]
    """
    response = await orchestrator.llm.generate_async(question=prompt, context=[])
    return response.strip()

async def generate_multihop_test_data(
    num_multi=NUM_MULTI,
    group_size_range=GROUP_SIZE_RANGE,
    input_file=SINGLEHOP_FILE,
    output_file=MULTIHOP_FILE
):
    orchestrator = RAGOrchestrator(mistral_api_key=MISTRAL_API_KEY)

    if os.path.exists(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            all_docs = json.load(f)
    else:
        raise FileNotFoundError(f"Input file {input_file} not found")

    multihop_data = []

    for _ in range(num_multi):
        group_size = random.randint(*group_size_range)
        docs_subset = random.sample(all_docs, group_size)
        question = await generate_multichunk_question(orchestrator, docs_subset)
        multihop_data.append({
            "question": question,
            "relevant_docs": [doc_id for doc in docs_subset for doc_id in doc["relevant_docs"]],
            "document_content": [doc["document_content"] for doc in docs_subset],
            "is_multihop": True,
            "num_docs": group_size  # optional: for analysis
        })
        await asyncio.sleep(1)  # Avoid rate limits

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(multihop_data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(multihop_data)} multi-hop questions to {output_file}")

if __name__ == "__main__":
    asyncio.run(generate_multihop_test_data())
