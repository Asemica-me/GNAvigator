# main.py
import asyncio
from dotenv import load_dotenv
import os
import logging
from llm_handler import RAGOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("httpx").propagate = False
logging.getLogger("httpcore").propagate = False
logging.getLogger("chromadb").propagate = False

# Load environment variables
load_dotenv()

# Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
SITEMAP_PATH = os.getenv("SITEMAP_PATH")
BASE_DOMAIN = os.getenv("BASE_DOMAIN")

async def main():
    try:
        # Initialize RAG system
        rag = RAGOrchestrator(mistral_api_key=MISTRAL_API_KEY)
        
        # Initialize vector store (only needed once)
        await rag.initialize_vector_store(SITEMAP_PATH, BASE_DOMAIN)
        
        # Example query
        results = await rag.query(
            "Come posso cercare i dati nel Geoportale?"
        )
        
        # Display results
        print("\nRisposta:", results["answer"])
        print("\nFonti utilizzate:")
        for source in results["sources"]:
            print(f"- {source}")

    except Exception as e:
        logging.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())