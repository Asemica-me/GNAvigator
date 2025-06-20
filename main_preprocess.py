# main_preprocess.py
import logging
import subprocess
import os
import json
from datetime import datetime

# Configuration
SITEMAP_PATH = "sitemap/sitemap.xml"
CHUNKS_PATH = "chunks/chunks.json"
VECTOR_STORE_PATH = "vector_store"
LOG_FILE = f"logs/preprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def setup_logging():
    """Configure logging system"""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

def run_sitemap_generator():
    """Run sitemap generation script"""
    logging.info("Starting sitemap generation...")
    result = subprocess.run(
        ["python", "generate_sitemap.py"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        logging.error(f"Sitemap generation failed: {result.stderr}")
        raise RuntimeError("Sitemap generation failed")
    
    logging.info(f"Sitemap generated successfully at {SITEMAP_PATH}")
    logging.info(f"Output: {result.stdout}")

def run_chunking():
    """Run chunk creation script"""
    logging.info("Starting chunk creation...")
    result = subprocess.run(
        ["python", "create_chunks_json.py", SITEMAP_PATH, CHUNKS_PATH],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        logging.error(f"Chunk creation failed: {result.stderr}")
        raise RuntimeError("Chunk creation failed")
    
    logging.info(f"Chunks created successfully at {CHUNKS_PATH}")
    logging.info(f"Output: {result.stdout}")

def run_vector_store_creator():
    """Run vector store creation script"""
    logging.info("Starting vector store creation...")
    result = subprocess.run(
        ["python", "vector_store.py", CHUNKS_PATH, VECTOR_STORE_PATH],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        logging.error(f"Vector store creation failed: {result.stderr}")
        raise RuntimeError("Vector store creation failed")
    
    logging.info(f"Vector store created successfully at {VECTOR_STORE_PATH}")
    logging.info(f"Output: {result.stdout}")

def validate_outputs():
    """Validate pipeline outputs"""
    logging.info("Validating outputs...")
    
    # Validate sitemap exists
    if not os.path.exists(SITEMAP_PATH):
        raise FileNotFoundError(f"Sitemap not found at {SITEMAP_PATH}")
    
    # Validate chunks file exists and is valid JSON
    with open(CHUNKS_PATH, 'r') as f:
        try:
            chunks = json.load(f)
            if not isinstance(chunks, list) or len(chunks) == 0:
                raise ValueError("Invalid chunks format")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in chunks file")
    
    # Validate vector store directory
    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError(f"Vector store not found at {VECTOR_STORE_PATH}")
    if not os.listdir(VECTOR_STORE_PATH):
        raise ValueError("Vector store directory is empty")
    
    logging.info("All outputs validated successfully")

def main():
    """Main preprocess pipeline execution"""
    try:
        setup_logging()
        logging.info("Starting preprocessing pipeline")
        
        run_sitemap_generator()
        run_chunking()
        run_vector_store_creator()
        validate_outputs()
        
        logging.info("Preprocessing pipeline completed successfully")
    except Exception as e:
        logging.exception("Pipeline failed with error")
        raise

if __name__ == "__main__":
    main()