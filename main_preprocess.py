# main_preprocess.py
import logging
import subprocess
import os
import json
import time
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration using os.path
SITEMAP_DIR = os.path.join(SCRIPT_DIR, "sitemap")
SITEMAP_PATH = os.path.join(SITEMAP_DIR, "GNA__sitemap.xml") 
CHUNKS_DIR = os.path.join(SCRIPT_DIR, "data")
CHUNKS_PATH = os.path.join(CHUNKS_DIR, "chunks_memory.json")
FAISS_DIR = os.path.join(SCRIPT_DIR, ".faiss_db")
REQUIRED_FAISS_FILES = ["index.faiss", "metadata.pkl"]

def setup_logging():
    """Configure logging to console only"""
    logger = logging.getLogger('preprocessor')
    logger.setLevel(logging.INFO)
    
    # Console handler logging
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(console_handler)
    return logger

def get_venv_python():
    """Get the correct Python executable path for the virtual environment"""
    if os.name == 'nt':  # Windows
        return os.path.join(SCRIPT_DIR, ".venv", "Scripts", "python.exe")
    else:  # Linux/Mac
        return os.path.join(SCRIPT_DIR, ".venv", "bin", "python")

def run_sitemap_generator(logger):
    """Run sitemap generation script"""
    logger.info("Starting sitemap generation...")
    start_time = time.time()
    
    # Create sitemap directory if needed
    os.makedirs(SITEMAP_DIR, exist_ok=True)
    
    script_path = os.path.join(SCRIPT_DIR, "generate_sitemap.py")
    venv_python = get_venv_python()
    
    # Create environment with activated virtualenv
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(venv_python) + os.pathsep + env["PATH"]
    
    result = subprocess.run(
        [venv_python, script_path],
        capture_output=True,
        text=True,
        env=env
    )
    
    duration = time.time() - start_time
    
    if result.returncode != 0:
        logger.error(f"Sitemap generation failed in {duration:.2f}s")
        logger.error(f"Error details: {result.stderr}")
        raise RuntimeError("Sitemap generation failed")
    
    # Verify sitemap was created
    if not os.path.exists(SITEMAP_PATH):
        logger.error(f"Expected sitemap not found at {SITEMAP_PATH}")
        logger.info(f"Files in sitemap directory: {os.listdir(SITEMAP_DIR)}")
        raise FileNotFoundError("Sitemap file not created")
    
    logger.info(f"Sitemap generated in {duration:.2f}s")

def run_chunking(logger):
    """Run chunk creation script"""
    logger.info("Starting chunk creation...")
    start_time = time.time()
    
    # Create chunks directory if needed
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    
    script_path = os.path.join(SCRIPT_DIR, "create_chunks_json.py")
    venv_python = get_venv_python()
    
    # Create environment with activated virtualenv
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(venv_python) + os.pathsep + env["PATH"]
    
    result = subprocess.run(
        [venv_python, script_path],
        capture_output=True,
        text=True,
        env=env
    )
    
    duration = time.time() - start_time
    
    if result.returncode != 0:
        logger.error(f"Chunk creation failed in {duration:.2f}s")
        logger.error(f"Error details: {result.stderr}")
        raise RuntimeError("Chunk creation failed")
    
    # Check if chunks were created
    if not os.path.exists(CHUNKS_PATH):
        logger.error(f"Chunks file not found at {CHUNKS_PATH}")
        raise FileNotFoundError("Chunks file not created")
    
    logger.info(f"Created chunks in {duration:.2f}s")

def run_vector_store_creator(logger):
    """Run vector store creation script"""
    logger.info("Starting vector store creation...")
    start_time = time.time()
    
    # Create vector store directory if needed
    os.makedirs(FAISS_DIR, exist_ok=True)
    
    script_path = os.path.join(SCRIPT_DIR, "vector_store.py")
    venv_python = get_venv_python()
    
    # Create environment with activated virtualenv
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(venv_python) + os.pathsep + env["PATH"]
    
    result = subprocess.run(
        [venv_python, script_path],
        capture_output=True,
        text=True,
        env=env
    )
    
    duration = time.time() - start_time
    
    if result.returncode != 0:
        logger.error(f"Vector store creation failed in {duration:.2f}s")
        logger.error(f"Error details: {result.stderr}")
        raise RuntimeError("Vector store creation failed")
    
    logger.info(f"Vector store created in {duration:.2f}s")

def validate_outputs(logger):
    """Validate pipeline outputs"""
    logger.info("Validating outputs...")
    
    # Validate sitemap exists
    if not os.path.exists(SITEMAP_PATH):
        raise FileNotFoundError(f"Sitemap not found at {SITEMAP_PATH}")
    
    # Validate chunks file exists
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(f"Chunks file not found at {CHUNKS_PATH}")
    
    # Validate vector store directory exists
    if not os.path.exists(FAISS_DIR):
        raise FileNotFoundError(f"Vector store not found at {FAISS_DIR}")
    
    # Check for required vector store files
    missing_files = []
    for file in REQUIRED_FAISS_FILES:
        path = os.path.join(FAISS_DIR, file)
        if not os.path.exists(path):
            missing_files.append(file)
    
    if missing_files:
        raise FileNotFoundError(
            f"Vector store files missing: {', '.join(missing_files)}"
        )
    
    logger.info("All outputs validated successfully")

def main():
    """Main preprocess pipeline execution"""
    logger = setup_logging()
    
    try:
        logger.info("Starting preprocessing pipeline")
        pipeline_start = time.time()
        
        # Step 1: Generate sitemap
        run_sitemap_generator(logger)
        
        # Step 2: Create chunks
        run_chunking(logger)
        
        # Step 3: Create vector store
        run_vector_store_creator(logger)
        
        # Step 4: Validate outputs
        validate_outputs(logger)
        
        total_time = time.time() - pipeline_start
        logger.info(f"Preprocessing pipeline completed in {total_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()