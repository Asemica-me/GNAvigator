import sqlite3
import logging
from pathlib import Path
import pandas as pd
import os
import subprocess
import time
from dotenv import load_dotenv
from urllib.parse import quote_plus  # For safe token URL encoding
import datetime 

load_dotenv()

FEEDBACK_DB = Path(__file__).parent / "feedback" / "feedbacks.db"

def init_db():
    """Initialize the SQLite database for feedbacks"""
    conn = None
    try:
        FEEDBACK_DB.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(FEEDBACK_DB)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS feedbacks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    message_index INTEGER NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    rating INTEGER NOT NULL
                    )''')
        conn.commit()
        logging.info("Feedback database initialized")
    except Exception as e:
        logging.error(f"Error initializing feedback database: {e}")
    finally:
        if conn:
            conn.close()

def git_sync():
    """Sync feedback database to GitHub using token authentication"""
    token = os.getenv("GITHUB_TOKEN")
    repo_url = os.getenv("GITHUB_REPO_URL")
    
    if not token or not repo_url:
        logging.error("GitHub credentials not found in environment variables")
        return
        
    try:
        repo_dir = Path(__file__).parent
        
        # 1. Configure ephemeral Git identity (local to repo only)
        subprocess.run(["git", "config", "--local", "user.email", "automated@streamlit.app"], 
                      cwd=repo_dir, check=True)
        subprocess.run(["git", "config", "--local", "user.name", "Streamlit Automated Process"], 
                      cwd=repo_dir, check=True)
        
        # 2. Initialize Git repo if needed
        if not (repo_dir / ".git").exists():
            subprocess.run(["git", "init"], cwd=repo_dir, check=True)
        
        # 3. Stage changes
        subprocess.run(["git", "add", str(FEEDBACK_DB)], cwd=repo_dir, check=True)
        
        # 4. Commit changes
        commit_message = f"Feedback update {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        subprocess.run(["git", "commit", "-m", commit_message], cwd=repo_dir, check=True)
        
        # 5. Push with token authentication (URL-encoded)
        encoded_token = quote_plus(token)  # Handle special characters
        auth_repo_url = repo_url.replace("https://", f"https://{encoded_token}@")
        
        # Use timeout to prevent hanging in ephemeral st environment
        subprocess.run(
            ["git", "push", auth_repo_url, "main", "--force"],
            cwd=repo_dir,
            check=True,
            timeout=30  # Fail fast if network issues
        )
        logging.info("Feedback database synced to GitHub")
        
    except subprocess.TimeoutExpired:
        logging.error("Git push timed out (30s)")
    except subprocess.CalledProcessError as e:
        logging.error(f"Git operation failed: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error during Git sync: {str(e)}")

def save_feedback(feedback_data: dict):
    """Save feedback data to the database with memory optimization"""
    conn = None
    try:
        conn = sqlite3.connect(FEEDBACK_DB)
        c = conn.cursor()
        c.execute('''INSERT INTO feedbacks 
                    (timestamp, message_index, question, answer, rating) 
                    VALUES (?, ?, ?, ?, ?)''',
                 (feedback_data["timestamp"], 
                  feedback_data["message_index"],
                  feedback_data["question"],
                  feedback_data["answer"],
                  feedback_data["rating"]))
        conn.commit()
        logging.info(f"Feedback saved: {feedback_data}")
    except Exception as e:
        logging.error(f"Failed to save feedback: {e}")
    finally:
        if conn:
            conn.close()
        
        # Explicit cleanup to prevent memory leaks
        del feedback_data
        import gc; gc.collect()
    
    # Sync to GitHub after saving
    git_sync()

def export_feedbacks():
    """Export feedbacks from the database as a DataFrame with resource cleanup"""
    conn = None
    try:
        conn = sqlite3.connect(FEEDBACK_DB)
        df = pd.read_sql_query("SELECT * FROM feedbacks", conn)
        return df
    except Exception as e:
        logging.error(f"Error exporting feedbacks: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()
        # Clean up resources
        import gc; gc.collect()