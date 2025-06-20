import sqlite3
import logging
from pathlib import Path
import pandas as pd
import os
import subprocess
import datetime
import sys
import time
import gc

FEEDBACK_DB = Path(__file__).parent / "feedback" / "feedbacks.db"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

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
    except Exception as e:
        logging.error(f"Error initializing feedback database: {e}")
    finally:
        if conn:
            conn.close()

def get_github_credentials():
    """Robust GitHub credential retrieval for all environments"""
    token = None
    repo_url = None
    
    # Try Streamlit secrets first
    try:
        import streamlit as st
        token = st.secrets.get("GITHUB_TOKEN", os.getenv("GITHUB_TOKEN"))
        repo_url = st.secrets.get("GITHUB_REPO_URL", os.getenv("GITHUB_REPO_URL"))
        if token and repo_url:
            return token, repo_url
    except (ImportError, AttributeError, KeyError):
        pass
    
    # Try environment variables
    token = os.getenv("GITHUB_TOKEN")
    repo_url = os.getenv("GITHUB_REPO_URL")
    if token and repo_url:
        return token, repo_url
    
    # Try dotenv file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        token = os.getenv("GITHUB_TOKEN")
        repo_url = os.getenv("GITHUB_REPO_URL")
        if token and repo_url:
            return token, repo_url
    except ImportError:
        pass
    
    # Final fallback
    return None, None

def git_sync():
    """Sync feedback database to GitHub using token authentication"""
    token, repo_url = get_github_credentials()
    
    if not token or not repo_url:
        logging.error("GitHub credentials not found. Check your configuration.")
        return
        
    try:
        repo_dir = Path(__file__).parent
        auth_repo_url = repo_url.replace("https://", f"https://{token}@")
        
        # 1. Handle Git lock file if it exists
        lock_file = repo_dir / ".git" / "index.lock"
        if lock_file.exists():
            try:
                lock_file.unlink()
                logging.warning("Removed stale Git index.lock file")
            except Exception as e:
                logging.error(f"Failed to remove lock file: {e}")
                return
        
        # 2. Configure ephemeral Git identity
        subprocess.run(["git", "config", "--local", "user.email", "automated@streamlit.app"], 
                      cwd=repo_dir, check=True)
        subprocess.run(["git", "config", "--local", "user.name", "Streamlit Automated Process"], 
                      cwd=repo_dir, check=True)
        
        # 3. Initialize Git repo if needed
        if not (repo_dir / ".git").exists():
            subprocess.run(["git", "init"], cwd=repo_dir, check=True)
            subprocess.run(["git", "branch", "-M", "main"], cwd=repo_dir, check=True)
            subprocess.run(["git", "remote", "add", "origin", repo_url], 
                          cwd=repo_dir, check=True)
        
        # 4. Pull latest changes first (to minimize conflicts)
        try:
            subprocess.run(
                ["git", "pull", auth_repo_url, "main", "--rebase"],
                cwd=repo_dir,
                check=True,
                timeout=30
            )
        except subprocess.CalledProcessError as e:
            logging.warning(f"Initial pull failed: {e}. Proceeding with local changes.")
        
        # 5. Stage changes
        subprocess.run(["git", "add", str(FEEDBACK_DB)], cwd=repo_dir, check=True)
        
        # 6. Check for changes
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True
        )
        if not status_result.stdout.strip():
            return
        
        # 7. Commit changes
        commit_message = f"Feedback update {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        subprocess.run(["git", "commit", "-m", commit_message], cwd=repo_dir, check=True)
        
        # 8. Push with retry logic
        max_retries = 2
        for attempt in range(max_retries):
            try:
                subprocess.run(
                    ["git", "push", auth_repo_url, "main"],
                    cwd=repo_dir,
                    check=True,
                    timeout=30
                )
                break
            except subprocess.TimeoutExpired:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    logging.warning(f"Git push timed out - retrying in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logging.error("Git push failed after multiple attempts")
                    raise
    except subprocess.CalledProcessError as e:
        logging.error(f"Git operation failed: {str(e)}")
        if e.stderr:
            logging.error(f"Command stderr: {e.stderr.decode().strip()}")
        if e.stdout:
            logging.error(f"Command stdout: {e.stdout.decode().strip()}")
    except Exception as e:
        logging.error(f"Unexpected error during Git sync: {str(e)}")
    finally:
        # Safe cleanup without UnboundLocalError
        if 'auth_repo_url' in locals():
            del auth_repo_url
        if 'token' in locals():
            token = None
        gc.collect()

def save_feedback(feedback_data: dict):
    """Save feedback data to the database"""
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
    
    # Sync to GitHub after saving
    git_sync()
    gc.collect()

def export_feedbacks():
    """Export feedbacks from the database as a DataFrame"""
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
        gc.collect()