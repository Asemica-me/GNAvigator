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
    except:
        pass
    
    # Fallback to environment variables
    token = os.getenv("GITHUB_TOKEN")
    repo_url = os.getenv("GITHUB_REPO_URL")
    return token, repo_url

def git_sync():
    """Sync feedback database to GitHub using token authentication"""
    token, repo_url = get_github_credentials()
    
    if not token or not repo_url:
        logging.error("GitHub credentials not found. Check your configuration.")
        return
        
    try:
        repo_dir = Path(__file__).parent
        auth_repo_url = repo_url.replace("https://", f"https://{token}@")
        
        # 1. Handle Git lock file if exists
        lock_file = repo_dir / ".git" / "index.lock"
        if lock_file.exists():
            lock_file.unlink(missing_ok=True)
            logging.warning("Removed stale Git index.lock file")
        
        # 2. Configure ephemeral Git identity
        subprocess.run(["git", "config", "--local", "user.email", "automated@streamlit.app"], 
                      cwd=repo_dir, check=True)
        subprocess.run(["git", "config", "--local", "user.name", "Streamlit Automated Process"], 
                      cwd=repo_dir, check=True)
        
        # 3. Initialize Git repo if needed
        if not (repo_dir / ".git").exists():
            subprocess.run(["git", "init"], cwd=repo_dir, check=True)
            subprocess.run(["git", "branch", "-M", "main"], cwd=repo_dir, check=True)
            subprocess.run(["git", "remote", "add", "origin", auth_repo_url], 
                          cwd=repo_dir, check=True)
        
        # 4. Stage feedback.db changes
        subprocess.run(["git", "add", str(FEEDBACK_DB)], cwd=repo_dir, check=True)
        
        # 5. Check for changes
        status_result = subprocess.run(
            ["git", "status", "--porcelain", "--", str(FEEDBACK_DB)],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True
        )
        if not status_result.stdout.strip():
            logging.info("No changes to commit for feedback database")
            return
        
        # 6. Commit changes
        commit_message = f"Automated feedback update {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        subprocess.run(["git", "commit", "-m", commit_message], cwd=repo_dir, check=True)
        
        # 7. Push with conflict resolution
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Try direct push first
                subprocess.run(
                    ["git", "push", auth_repo_url, "main"],
                    cwd=repo_dir,
                    check=True,
                    timeout=30
                )
                logging.info("Feedback database pushed successfully")
                break
            except subprocess.CalledProcessError as push_error:
                err_output = push_error.stderr.decode() if push_error.stderr else str(push_error)
                if "non-fast-forward" in err_output:
                    # Handle merge conflict by rebasing
                    logging.warning("Remote has changes, rebasing...")
                    try:
                        # Reset the commit we just made
                        subprocess.run(["git", "reset", "--soft", "HEAD~1"], cwd=repo_dir, check=True)
                        
                        # Pull latest changes with rebase
                        subprocess.run(
                            ["git", "pull", "--rebase", auth_repo_url, "main"],
                            cwd=repo_dir,
                            check=True,
                            timeout=30
                        )
                        
                        # Re-commit and push
                        subprocess.run(["git", "add", str(FEEDBACK_DB)], cwd=repo_dir, check=True)
                        subprocess.run(["git", "commit", "-m", commit_message], cwd=repo_dir, check=True)
                    except Exception as rebase_error:
                        logging.error(f"Rebase failed: {rebase_error}")
                        if attempt < max_retries - 1:
                            wait_time = 5
                            logging.info(f"Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            logging.error("Failed to sync after rebase attempts")
                            raise
                else:
                    logging.error(f"Git push failed: {err_output}")
                    if attempt < max_retries - 1:
                        wait_time = 10
                        logging.info(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logging.error("Git push failed after multiple attempts")
                        raise
    except Exception as e:
        logging.error(f"Git sync failed: {str(e)}")
    finally:
        # Clean credentials from memory
        if 'auth_repo_url' in locals():
            del auth_repo_url
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
        logging.info(f"Feedback saved.")
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