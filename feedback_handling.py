import sqlite3
import logging
from pathlib import Path
import pandas as pd
import os
import subprocess
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the feedback database path
FEEDBACK_DB = Path(__file__).parent / "feedback" / "feedbacks.db"

def init_db():
    """Initialize the SQLite database for feedbacks."""
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
    """Sync feedback database to GitHub using token from .env"""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        logging.error("GitHub token not found in environment variables")
        return
        
    try:
        repo_dir = Path(__file__).parent
        repo_url = os.getenv("GITHUB_REPO_URL", "https://github.com/your-username/your-repo.git")
        
        # Configure Git to store credentials temporarily
        credentials_file = repo_dir / ".git-credentials"
        with open(credentials_file, "w") as f:
            f.write(f"https://{token}@github.com")
        
        # Set Git configuration for this operation
        env = os.environ.copy()
        env["GIT_CONFIG_PARAMETERS"] = f"'credential.helper=store --file {credentials_file}'"
        env["GIT_TERMINAL_PROMPT"] = "0"
        
        # Initialize Git repo if needed
        if not (repo_dir / ".git").exists():
            subprocess.run(["git", "init"], cwd=repo_dir, check=True, env=env)
            subprocess.run(["git", "remote", "add", "origin", repo_url], 
                          cwd=repo_dir, check=True, env=env)
        
        # Add and commit changes
        subprocess.run(["git", "add", str(FEEDBACK_DB)], cwd=repo_dir, check=True, env=env)
        commit_message = f"Feedback update {time.strftime('%Y%m%d-%H%M%S')}"
        subprocess.run(["git", "commit", "-m", commit_message], 
                      cwd=repo_dir, check=True, env=env)
        
        # Push changes
        subprocess.run(["git", "push", "origin", "main", "--force"], 
                      cwd=repo_dir, check=True, env=env)
        logging.info("Feedback database synced to GitHub")
        
        # Clean up credentials file
        credentials_file.unlink()
    except Exception as e:
        logging.error(f"Git sync failed: {e}")

def save_feedback(feedback_data: dict):
    """Save feedback data to the database."""
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

def export_feedbacks():
    """Export feedbacks from the database as a DataFrame."""
    try:
        conn = sqlite3.connect(FEEDBACK_DB)
        df = pd.read_sql_query("SELECT * FROM feedbacks", conn)
        return df
    except Exception as e:
        logging.error(f"Error exporting feedbacks: {e}")
        return pd.DataFrame()  # Return empty DF on error
    finally:
        if conn:
            conn.close()