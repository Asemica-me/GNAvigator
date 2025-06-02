# feedback_handling.py
import sqlite3
import logging
from pathlib import Path
import pandas as pd
import os

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