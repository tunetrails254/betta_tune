import sqlite3
import logging
import os
from datetime import datetime
# Logging Configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def init_db():
    try:
        conn = sqlite3.connect('predictions.db')
        cursor = conn.cursor()

        # Create predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_file TEXT,
                predicted_gender TEXT,
                predicted_age_group TEXT,
                confidence_score REAL,
                gender_confidence REAL,
                age_confidence REAL,
                is_correct INTEGER,
                corrected_gender TEXT,
                corrected_age_group TEXT,
                user_feedback TEXT,
                features TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # ✅ Create usage tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                ip TEXT,
                date TEXT,
                request_count INTEGER,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
         # ✅ Table for users (API keys)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
            password_hash TEXT,
            api_key TEXT UNIQUE,
            plan TEXT DEFAULT 'free',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)


        conn.commit()
        conn.close()
        # Log success
        logger.info("✅ Database initialized with all tables.")
    except Exception as e:
        logger.error(f"❌ Database init failed: {str(e)}")
        raise
