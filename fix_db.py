
import sqlite3
import os

DB_PATH = "analysis_cache.db"

def fix_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("Dropping incorrect table...")
    cursor.execute("DROP TABLE IF EXISTS file_explanations")
    
    print("Recreating table with correct schema...")
    cursor.execute("""
        CREATE TABLE file_explanations (
            repo_url TEXT,
            path TEXT,
            explanation TEXT,
            PRIMARY KEY (repo_url, path)
        )
    """)
    
    conn.commit()
    conn.close()
    print("Database fixed successfully.")

if __name__ == "__main__":
    fix_db()
