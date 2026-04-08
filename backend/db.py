"""
Simple SQLite database for storing analysis results.
This allows us to cache results so we don't re-analyze the same repository.
"""

import sqlite3
import os
from typing import List, Dict, Optional

# Database file path
DB_PATH = "analysis_cache.db"


def init_db():
    """Create the database tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Users table for authentication
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL
        )
    """)

    # Table for repository-level summaries
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis (
            repo_url TEXT PRIMARY KEY,
            repo_summary TEXT
        )
    """)
    
    # Table for file-level summaries
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS file_summary (
            repo_url TEXT,
            path TEXT,
            summary TEXT,
            PRIMARY KEY (repo_url, path)
        )
    """)


    # Table for file-level detailed explanations
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS file_explanations (
            repo_url TEXT,
            path TEXT,
            explanation TEXT,
            PRIMARY KEY (repo_url, path)
        )
    """)
    
    conn.commit()
    conn.close()


def get_analysis(repo_url: str) -> Optional[Dict]:
    """
    Get stored analysis for a repository URL.
    Returns None if not found, otherwise returns:
    {
        "repo_summary": "...",
        "file_summaries": [{"path": "...", "summary": "..."}, ...]
    }
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get repo summary
    cursor.execute("SELECT repo_summary FROM analysis WHERE repo_url = ?", (repo_url,))
    repo_row = cursor.fetchone()
    
    if repo_row is None:
        conn.close()
        return None
    
    repo_summary = repo_row[0]
    
    # Get file summaries
    cursor.execute(
        "SELECT path, summary FROM file_summary WHERE repo_url = ?",
        (repo_url,)
    )
    file_rows = cursor.fetchall()
    
    file_summaries = [
        {"path": row[0], "summary": row[1]}
        for row in file_rows
    ]
    
    conn.close()
    
    return {
        "repo_summary": repo_summary,
        "file_summaries": file_summaries
    }


def save_analysis(repo_url: str, repo_summary: str, file_summaries: List[Dict]):
    """
    Save analysis results to the database.
    
    Args:
        repo_url: The GitHub repository URL
        repo_summary: The repository-level summary text
        file_summaries: List of dicts with "path" and "summary" keys
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Save repo summary (replace if exists)
    cursor.execute(
        "INSERT OR REPLACE INTO analysis (repo_url, repo_summary) VALUES (?, ?)",
        (repo_url, repo_summary)
    )
    
    # Delete old file summaries for this repo
    cursor.execute("DELETE FROM file_summary WHERE repo_url = ?", (repo_url,))
    
    # Insert new file summaries
    for file_summary in file_summaries:
        cursor.execute(
            "INSERT INTO file_summary (repo_url, path, summary) VALUES (?, ?, ?)",
            (repo_url, file_summary["path"], file_summary["summary"])
        )
    
    conn.commit()
    conn.close()


def get_file_explanation(repo_url: str, path: str) -> Optional[str]:
    """Get stored explanation for a specific file."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT explanation FROM file_explanations WHERE repo_url = ? AND path = ?",
        (repo_url, path)
    )
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return row[0]
    return None


def save_file_explanation(repo_url: str, path: str, explanation: str):
    """Save explanation for a specific file."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO file_explanations (repo_url, path, explanation) VALUES (?, ?, ?)",
        (repo_url, path, explanation)
    )
    conn.commit()
    conn.close()


def create_user(username: str, password_hash: str) -> bool:
    """Create a new user. Returns True if successful, False if username exists."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, password_hash)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_user(username: str) -> Optional[Dict]:
    """Get user by username."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT username, password_hash FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {"username": row[0], "password_hash": row[1]}
    return None



