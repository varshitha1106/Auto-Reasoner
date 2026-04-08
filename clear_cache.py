
import sqlite3

DB_PATH = "analysis_cache.db"

def clear_cache():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("Clearing file_explanations table...")
    try:
        cursor.execute("DELETE FROM file_explanations")
        conn.commit()
        print("✅ Cache cleared successfully.")
    except Exception as e:
        print(f"❌ Failed to clear cache: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    clear_cache()
