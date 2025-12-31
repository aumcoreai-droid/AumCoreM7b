# memory_db.py - TiDB Integration
import pymysql
import os
from datetime import datetime

class TiDBMemory:
    def __init__(self):
        self.connection = None
        self.connect()
    
    def connect(self):
        try:
            self.connection = pymysql.connect(
                host=os.getenv("TIDB_HOST", "gateway01.ap-southeast-1.prod.aws.tidbcloud.com"),
                port=int(os.getenv("TIDB_PORT", 4000)),
                user=os.getenv("TIDB_USER", "2Rg6kfo2rNEB3PN.root"),
                password=os.getenv("TIDB_PASSWORD", "9JJabiRfo0WpH9FP"),
                database=os.getenv("TIDB_DATABASE", "test"),
                ssl={'ssl': {'ca': ''}}
            )
            self.create_tables()
            print("✅ TiDB connected successfully")
        except Exception as e:
            print(f"❌ TiDB connection failed: {e}")
            self.connection = None
    
    def create_tables(self):
        if not self.connection:
            return
            
        with self.connection.cursor() as cursor:
            # Create chat_history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_input TEXT,
                    ai_response MEDIUMTEXT,
                    language_mode VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Check if language_mode column exists, if not add it
            try:
                cursor.execute("SHOW COLUMNS FROM chat_history LIKE 'language_mode'")
                if not cursor.fetchone():
                    print("⚠️ Adding missing 'language_mode' column to chat_history")
                    cursor.execute("ALTER TABLE chat_history ADD COLUMN language_mode VARCHAR(20) DEFAULT 'en'")
            except Exception as e:
                print(f"⚠️ Column check failed: {e}")
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS code_snippets (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    code_type VARCHAR(100),
                    code_content LONGTEXT,
                    description TEXT,
                    usage_count INT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.connection.commit()
    
    def save_chat(self, user_input, ai_response, language_mode="en"):
        if not self.connection:
            return False
            
        try:
            with self.connection.cursor() as cursor:
                # Try with language_mode column
                try:
                    cursor.execute(
                        "INSERT INTO chat_history (user_input, ai_response, language_mode) VALUES (%s, %s, %s)",
                        (user_input, ai_response, language_mode)
                    )
                except pymysql.err.OperationalError as e:
                    # If language_mode column doesn't exist, try without it
                    if "Unknown column 'language_mode'" in str(e):
                        print("⚠️ language_mode column missing, inserting without it")
                        cursor.execute(
                            "INSERT INTO chat_history (user_input, ai_response) VALUES (%s, %s)",
                            (user_input, ai_response)
                        )
                    else:
                        raise e
                
                self.connection.commit()
                return True
        except Exception as e:
            print(f"❌ Save chat error: {e}")
            return False
    
    def get_recent_chats(self, limit=10):
        if not self.connection:
            return []
            
        try:
            with self.connection.cursor() as cursor:
                # Try with language_mode column
                try:
                    cursor.execute(
                        "SELECT user_input, ai_response, language_mode FROM chat_history ORDER BY created_at DESC LIMIT %s",
                        (limit,)
                    )
                    return cursor.fetchall()
                except pymysql.err.OperationalError as e:
                    # If language_mode column doesn't exist, select without it
                    if "Unknown column 'language_mode'" in str(e):
                        print("⚠️ language_mode column missing, selecting without it")
                        cursor.execute(
                            "SELECT user_input, ai_response FROM chat_history ORDER BY created_at DESC LIMIT %s",
                            (limit,)
                        )
                        # Add default language_mode for compatibility
                        rows = cursor.fetchall()
                        return [(row[0], row[1], "en") for row in rows]
                    else:
                        raise e
        except Exception as e:
            print(f"❌ Get chats error: {e}")
            return []
    
    def save_code_snippet(self, code_type, code_content, description):
        if not self.connection:
            return False
            
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO code_snippets (code_type, code_content, description) 
                       VALUES (%s, %s, %s)""",
                    (code_type, code_content, description)
                )
                self.connection.commit()
                return True
        except Exception as e:
            print(f"❌ Save code error: {e}")
            return False

# Global instance
tidb_memory = TiDBMemory()