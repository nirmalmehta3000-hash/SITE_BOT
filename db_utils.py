import os
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Railway MySQL configuration
def get_db_config():
    """Get database configuration from environment variables"""
    return {
        'host': os.environ.get('MYSQLHOST'),
        'port': int(os.environ.get('MYSQLPORT', 3306)),
        'database': os.environ.get('MYSQLDATABASE'),
        'user': os.environ.get('MYSQLUSER'),
        'password': os.environ.get('MYSQLPASSWORD'),
        'charset': 'utf8mb4',
        'use_unicode': True,
        'autocommit': True
    }

def get_db_connection():
    """Create and return database connection"""
    try:
        config = get_db_config()
        
        # Check if all required config is present
        required_keys = ['host', 'database', 'user', 'password']
        missing_keys = [key for key in required_keys if not config.get(key)]
        
        if missing_keys:
            logger.error(f"Missing database configuration: {missing_keys}")
            return None
        
        connection = mysql.connector.connect(**config)
        
        if connection.is_connected():
            return connection
        else:
            logger.error("Failed to establish database connection")
            return None
            
    except Error as e:
        logger.error(f"Database connection error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error connecting to database: {e}")
        return None

def create_chat_table():
    """Create the chat history table"""
    connection = get_db_connection()
    if not connection:
        return False
    
    try:
        cursor = connection.cursor()
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS chat_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_name VARCHAR(255) NOT NULL,
            user_email VARCHAR(255) NOT NULL,
            user_mobile VARCHAR(20) NOT NULL,
            user_question TEXT NOT NULL,
            assistant_answer LONGTEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_email (user_email),
            INDEX idx_created_at (created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
        
        cursor.execute(create_table_query)
        logger.info("Chat history table created successfully")
        return True
        
    except Error as e:
        logger.error(f"Error creating table: {e}")
        return False
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def test_database_connection():
    """Test database connection"""
    connection = get_db_connection()
    if not connection:
        logger.error("Database connection test failed")
        return False
    
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        logger.info("Database connection test successful")
        return result is not None
        
    except Error as e:
        logger.error(f"Database test error: {e}")
        return False
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def initialize_database():
    """Initialize database with required tables"""
    logger.info("Initializing database...")
    
    # Test connection first
    if not test_database_connection():
        logger.error("Database initialization failed - no connection")
        return False
    
    # Create tables
    if create_chat_table():
        logger.info("Database initialized successfully")
        return True
    else:
        logger.error("Database initialization failed - table creation error")
        return False

def save_chat_to_database(user_name, user_email, user_mobile, question, answer):
    """Save chat entry to database"""
    connection = get_db_connection()
    if not connection:
        logger.error("Cannot save chat - no database connection")
        return False
    
    try:
        cursor = connection.cursor()
        
        insert_query = """
        INSERT INTO chat_history (user_name, user_email, user_mobile, user_question, assistant_answer)
        VALUES (%s, %s, %s, %s, %s)
        """
        
        cursor.execute(insert_query, (user_name, user_email, user_mobile, question, answer))
        
        if cursor.rowcount == 1:
            logger.info(f"Chat saved successfully for {user_email}")
            return True
        else:
            logger.warning(f"Unexpected rowcount: {cursor.rowcount}")
            return False
            
    except Error as e:
        logger.error(f"Error saving chat: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving chat: {e}")
        return False
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def get_user_chat_history(user_email, limit=10):
    """Get chat history for a user"""
    connection = get_db_connection()
    if not connection:
        return []
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        query = """
        SELECT user_question, assistant_answer, created_at
        FROM chat_history
        WHERE user_email = %s
        ORDER BY created_at DESC
        LIMIT %s
        """
        
        cursor.execute(query, (user_email, limit))
        results = cursor.fetchall()
        
        return results
        
    except Error as e:
        logger.error(f"Error fetching chat history: {e}")
        return []
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def get_total_chats():
    """Get total number of chats in database"""
    connection = get_db_connection()
    if not connection:
        return 0
    
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM chat_history")
        result = cursor.fetchone()
        return result[0] if result else 0
        
    except Error as e:
        logger.error(f"Error getting total chats: {e}")
        return 0
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def get_unique_users():
    """Get total number of unique users"""
    connection = get_db_connection()
    if not connection:
        return 0
    
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(DISTINCT user_email) FROM chat_history")
        result = cursor.fetchone()
        return result[0] if result else 0
        
    except Error as e:
        logger.error(f"Error getting unique users: {e}")
        return 0
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

# Debug function for Railway deployment
def debug_environment():
    """Debug environment variables for Railway deployment"""
    env_vars = ['MYSQLHOST', 'MYSQLPORT', 'MYSQLDATABASE', 'MYSQLUSER', 'MYSQLPASSWORD']
    
    print("=== Railway Database Environment Debug ===")
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            if var == 'MYSQLPASSWORD':
                print(f"{var}: {'*' * len(value)}")
            else:
                print(f"{var}: {value}")
        else:
            print(f"{var}: NOT SET")
    
    print("=== Connection Test ===")
    if test_database_connection():
        print("✅ Database connection successful")
    else:
        print("❌ Database connection failed")

if __name__ == "__main__":
    # Run debug when called directly
    debug_environment()
