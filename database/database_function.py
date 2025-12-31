from config.settings import pg_pool
from utils.logger import logger

def get_db_connection():
    """
    Get a PostgreSQL connection from the connection pool.
    Use with a context manager or manually close after use.
    """
    try:
        conn = pg_pool.getconn()
        return conn
    except Exception as e:
        logger.error(f"Failed to get PostgreSQL connection: {e}")
        return None
