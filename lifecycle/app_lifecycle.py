import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from core.database import db_manager
from config.settings import pg_pool

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    try:
        await db_manager.initialize()
        logger.info("Milvus initialized successfully")
    except Exception as e:
        logger.error(f"Milvus startup failed: {e}")

    conn = None
    try:
        conn = pg_pool.getconn()
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SELECT 1;")
        result = cur.fetchone()
        cur.close()
        if result and result[0] == 1:
            logger.info("PostgreSQL connection successfully")
        else:
            logger.error("PostgreSQL connection failed at startup")
    except Exception as e:
        logger.error(f"PostgreSQL connection failed at startup: {e}")
    finally:
        if conn:
            pg_pool.putconn(conn)

    yield  # app runs here

    # --- Shutdown ---
    try:
        logger.info("Application shutdown complete")
        pg_pool.closeall()
    except Exception as e:
        logger.error(f"Shutdown error: {e}")
