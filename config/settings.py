from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from psycopg_pool import ConnectionPool
from pymilvus import connections, Collection
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()

class Settings(BaseSettings):
    model_config = ConfigDict(extra='allow', env_file='.env')
    """Application settings loaded from environment variables."""

    
    # PostgreSQL Configuration
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "winery-bot"
    postgres_user: str = "postgres"
    postgres_password: str = "dev_jhon"

    # database_url: str = "postgresql://postgres:dev_jhon@localhost:5432/winery-bot"

    pg_min_size: int = 1
    pg_max_size: int = 10
    
    # Milvus Configuration (Primary Database)
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    
    # AI Services
    openai_api_key: str = ""
    
    # Query Settings
    confidence_threshold: float = 0.7
    max_results: int = 10
    query_timeout: int = 30
    
    # Application
    secret_key: str = "dev-secret-key"
    debug: bool = True
    log_level: str = "INFO"
    
    # Cache Settings (in-memory)
    cache_ttl: int = 3600
    max_cache_size: int = 1000
    
    # Performance Settings
    max_concurrent_requests: int = 100
    request_timeout: int = 30
    
    # Vector Search Settings
    vector_search_threshold: float = 0.7
    max_vector_results: int = 5
    
    # Content Processing
    max_content_length: int = 65535
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    
    # Session Management
    session_timeout: int = 3600

settings = Settings()

# PostgreSQL Connection Pool
pg_pool = ConnectionPool(
    conninfo=f"postgresql://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}", 
    min_size=settings.pg_min_size,
    max_size=settings.pg_max_size
)

# Milvus Connection
def init_milvus():
    connections.connect(alias="default", host=settings.milvus_host, port=settings.milvus_port)

def get_milvus_collection(collection_name="knowledge_base"):
    return Collection(collection_name)
