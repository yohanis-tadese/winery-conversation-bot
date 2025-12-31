import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from psycopg_pool import ConnectionPool
from config.settings import settings
import structlog
import logging

logger = structlog.get_logger()

from config.settings import pg_pool

class DatabaseManager:
    def __init__(self):
        self.settings = settings
        self.milvus_connected: bool = False
        self.postgres_initialized: bool = False
        self.cache: Dict[str, Dict[str, Any]] = {}

    async def initialize(self) -> None:
        # Safely initialize Milvus and PostgreSQL
        await self._init_milvus()
        await self._init_postgres()

    # -----------------------------
    # Milvus initialization
    # -----------------------------
    async def _init_milvus(self):
        if self.milvus_connected:
            logger.info("Milvus already connected, skipping initialization")
            return

        try:
            connections.connect(
                alias="default",
                host=self.settings.milvus_host,
                port=self.settings.milvus_port
            )
            self.milvus_connected = True
            await self._ensure_knowledge_base()
            await self._ensure_user_collections()
            logger.info(f"Milvus initialized safely on {self.settings.milvus_host}:{self.settings.milvus_port}")
        except Exception as e:
            logger.error(f"Milvus initialization failed: {e}")
            self.milvus_connected = False

    async def _ensure_knowledge_base(self, dim: int = 1536) -> None:
        coll_name = "KnowledgeBase"
        if not utility.has_collection(coll_name):
            fields = [
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
                FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            ]
            schema = CollectionSchema(fields, description="Multi-tenant knowledge")
            collection = Collection(coll_name, schema)
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            collection.load()
    
    async def _ensure_user_collections(self) -> None:
        collections = {
            "Users": [
                FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
                FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="email", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="preferences", dtype=DataType.VARCHAR, max_length=4096),
                FieldSchema(name="dummy_vector", dtype=DataType.FLOAT_VECTOR, dim=1),
            ],
            "UserChats": [
                FieldSchema(name="chat_id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
                FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=4096),
                FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="dummy_vector", dtype=DataType.FLOAT_VECTOR, dim=1),
            ]
        }

        for name, fields in collections.items():
            if not utility.has_collection(name):
                self._create_indexed_collection(name, fields, "dummy_vector")

    def _create_indexed_collection(self, name, fields, index_field):
        schema = CollectionSchema(fields)
        col = Collection(name, schema)
        col.create_index(index_field, {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64}
        })
        col.load()

    # -----------------------------
    # PostgreSQL initialization
    # -----------------------------
    async def _init_postgres(self):
        if self.postgres_initialized:
            logger.info("PostgreSQL already initialized, skipping")
            return

        sql = """
        -- Enable UUID generation extension
        CREATE EXTENSION IF NOT EXISTS "pgcrypto";

        -- Tenants table: stores tenant information
        CREATE TABLE IF NOT EXISTS tenants (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );

        -- Users table: stores user accounts per tenant
        CREATE TABLE IF NOT EXISTS users (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
            username VARCHAR(100) NOT NULL,
            email VARCHAR(255),
            password VARCHAR(255),
            role VARCHAR(50) DEFAULT 'user',
            created_at TIMESTAMP DEFAULT NOW()
        );

        -- Documents table: stores documents belonging to tenants
        CREATE TABLE IF NOT EXISTS documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        );

        -- UserChats table: stores chat messages and responses
        CREATE TABLE IF NOT EXISTS user_chats (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
            user_id UUID REFERENCES users(id) ON DELETE SET NULL,
            message TEXT NOT NULL,
            response TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );

        -- Indexes to optimize queries filtered by tenant_id
        CREATE INDEX IF NOT EXISTS idx_documents_tenant_id ON documents(tenant_id);
        CREATE INDEX IF NOT EXISTS idx_user_chats_tenant_id ON user_chats(tenant_id);

        -- Tenant PrePrompts table: stores pre-defined prompts per tenant
        CREATE TABLE IF NOT EXISTS tenant_preprompts (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
            preprompt TEXT NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        );

        -- Index to optimize queries for preprompts by tenant
        CREATE INDEX IF NOT EXISTS idx_tenant_preprompts_tenant_id
        ON tenant_preprompts(tenant_id);


        -- State Dialog Trees table: stores FSMs per tenant
        CREATE TABLE IF NOT EXISTS state_dialog_trees (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
            dialog_key VARCHAR(255) UNIQUE NOT NULL,
            intent VARCHAR(255) NOT NULL,
            version VARCHAR(50) NOT NULL,
            initial_state VARCHAR(50) NOT NULL,
            definition JSONB NOT NULL,
            is_active VARCHAR(10) NOT NULL DEFAULT 'true',
            created_at TIMESTAMP DEFAULT NOW()
        );

        -- Index to optimize queries filtered by tenant_id
        CREATE INDEX IF NOT EXISTS idx_state_dialog_trees_tenant_id
        ON state_dialog_trees(tenant_id);

        """

        try:
            with pg_pool.getconn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                conn.commit()

            self.postgres_initialized = True
            logger.info("PostgreSQL initialized safely")
        except Exception as e:
            logger.error(f"PostgreSQL initialization failed: {e}")

# -----------------------------
# MilvusStore (unchanged)
# -----------------------------
class MilvusStore:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    async def upsert_document(
        self,
        *,
        document_id: str,
        tenant_id: str,
        title: str,
        content: str,
        metadata: Dict[str, Any],
        embedding: List[float],
    ) -> None:
        if not self.db_manager.milvus_connected: return
        
        coll = Collection("KnowledgeBase")
        row = [
            [document_id],
            [tenant_id],
            [title],
            [content],
            [json.dumps(metadata)],
            [datetime.now(timezone.utc).isoformat()],
            [embedding],
        ]
        
        if hasattr(coll, "upsert"):
            coll.upsert(row)
        else:
            try: coll.delete(f'doc_id == "{document_id}"')
            except: pass
            coll.insert(row)
        coll.flush()

    async def list_all_documents(
        self,
        tenant_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        List all documents stored in Milvus.
        - If tenant_id is provided, filter by tenant_id
        - If tenant_id is None, fetch all documents
        """
        try:
            if not self.db_manager.milvus_connected:
                return []

            coll = Collection("KnowledgeBase")
            coll.load()

            # Milvus requires an expression; fetch all if tenant_id not provided
            expr = f'tenant_id == "{tenant_id}"' if tenant_id else 'doc_id != ""'

            results = coll.query(
                expr=expr,
                output_fields=[
                    "doc_id",
                    "tenant_id",
                    "title",
                    "content",
                    "metadata_json",
                    "created_at",
                ],
                limit=limit,
            )

            documents: List[Dict[str, Any]] = []

            for r in results:
                metadata_raw = r.get("metadata_json") or "{}"
                try:
                    metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
                except Exception:
                    metadata = {}

                documents.append({
                    "id": r.get("doc_id"),
                    "tenant_id": r.get("tenant_id"),
                    "title": r.get("title") or "",
                    "content": r.get("content") or "",
                    "metadata": metadata,
                    "created_at": r.get("created_at"),
                    "source": "milvus",  # indicate origin
                    "size": len(r.get("content", "")) if r.get("content") else 0,
                })

            return documents

        except Exception as e:
            logger.error(f"Failed to list documents from Milvus: {e}")
            return []

    async def search(self, tenant_id: str, query_embedding: List[float], limit: int = 5) -> List[Dict]:
        coll = Collection("KnowledgeBase")
        coll.load()
        res = coll.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=limit,
            expr=f'tenant_id == "{tenant_id}"',
            output_fields=["doc_id", "title", "content", "metadata_json"]
        )
        return [ {
            "id": h.entity.get("doc_id"),
            "content": h.entity.get("content"),
            "metadata": json.loads(h.entity.get("metadata_json") or "{}")
        } for h in res[0] ]

# -----------------------------
# MilvusUserManager (unchanged)
# -----------------------------
class MilvusUserManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    async def create_user(self, user_id: str, name: str, email: str, tenant_id: str) -> str:
        coll = Collection("Users")
        data = [[user_id], [name], [email], [tenant_id], ["{}"], [[0.0]]]
        coll.insert(data)
        coll.flush()
        return user_id

    async def get_user(self, user_id: str) -> Optional[Dict]:
        coll = Collection("Users")
        res = coll.query(expr=f'user_id == "{user_id}"', limit=1, output_fields=["*"])
        return res[0] if res else None

    async def create_chat_entry(self, user_id: str, question: str, answer: str) -> str:
        coll = Collection("UserChats")
        chat_id = str(uuid.uuid4())
        data = [[chat_id], [user_id], [question], [answer], [datetime.now(timezone.utc).isoformat()], [[0.0]]]
        coll.insert(data)
        coll.flush()
        return chat_id

# -----------------------------
# Global instances
# -----------------------------
db_manager = DatabaseManager()
milvus_store = MilvusStore(db_manager)
milvus_user_manager = MilvusUserManager(db_manager)
