import uuid
import json
from typing import Dict, Any, List, Optional
from fastapi import HTTPException
from core.database import milvus_store, db_manager
from utils.utility_function import parse_document_metadata
from database.database_function import get_db_connection
from ai.openai_integration import generate_openai_embedding
import logging
from pymilvus import Collection
from datetime import datetime
from config.settings import pg_pool

# Use standard logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------------
# Document Handling
# -----------------------------
async def get_documents_handler(tenant_id: Optional[str] = None):
    try:
        try:
            milvus_documents = await milvus_store.list_all_documents(tenant_id=tenant_id)

            if milvus_documents:
                return {
                    "documents": [
                        {
                            "id": doc["id"],
                            "title": doc["title"],
                            "content": doc["content"],
                            "metadata": parse_document_metadata(doc["metadata"]),
                            "created_at": doc["created_at"],
                            "size": doc.get("file_size", len(doc.get("content", ""))),
                            "status": "processed",
                            "source": "milvus"
                        }
                        for doc in milvus_documents
                    ],
                    "total": len(milvus_documents),
                    "message": "Documents retrieved from Milvus",
                }

        except Exception as e:
            logger.warning(f"Could not retrieve documents from Milvus: {str(e)}")

        conn = get_db_connection()
        if not conn:
            return {
                "documents": [],
                "total": 0,
                "message": "No documents available",
            }

        cursor = conn.cursor()

        if tenant_id:
            cursor.execute(
                """
                SELECT id, title, content, metadata, created_at
                FROM documents
                WHERE tenant_id = %s
                ORDER BY created_at DESC
                """,
                (tenant_id,),
            )
        else:
            cursor.execute(
                """
                SELECT id, title, content, metadata, created_at
                FROM documents
                ORDER BY created_at DESC
                """
            )

        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        documents = [
            {
                "id": str(doc[0]),
                "title": doc[1],
                "content": doc[2],
                "metadata": parse_document_metadata(doc[3]),
                "created_at": doc[4].isoformat() if hasattr(doc[4], "isoformat") else str(doc[4]),
                "size": len(doc[2].encode("utf-8")) if doc[2] else 0,
                "status": "processed",
                "source": "postgres"
            }
            for doc in rows
        ]

        return {
            "documents": documents,
            "total": len(documents),
            "message": "Documents retrieved from database",
        }

    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting documents: {str(e)}",
        )

async def add_document(
    title: str,
    content: str,
    tenant_id: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a tenant document:
    - Generate embedding
    - Save to Milvus and PostgreSQL together
    - If either fails, rollback to maintain consistency
    """
    metadata = metadata or {}
    doc_id = str(uuid.uuid4())
    created_at = datetime.utcnow()

    # --- Generate embedding ---
    embedding = await generate_openai_embedding(content)
    if not isinstance(embedding, list) or len(embedding) != 1536:
        raise HTTPException(status_code=500, detail="Invalid embedding length")

    # --- Milvus ---
    if not db_manager.milvus_connected:
        raise HTTPException(status_code=503, detail="Milvus not connected")

    try:
        await milvus_store.upsert_document(
            document_id=doc_id,
            tenant_id=tenant_id,
            title=title,
            content=content,
            metadata=metadata,
            embedding=embedding
        )
        logger.info(f"Document {doc_id} uploaded to Milvus for tenant {tenant_id}")
    except Exception as e:
        logger.error(f"Error uploading document to Milvus: {e}")
        raise HTTPException(status_code=500, detail=f"Milvus insert failed: {e}")

    # --- PostgreSQL ---
    conn = pg_pool.getconn()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO documents (id, tenant_id, title, content, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (doc_id, tenant_id, title, content, json.dumps(metadata), created_at)
            )
            conn.commit()
            logger.info(f"Document {doc_id} saved in PostgreSQL for tenant {tenant_id}")
    except Exception as e:
        # Rollback Milvus insert if Postgres fails
        try:
            coll = Collection("Documents")
            coll.delete(expr=f'document_id == "{doc_id}"')
            coll.flush()
        except Exception as milvus_del_err:
            logger.error(f"Failed to rollback Milvus after PostgreSQL failure: {milvus_del_err}")

        logger.error(f"Failed to insert document into PostgreSQL: {e}")
        raise HTTPException(status_code=500, detail=f"PostgreSQL insert failed: {e}")
    finally:
        conn.close()

    return doc_id

# -----------------------------
# PrePrompt handler (PostgreSQL only)
# -----------------------------
async def create_preprompt(
    tenant_id: str,
    preprompt: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a tenant preprompt in PostgreSQL only.
    """
    metadata = metadata or {}
    preprompt_id = str(uuid.uuid4())
    created_at = datetime.utcnow()

    # --- PostgreSQL ---
    conn = pg_pool.getconn()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO tenant_preprompts (id, tenant_id, preprompt, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (preprompt_id, tenant_id, preprompt, json.dumps(metadata), created_at)
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to insert preprompt into PostgreSQL: {e}")
        raise HTTPException(status_code=500, detail=f"PostgreSQL insert failed: {e}")
    finally:
        conn.close()

    return preprompt_id

async def get_preprompt_handler(tenant_id: Optional[str] = None) -> Dict:
    """
    Fetch preprompts from PostgreSQL only.
    """
    conn = get_db_connection()
    if not conn:
        logger.warning("No PostgreSQL connection available for preprompts")
        return {"preprompts": []}

    preprompts = []
    try:
        cursor = conn.cursor()

        if tenant_id:
            cursor.execute(
                """
                SELECT id, tenant_id, preprompt, metadata, created_at
                FROM tenant_preprompts
                WHERE tenant_id = %s
                ORDER BY created_at DESC
                """,
                (tenant_id,),
            )
        else:
            cursor.execute(
                """
                SELECT id, tenant_id, preprompt, metadata, created_at
                FROM tenant_preprompts
                ORDER BY created_at DESC
                """
            )

        rows = cursor.fetchall()
        for r in rows:
            metadata_value = r[3]
            if isinstance(metadata_value, str):
                metadata = json.loads(metadata_value)
            elif isinstance(metadata_value, dict):
                metadata = metadata_value
            else:
                metadata = {}

            preprompts.append({
                "id": str(r[0]),
                "tenant_id": r[1],
                "preprompt": r[2],
                "metadata": metadata,
                "created_at": r[4].isoformat() if hasattr(r[4], "isoformat") else str(r[4]),
            })

        return {"preprompts": preprompts}

    except Exception as e:
        logger.error(f"Error fetching preprompts from PostgreSQL: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching preprompts: {e}")

    finally:
        conn.close()

async def update_preprompt_handler(preprompt_id: str, payload: Dict[str, Any]) -> Dict:
    """
    Update a preprompt in PostgreSQL only.
    """
    tenant_id = payload["tenant_id"]
    preprompt_text = payload.get("preprompt")
    metadata = payload.get("metadata")

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        updates = []
        values = []

        if preprompt_text:
            updates.append("preprompt=%s")
            values.append(preprompt_text)
        if metadata:
            updates.append("metadata=%s")
            values.append(json.dumps(metadata))

        if updates:
            values.extend([preprompt_id, tenant_id])
            sql = f"UPDATE tenant_preprompts SET {', '.join(updates)} WHERE id=%s AND tenant_id=%s"
            cursor.execute(sql, values)
            conn.commit()

    finally:
        conn.close()

    return {"preprompt_id": preprompt_id}

async def delete_preprompt_handler(preprompt_id: str) -> Dict:
    """
    Delete a preprompt from PostgreSQL only.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM tenant_preprompts WHERE id=%s",
            (preprompt_id,)
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Preprompt not found")
        conn.commit()
    finally:
        conn.close()

    return {"preprompt_id": preprompt_id}


# -----------------------------
# Dialog tree handler
# -----------------------------
async def create_dialog_tree(
    tenant_id: str,
    dialog_key: str,
    intent: str,
    version: str,
    initial_state: str,
    definition: Dict[str, Any],
    is_active: str = "true"
) -> str:
    dialog_tree_id = str(uuid.uuid4())
    created_at = datetime.utcnow()

    conn = pg_pool.getconn()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO state_dialog_trees
                (id, tenant_id, dialog_key, intent, version, initial_state, definition, is_active, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (dialog_tree_id, tenant_id, dialog_key, intent, version, initial_state, json.dumps(definition), is_active, created_at)
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to insert dialog tree into PostgreSQL: {e}")
        raise HTTPException(status_code=500, detail=f"PostgreSQL insert failed: {e}")
    finally:
        conn.close()

    return dialog_tree_id

async def get_dialog_trees_handler(tenant_id: Optional[str] = None) -> Dict:
    conn = pg_pool.getconn()
    dialog_trees = []

    try:
        cursor = conn.cursor()
        if tenant_id:
            cursor.execute(
                """
                SELECT id, tenant_id, dialog_key, intent, version, initial_state, definition, is_active, created_at
                FROM state_dialog_trees
                WHERE tenant_id = %s
                ORDER BY created_at DESC
                """,
                (tenant_id,)
            )
        else:
            cursor.execute(
                """
                SELECT id, tenant_id, dialog_key, intent, version, initial_state, definition, is_active, created_at
                FROM state_dialog_trees
                ORDER BY created_at DESC
                """
            )

        rows = cursor.fetchall()
        for r in rows:
            definition_value = r[6]
            if isinstance(definition_value, str):
                definition = json.loads(definition_value)
            else:
                definition = definition_value

            dialog_trees.append({
                "id": str(r[0]),
                "tenant_id": r[1],
                "dialog_key": r[2],
                "intent": r[3],
                "version": r[4],
                "initial_state": r[5],
                "definition": definition,
                "is_active": r[7],
                "created_at": r[8].isoformat() if hasattr(r[8], "isoformat") else str(r[8])
            })

        return {"dialog_trees": dialog_trees}

    except Exception as e:
        logger.error(f"Error fetching dialog trees from PostgreSQL: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching dialog trees: {e}")

    finally:
        conn.close()

async def update_dialog_tree_handler(dialog_tree_id: str, payload: Dict[str, Any]) -> Dict:
    conn = pg_pool.getconn()
    source_list = []

    try:
        cursor = conn.cursor()
        updates = []
        values = []

        for key in ["dialog_key", "intent", "version", "initial_state", "definition", "is_active"]:
            if key in payload:
                if key == "definition":
                    updates.append(f"{key}=%s")
                    values.append(json.dumps(payload[key]))
                else:
                    updates.append(f"{key}=%s")
                    values.append(payload[key])

        if updates:
            values.append(dialog_tree_id)
            sql = f"UPDATE state_dialog_trees SET {', '.join(updates)} WHERE id=%s"
            cursor.execute(sql, values)
            conn.commit()
            source_list.append("postgres")

    except Exception as e:
        logger.error(f"Failed to update dialog tree in PostgreSQL: {e}")
        raise HTTPException(status_code=500, detail=f"PostgreSQL update failed: {e}")

    finally:
        conn.close()

    return {"source": source_list, "dialog_tree_id": dialog_tree_id}


async def delete_dialog_tree_handler(dialog_tree_id: str) -> Dict:
    conn = pg_pool.getconn()
    source_list = []

    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM state_dialog_trees WHERE id=%s", (dialog_tree_id,))
        if cursor.rowcount > 0:
            source_list.append("postgres")
        conn.commit()

    except Exception as e:
        logger.error(f"Failed to delete dialog tree in PostgreSQL: {e}")
        raise HTTPException(status_code=500, detail=f"PostgreSQL delete failed: {e}")

    finally:
        conn.close()

    if not source_list:
        raise HTTPException(status_code=404, detail="Dialog tree not found in PostgreSQL")

    return {"source": source_list, "dialog_tree_id": dialog_tree_id}