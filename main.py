import logging
from fastapi import FastAPI, Query, HTTPException
from datetime import datetime, timezone
from typing import Optional
from lifecycle.app_lifecycle import lifespan
from models.pydantic_models import ( 
    PrePromptCreateRequest,
    DocumentCreateRequest,
    DialogTreeCreateRequest,
    DialogTreeUpdateRequest
)
from endpoints.api_endpoints import ( 
    add_document, get_documents_handler, create_preprompt, get_preprompt_handler, 
    update_preprompt_handler, delete_preprompt_handler, get_dialog_trees_handler, 
    create_dialog_tree, update_dialog_tree_handler, delete_dialog_tree_handler
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app with imported lifespan
app = FastAPI(
    title="Winery Chatbot API",
    version="2.0.0",
    lifespan=lifespan,
)

@app.get("/")
async def root():
    return {
        "message": "Winery Chatbot API",
        "version": "1.0.1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "docs": "/docs"
    }

@app.get("/api/v1/documents")
async def get_documents(tenant_id: Optional[str] = Query(None, description="Tenant ID to filter documents")):
    documents = await get_documents_handler(tenant_id=tenant_id)
    return {"status": "success", "documents": documents or []}


@app.post("/api/v1/documents")
async def create_document(request: DocumentCreateRequest):
    doc_id = await add_document(
        title=request.title,
        content=request.content,
        tenant_id=request.tenant_id,
        metadata=request.metadata,
    )
    return {"status": "success", "document_id": doc_id}

@app.get("/api/v1/preprompts")
async def get_tenant_preprompt(tenant_id: Optional[str] = Query(None, description="Tenant ID to fetch preprompt")):
    preprompts = await get_preprompt_handler(tenant_id)
    if tenant_id and not preprompts:
        raise HTTPException(status_code=404, detail="Preprompt not found for the given tenant ID")
    
    return {"status": "success", "preprompts": preprompts}

@app.post("/api/v1/preprompts")
async def create_tenant_preprompt(request: PrePromptCreateRequest):
    preprompt_id = await create_preprompt(
        tenant_id=request.tenant_id,
        preprompt=request.preprompt,
        metadata=request.metadata
    )
    return {"status": "success", "preprompt_id": preprompt_id}

@app.put("/api/v1/preprompts/{preprompt_id}")
async def update_tenant_preprompt(preprompt_id: str, payload: dict):
    result = await update_preprompt_handler(preprompt_id, payload)
    return {"status": "success", **result}


@app.delete("/api/v1/preprompts/{preprompt_id}")
async def delete_tenant_preprompt(preprompt_id: str):
    result = await delete_preprompt_handler(preprompt_id)
    return {"status": "success", **result}


# -----------------------------
# Dialog Tree Handling
# -----------------------------
@app.get("/api/v1/dialog_trees")
async def get_tenant_dialog_trees(tenant_id: Optional[str] = Query(None, description="Tenant ID to fetch dialog trees")):
    dialog_trees = await get_dialog_trees_handler(tenant_id)
    return {"status": "success", "dialog_trees": dialog_trees or []}

@app.post("/api/v1/dialog_trees")
async def create_tenant_dialog_tree(request: DialogTreeCreateRequest):
    dialog_tree_id = await create_dialog_tree(
        tenant_id=request.tenant_id,
        dialog_key=request.dialog_key,
        intent=request.intent,
        version=request.version,
        initial_state=request.initial_state,
        definition=request.definition,
        is_active=request.is_active
    )
    return {"status": "success", "dialog_tree_id": dialog_tree_id}

@app.put("/api/v1/dialog_trees/{dialog_tree_id}")
async def update_tenant_dialog_tree(dialog_tree_id: str, payload: DialogTreeUpdateRequest):
    result = await update_dialog_tree_handler(dialog_tree_id, payload.dict(exclude_unset=True))
    return {"status": "success", **result}

@app.delete("/api/v1/dialog_trees/{dialog_tree_id}")
async def delete_tenant_dialog_tree(dialog_tree_id: str):
    result = await delete_dialog_tree_handler(dialog_tree_id)
    return {"status": "success", **result}