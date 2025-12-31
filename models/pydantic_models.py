from typing import Dict, Any, Optional
from pydantic import BaseModel

class PrePromptCreateRequest(BaseModel):
    tenant_id: str
    preprompt: str
    metadata: Dict[str, Any] = {}


class QuestionRequest(BaseModel):
    question: str
    context: Dict[str, Any] = {}

class DocumentCreateRequest(BaseModel):
    title: str
    content: str
    tenant_id: str
    metadata: Dict[str, Any]

class DialogTreeCreateRequest(BaseModel):
    tenant_id: str
    dialog_key: str
    intent: str
    version: str
    initial_state: str
    definition: Dict
    is_active: Optional[str] = "true"

class DialogTreeUpdateRequest(BaseModel):
    dialog_key: Optional[str]
    intent: Optional[str]
    version: Optional[str]
    initial_state: Optional[str]
    definition: Optional[Dict]
    is_active: Optional[str]