import json
import logging

logger = logging.getLogger(__name__)


def safe_get_field(doc, field: str, default=None):
    try:
        if isinstance(doc, dict):
            return doc.get(field, default)

        if hasattr(doc, "__getitem__"):
            try:
                value = doc[field]
                return value if value is not None else default
            except Exception:
                return default

        return default
    except Exception:
        return default


def get_document_fields(doc, index: int = 0):
    return {
        "id": safe_get_field(doc, "id", f"doc_{index}"),
        "title": safe_get_field(doc, "title", "Untitled"),
        "content": safe_get_field(doc, "content", ""),
        "metadata": safe_get_field(doc, "metadata", {}),
        "created_at": safe_get_field(doc, "created_at", ""),
    }


def parse_document_metadata(metadata):
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            return json.loads(metadata)
        except Exception:
            return {}
    return {}
