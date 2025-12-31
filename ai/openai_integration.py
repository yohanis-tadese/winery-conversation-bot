import os
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from fastapi import HTTPException

import time
import random
import asyncio

logger = logging.getLogger(__name__)

# Configure OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-5')
OPENAI_EMBEDDING_MODEL = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')

# Initialize OpenAI client
from openai import OpenAI
import openai  # keep to reference exception classes (RateLimitError, etc.)

openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        safe_key_head = OPENAI_API_KEY[:10]
        safe_key_tail = OPENAI_API_KEY[-5:]
        logger.info(f"OpenAI client initialized successfully with key: {safe_key_head}...{safe_key_tail}")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        openai_client = None
else:
    logger.warning("OPENAI_API_KEY not found in environment variables. Embeddings/answers will be unavailable.")


# Embedding cache to avoid regenerating embeddings within the same request
embedding_cache = {}


async def generate_openai_embedding(text: str, cache_key: str = None) -> List[float]:
    """Generate embedding using OpenAI API with caching and token limit handling."""
    if cache_key and cache_key in embedding_cache:
        logger.info(f"Using cached embedding for key: {cache_key[:50]}...")
        return embedding_cache[cache_key]
    
    if not openai_client:
        logger.error("OpenAI client not available - cannot generate embeddings")
        raise HTTPException(status_code=503, detail="OpenAI service not available - embeddings cannot be generated")
    
    max_chars = 8192 * 3  # conservative
    if len(text) > max_chars:
        logger.warning(f"Text too long ({len(text)} chars), truncating to {max_chars} chars for embedding.")
        truncated_text = text[:max_chars]
        last_space = truncated_text.rfind(' ')
        if last_space > max_chars * 0.9:
            truncated_text = truncated_text[:last_space]
        text = truncated_text
    
    try:
        logger.info(f"Calling OpenAI embeddings API with model: {OPENAI_EMBEDDING_MODEL}, text length: {len(text)} chars")
        response = openai_client.embeddings.create(
            input=text,
            model=OPENAI_EMBEDDING_MODEL
        )
        embedding = response.data[0].embedding
        if cache_key:
            embedding_cache[cache_key] = embedding
        return embedding
        
    except openai.AuthenticationError as e:
        logger.error(f"OpenAI authentication error: {e}")
        raise HTTPException(status_code=401, detail="OpenAI authentication failed - check API key")
    except openai.RateLimitError as e:
        logger.error(f"OpenAI rate limit error: {e}")
        raise HTTPException(status_code=429, detail="OpenAI rate limit exceeded - please try again later")
    except openai.APIError as e:
        logger.error(f"OpenAI API error: {e}")
        if "maximum context length" in str(e).lower():
            shorter_text = text[: (8192 * 3) // 2]
            last_space = shorter_text.rfind(' ')
            if last_space > len(shorter_text) * 0.8:
                shorter_text = shorter_text[:last_space]
            try:
                response = openai_client.embeddings.create(
                    input=shorter_text,
                    model=OPENAI_EMBEDDING_MODEL
                )
                embedding = response.data[0].embedding
                if cache_key:
                    embedding_cache[cache_key] = embedding
                return embedding
            except Exception as retry_error:
                raise HTTPException(status_code=502, detail=f"OpenAI API error even with shortened text: {retry_error}")
        else:
            raise HTTPException(status_code=502, detail=f"OpenAI API error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error calling OpenAI API: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error generating embedding: {str(e)}")


async def generate_openai_answer(question: str, context: str) -> Dict[str, Any]:
    """Generate answer using OpenAI GPT model (single-turn)."""
    if not openai_client:
        raise HTTPException(status_code=503, detail="OpenAI service not available - answers cannot be generated")
    
    system_prompt = (
        "You are an expert AI assistant that provides comprehensive, detailed, and highly accurate answers. "
        "When documents are provided in the context and are relevant to the question, you MUST reference and cite them explicitly throughout your response with specific details and quotes. "
        "When no documents are provided or when answering general questions, provide helpful, accurate responses based on your knowledge. "
        "Always structure your responses professionally with clear sections, detailed analysis, and comprehensive coverage of the topic. "
        "IMPORTANT: If personality traits are specified in the context, you must embody those traits consistently throughout your entire response, adjusting your tone, style, and manner of communication accordingly. "
        "Aim for responses that are detailed, well-structured, and demonstrate understanding while maintaining the specified personality and communication style.Explain everything in detail ,donot wirte citations. in answer like [1] this ."
    )
    # Augment: context handling + strict citation behavior (default only)
    if not system_prompt:
        sys = (
            "You are an expert AI assistant that provides comprehensive, accurate, and concise answers.\n\n"
            "CRITICAL CONTEXT HANDLING:\n"
            "- If the user mentions 'attached document', 'uploaded file', 'the document', 'this file', or similar phrases, they are referring to the text provided in the 'Additional context for this turn' section below.\n"
            "- You MUST use that context as your PRIMARY source of information for such questions.\n"
            "- If context is provided, assume it is the 'document' the user is asking about.\n"
            "- DO NOT say 'I don't see an attachment' if context is present. The context IS the attachment.\n\n"
            "CITATION RULES:\n"
            "If a SOURCES block or a list of numbered sources appears in earlier messages, you MUST:\n"
            "  1) Add inline citation markers like [1], [2] immediately after the specific sentences using those sources.\n"
            "  2) Append a final section titled 'References' listing each source as: [n] Title - URL.\n"
            "Never fabricate citations or URLs. If no SOURCES are provided, do not include citations."
        )
    
    user_prompt = f"Context: {context}\n\nQuestion: {question}\n\nPlease provide a comprehensive, detailed answer..."
    
    model = OPENAI_MODEL
    
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        choice = response.choices[0]
        answer = (choice.message.content or "").strip()
        usage = getattr(response, "usage", None)
        tokens_used = getattr(usage, "total_tokens", 0) or 0
        if not answer:
            answer = "I apologize, but I wasn't able to generate a comprehensive answer."
        return {"answer": answer, "model_used": model, "tokens_used": tokens_used}
    except openai.AuthenticationError as e:
        raise HTTPException(status_code=401, detail="OpenAI authentication failed - check API key")
    except openai.RateLimitError as e:
        raise HTTPException(status_code=429, detail="OpenAI rate limit exceeded - please try again later")
    except openai.APIError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error generating answer: {str(e)}")


# --- History-aware answer (non-streaming) -------------------------------------

_HISTORY_CHAR_BUDGET = int(8192 * 3.2)
_USER_MSG_CHAR_BUDGET = int(2048 * 3.2)

def _prune_history_by_chars(messages: List[Dict[str, str]], budget: int) -> List[Dict[str, str]]:
    total = 0
    pruned: List[Dict[str, str]] = []
    for m in reversed(messages):
        sz = len(m.get("content", ""))
        if total + sz > budget and pruned:
            break
        pruned.append(m)
        total += sz
    pruned.reverse()
    return pruned


async def generate_openai_answer_with_history(
    history_messages: List[Dict[str, str]],
    user_question: str,
    system_prompt: Optional[str] = None,
    context: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate an answer using OpenAI GPT model with prior conversation history + strict citation behavior."""
    if not openai_client:
        raise HTTPException(status_code=503, detail="OpenAI service not available - answers cannot be generated")

    # Strong default: only cite when SOURCES are provided
    sys = system_prompt or (
        "You are an expert AI assistant that provides comprehensive, accurate, and concise answers.\n"
        "If a SOURCES block or a list of numbered sources appears in earlier messages, you MUST:\n"
        "  1) Add inline citation markers like [1], [2] immediately after the specific sentences using those sources.\n"
        "  2) Append a final section titled 'References' listing each source as: [n] Title — URL.\n"
        "Never fabricate citations or URLs. If no SOURCES are provided, do not include citations."
    )
    # Augment: context handling + strict citation behavior (default only)
    if not system_prompt:
        sys = (
            "You are an expert AI assistant that provides comprehensive, accurate, and concise answers.\n\n"
            "CRITICAL CONTEXT HANDLING:\n"
            "- If the user mentions 'attached document', 'uploaded file', 'the document', 'this file', or similar phrases, they are referring to the text provided in the 'Additional context for this turn' section below.\n"
            "- You MUST use that context as your PRIMARY source of information for such questions.\n"
            "- If context is provided, assume it is the 'document' the user is asking about.\n"
            "- DO NOT say 'I don't see an attachment' if context is present. The context IS the attachment.\n\n"
            "CITATION RULES:\n"
            "If a SOURCES block or a list of numbered sources appears in earlier messages, you MUST:\n"
            "  1) Add inline citation markers like [1], [2] immediately after the specific sentences using those sources.\n"
            "  2) Append a final section titled 'References' listing each source as: [n] Title - URL.\n"
            "Never fabricate citations or URLs. If no SOURCES are provided, do not include citations."
        )

    messages: List[Dict[str, str]] = [{"role": "system", "content": sys}]
    if context:
        messages.append({"role": "system", "content": f"Additional context for this turn:\n{context}"})
    for m in (history_messages or []):
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role in ("system", "user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_question})

    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
        )
        choice = response.choices[0]
        answer = (choice.message.content or "").strip()
        usage = getattr(response, "usage", None)
        tokens_used = getattr(usage, "total_tokens", 0) or 0
        if not answer:
            answer = "I apologize, but I couldn't generate a response for that."
        return {"answer": answer, "model_used": OPENAI_MODEL, "tokens_used": tokens_used}
    except openai.AuthenticationError as e:
        raise HTTPException(status_code=401, detail="OpenAI authentication failed - check API key")
    except openai.RateLimitError as e:
        raise HTTPException(status_code=429, detail="OpenAI rate limit exceeded - please try again later")
    except openai.APIError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error generating answer: {str(e)}")

# --- NEW: History-aware answer (STREAMING) ------------------------------------

async def generate_openai_answer_with_history_stream(
    history_messages: List[Dict[str, str]],
    user_question: str,
    system_prompt: Optional[str] = None,
    context: Optional[str] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Streaming with strict citation behavior:
    - If sources are present in prior messages (SOURCES block or numbered list), enforce [n] markers + 'References'.
    """
    if not openai_client:
        yield {"type": "error", "message": "OpenAI service not available - answers cannot be generated"}
        return

    sys = system_prompt or (
        "You are an expert AI assistant that provides comprehensive, accurate, and concise answers.\n"
        "If a SOURCES block or a list of numbered sources appears in earlier messages, you MUST:\n"
        "  1) Add inline citation markers like [1], [2] immediately after the specific sentences using those sources.\n"
        "  2) Append a final section titled 'References' listing each source as: [n] Title — URL.\n"
        "Never fabricate citations or URLs. If no SOURCES are provided, do not include citations."
    )

    messages: List[Dict[str, str]] = [{"role": "system", "content": sys}]
    if context:
        messages.append({"role": "system", "content": f"Additional context for this turn:\n{context}"})
    for m in (history_messages or []):
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role in ("system", "user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_question})

    try:
        model = OPENAI_MODEL
        model_used = model
        tokens_used: Optional[int] = None

        stream_ctx = getattr(openai_client.chat.completions, "stream", None)
        if callable(stream_ctx):
            # Modern streaming context manager
            async with stream_ctx(model=model, messages=messages) as stream:
                yield {"type": "start", "model_used": model_used}
                async for event in stream:
                    try:
                        delta = getattr(event, "delta", None)
                        if delta and getattr(delta, "content", None):
                            yield {"type": "delta", "content": delta.content}
                        if getattr(event, "type", "") in ("response.completed", "response.refusal.delta"):
                            usage = getattr(stream, "usage", None)
                            if usage and getattr(usage, "total_tokens", None):
                                tokens_used = usage.total_tokens
                    except Exception:
                        pass
                yield {"type": "done", "tokens_used": tokens_used}
                return

        # Fallback: legacy stream=True
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            stream=True,
        )
        yield {"type": "start", "model_used": model_used}
        for chunk in response:
            try:
                choice = chunk.choices[0]
                piece = getattr(choice, "delta", None)
                text = ""
                if piece is not None:
                    text = getattr(piece, "content", "") or ""
                else:
                    msg = getattr(choice, "message", None)
                    text = getattr(msg, "content", "") or ""
                if text:
                    yield {"type": "delta", "content": text}
            except Exception:
                pass
        yield {"type": "done", "tokens_used": None}
        return

    except openai.AuthenticationError:
        yield {"type": "error", "message": "OpenAI authentication failed - check API key"}
    except openai.RateLimitError:
        yield {"type": "error", "message": "OpenAI rate limit exceeded - please try again later"}
    except openai.APIError as e:
        yield {"type": "error", "message": f"OpenAI API error: {str(e)}"}
    except Exception as e:
        # Final fallback: non-streaming + chunk
        try:
            non_stream = await generate_openai_answer_with_history(
                history_messages=history_messages,
                user_question=user_question,
                system_prompt=system_prompt,
                context=context,
            )
            yield {"type": "start", "model_used": non_stream.get("model_used")}
            text = (non_stream.get("answer") or "").strip()
            chunk_size = 120
            for i in range(0, len(text), chunk_size):
                await asyncio.sleep(0)
                yield {"type": "delta", "content": text[i:i+chunk_size]}
            yield {"type": "done", "tokens_used": non_stream.get("tokens_used")}
        except Exception as ee:
            yield {"type": "error", "message": f"LLM error: {ee}"}

# ai/openai_integration.py

from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)
from openai import OpenAI;
client = OpenAI()
# import your existing chat helper
# adjust this import if your function is named differently


async def llm_should_use_web_search(
    question: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    # Build tiny history window (optional)
    history_text = ""
    if history:
        last_msgs = history[-2:]
        pieces = []
        for m in last_msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            pieces.append(f"{role}: {content}")
        history_text = "\n".join(pieces)

    system_prompt = (
        "You are a strict boolean router that decides if a user's question "
        "needs external web search.\n\n"
        "Rules:\n"
        "- Answer 'true' if the question clearly needs fresh, external, or factual data "
        "  that is unlikely to be contained in the local chat/document context "
        "  (e.g. latest news, current prices, live sports scores, recent events).\n"
        "- Answer 'false' if the question can be answered purely from the existing "
        "  conversation or documents (e.g. 'explain above', 'summarize previous', "
        "  'rewrite this', 'continue', 'clarify that', 'improve the text', etc.).\n"
        "- When in doubt, prefer 'false'.\n"
        "- Your entire response MUST be exactly 'true' or 'false' in lowercase, with no extra text."
    )

    user_prompt = (
        (f"Recent conversation:\n{history_text}\n\n" if history_text else "") +
        "Question:\n"
        f"{question}\n\n"
        "Should we use web search? Reply with exactly 'true' or 'false'."
    )

    try:
        resp = await client.chat.completions.create(
            model= "gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=16,
        )

        router_text = resp.choices[0].message.content or ""
        ans = router_text.strip().lower()

        if ans == "false":
            return False
        if ans == "true":
            return True

        logger.warning(f"Router LLM returned unexpected content: {router_text!r}, defaulting to False")
        return False

    except Exception as e:
        logger.warning(f"llm_should_use_web_search failed: {e}", exc_info=True)
        return False
