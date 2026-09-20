"""OpenAI-compatible chat-completions endpoint MinusPod points at.

Mounted at the root so both /chat/completions and /v1/chat/completions resolve
(MinusPod's base_url may or may not include /v1). GET /models answers client
probes with the configured Jev model id.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict

from app.config import settings
from app.services.openai_adapter import (
    CallerPolicyTooLongError,
    ReviewUnavailableError,
    extract_system_text,
    extract_user_text,
    run_chat_completion,
)

router = APIRouter()
_REQUEST_SLOTS = threading.BoundedSemaphore(
    settings.JEV_MAX_CONCURRENT_REQUESTS
)
_SEGMENT_LINE = re.compile(r"^\s*\[\d+(?:\.\d+)?s\s*-|^\s*\[\d+\]")


def _bearer(authorization: str | None) -> str | None:
    """Extract the token from an `Authorization: Bearer <token>` header."""
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip() or None
    return None


def _resolve_api_key(authorization: str | None) -> str:
    """Prefer the caller's bearer token (MinusPod's provider api_key is the
    TypeSafe key); fall back to the proxy's own TYPESAFE_API_KEY."""
    bearer = _bearer(authorization)
    allow_fallback = settings.JEV_ALLOW_UNAUTHENTICATED_FALLBACK
    if not bearer and not settings.TYPESAFE_API_KEY:
        raise HTTPException(status_code=503, detail="No TypeSafe API key configured")
    if not bearer and not allow_fallback:
        raise HTTPException(status_code=401, detail="Bearer token required")
    key = bearer or settings.TYPESAFE_API_KEY
    if not key:
        raise HTTPException(
            status_code=503,
            detail=(
                "No TypeSafe API key. Send it as the Authorization bearer token "
                "(MinusPod provider api_key) or set TYPESAFE_API_KEY on the proxy."
            ),
        )
    return key


def _guard_request(messages: list[ChatMessage]) -> None:
    parsed_messages = [message.model_dump() for message in messages]
    text = extract_user_text(parsed_messages)
    max_chars = settings.JEV_MAX_TRANSCRIPT_CHARS
    max_segments = settings.JEV_MAX_SEGMENTS
    if len(text) > max_chars:
        raise HTTPException(status_code=413, detail="Transcript exceeds configured size limit")
    if sum(1 for line in text.splitlines() if _SEGMENT_LINE.search(line)) > max_segments:
        raise HTTPException(status_code=413, detail="Transcript exceeds configured segment limit")
    if len(extract_system_text(parsed_messages).strip()) > 12_000:
        raise HTTPException(status_code=422, detail="System policy exceeds configured size limit")


class ChatMessage(BaseModel):
    role: str
    content: Any = None

    model_config = ConfigDict(extra="allow")


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = []
    response_format: Any = None
    temperature: float | None = None

    model_config = ConfigDict(extra="allow")


@router.post("/chat/completions")
def chat_completions(
    request: ChatCompletionRequest, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    """Turn an OpenAI chat request into a Jev detection and shape the reply."""
    api_key = _resolve_api_key(authorization)
    _guard_request(request.messages)
    if not _REQUEST_SLOTS.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Proxy is at capacity")

    try:
        try:
            return run_chat_completion(
                messages=[m.model_dump() for m in request.messages],
                request_model=request.model or settings.JEV_PUBLIC_MODEL,
                url=settings.TYPESAFE_API_URL,
                api_key=api_key,
                timeout=settings.JEV_TIMEOUT_SECONDS,
                cache_path=settings.JEV_CACHE_PATH,
                model=settings.JEV_MODEL,
                enter=settings.JEV_ENTER,
                stay=settings.JEV_STAY,
                category_pass=settings.JEV_CATEGORY_PASS,
                category_context=settings.JEV_CATEGORY_CONTEXT,
                default_category=settings.JEV_DEFAULT_CATEGORY,
                max_retries=settings.JEV_MAX_RETRIES,
                cache_max_entries=settings.JEV_CACHE_MAX_ENTRIES,
                request_deadline=settings.JEV_REQUEST_DEADLINE_SECONDS,
                retry_after_max=settings.JEV_RETRY_AFTER_MAX_SECONDS,
            )
        except ReviewUnavailableError as exc:
            raise HTTPException(status_code=503, detail="Review unavailable") from exc
        except CallerPolicyTooLongError as exc:
            raise HTTPException(status_code=422, detail="System policy exceeds configured size limit") from exc
    finally:
        _REQUEST_SLOTS.release()


@router.get("/models")
def list_models() -> dict[str, Any]:
    """Minimal OpenAI model list advertising the public Jev model id."""
    return {
        "object": "list",
        "data": [
            {
                "id": settings.JEV_PUBLIC_MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "typesafe",
            }
        ],
    }
