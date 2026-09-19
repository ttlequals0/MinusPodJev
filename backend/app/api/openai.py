"""OpenAI-compatible chat-completions endpoint MinusPod points at.

Mounted at the root so both /chat/completions and /v1/chat/completions resolve
(MinusPod's base_url may or may not include /v1). GET /models answers client
probes with the configured Jev model id.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict

from app.config import settings
from app.services.openai_adapter import run_chat_completion

router = APIRouter()


def _bearer(authorization: str | None) -> str | None:
    """Extract the token from an `Authorization: Bearer <token>` header."""
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip() or None
    return None


def _resolve_api_key(authorization: str | None) -> str:
    """Prefer the caller's bearer token (MinusPod's provider api_key is the
    TypeSafe key); fall back to the proxy's own TYPESAFE_API_KEY."""
    key = _bearer(authorization) or settings.TYPESAFE_API_KEY
    if not key:
        raise HTTPException(
            status_code=503,
            detail=(
                "No TypeSafe API key. Send it as the Authorization bearer token "
                "(MinusPod provider api_key) or set TYPESAFE_API_KEY on the proxy."
            ),
        )
    return key


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

    return run_chat_completion(
        messages=[m.model_dump() for m in request.messages],
        request_model=request.model or settings.JEV_PUBLIC_MODEL,
        url=settings.TYPESAFE_API_URL,
        api_key=api_key,
        timeout=settings.JEV_TIMEOUT_SECONDS,
        cache_path=settings.JEV_CACHE_PATH,
        # Always call upstream (and key the cache) with the real Jev model,
        # whatever public id the caller sent; echo the public id back.
        model=settings.JEV_MODEL,
        enter=settings.JEV_ENTER,
        stay=settings.JEV_STAY,
        category_pass=settings.JEV_CATEGORY_PASS,
        category_context=settings.JEV_CATEGORY_CONTEXT,
        default_category=settings.JEV_DEFAULT_CATEGORY,
    )


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
