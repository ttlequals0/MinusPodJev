"""OpenAI-compatible chat-completions endpoint MinusPod points at.

Mounted at the root so both /chat/completions and /v1/chat/completions resolve
(MinusPod's base_url may or may not include /v1). GET /models answers client
probes with the configured Jev model id.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from app.config import settings
from app.services.jev import _retry_after_seconds
from app.services.openai_adapter import (
    ReviewInconclusiveError,
    ReviewInvalidRequestError,
    ReviewUnavailableError,
    ReviewUpstreamInvalidResponseError,
    extract_system_text,
    extract_user_text,
    is_review_request,
    parse_candidate_bounds,
    run_chat_completion,
)
from app.services.runtime_settings import RuntimeSettingsError, effective_from_settings
from app.utils.metrics import metrics

router = APIRouter()
logger = logging.getLogger(__name__)
_REQUEST_SLOTS = threading.BoundedSemaphore(
    settings.JEV_MAX_CONCURRENT_REQUESTS
)


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


@router.post("/chat/completions", response_model=None)
def chat_completions(
    request: ChatCompletionRequest, authorization: str | None = Header(default=None)
) -> dict[str, Any] | JSONResponse:
    """Turn an OpenAI chat request into a Jev detection and shape the reply."""
    api_key = _resolve_api_key(authorization)
    if not _REQUEST_SLOTS.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Proxy is at capacity")

    messages = [m.model_dump() for m in request.messages]
    review = is_review_request(extract_user_text(messages), extract_system_text(messages))
    request_id = uuid.uuid4().hex if review else None
    started = time.monotonic()
    outcome: str | None = None
    reason_code: str | None = None
    try:
        try:
            thresholds = effective_from_settings(settings)
        except RuntimeSettingsError as exc:
            raise HTTPException(status_code=503, detail="Runtime settings unavailable") from exc
        try:
            response = run_chat_completion(
                messages=messages,
                request_model=request.model or settings.JEV_PUBLIC_MODEL,
                url=settings.TYPESAFE_API_URL,
                api_key=api_key,
                timeout=settings.JEV_TIMEOUT_SECONDS,
                cache_path=settings.JEV_CACHE_PATH,
                model=settings.JEV_MODEL,
                enter=thresholds.detection_enter,
                stay=thresholds.detection_stay,
                review_evidence_enter=thresholds.review_evidence,
                review_choice_enter=thresholds.review_choice,
                category_pass=settings.JEV_CATEGORY_PASS,
                category_context=settings.JEV_CATEGORY_CONTEXT,
                default_category=settings.JEV_DEFAULT_CATEGORY,
                max_retries=settings.JEV_MAX_RETRIES,
                cache_max_entries=settings.JEV_CACHE_MAX_ENTRIES,
                request_deadline=settings.JEV_REQUEST_DEADLINE_SECONDS,
                retry_after_max=settings.JEV_RETRY_AFTER_MAX_SECONDS,
                refine_boundaries=settings.JEV_REVIEW_REFINE_BOUNDARIES,
                review_request_id=request_id,
            )
            if review:
                outcome = _review_outcome(response, extract_user_text(messages))
                return JSONResponse(content=response, headers={"X-Request-ID": request_id or ""})
            return response
        except ReviewInvalidRequestError:
            outcome, reason_code = "invalid_request", "invalid_request"
            return _review_error(422, "jev_review_invalid_request", "Review request is invalid", request_id)
        except ReviewInconclusiveError as exc:
            outcome, reason_code = "inconclusive", _inconclusive_reason(str(exc))
            return _review_error(422, "jev_review_inconclusive", "Review is inconclusive", request_id)
        except ReviewUpstreamInvalidResponseError:
            outcome, reason_code = "upstream_error", "upstream_invalid_response"
            return _review_error(503, "jev_review_upstream_invalid_response", "Review unavailable", request_id)
        except ReviewUnavailableError:
            outcome, reason_code = "upstream_error", "upstream_failure"
            return _review_error(503, "jev_review_upstream_failure", "Review unavailable", request_id)
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.TransportError) as exc:
            outcome, reason_code = "upstream_error", "upstream_failure"
            return _upstream_error_response(exc, request_id)
        except Exception:
            if review:
                outcome, reason_code = "internal_error", "internal_error"
            raise
    finally:
        _REQUEST_SLOTS.release()
        if review:
            metrics.record_review(outcome or "internal_error", (time.monotonic() - started) * 1000, reason_code)
            logger.info("review request_id=%s outcome=%s reason=%s bounds=%s elapsed_ms=%.0f", request_id, outcome or "internal_error", reason_code or "unknown", parse_candidate_bounds(extract_user_text(messages)), (time.monotonic() - started) * 1000)


def _inconclusive_reason(message: str) -> str:
    lowered = message.lower()
    if "unambiguous" in lowered:
        return "ambiguous_spans"
    if "evidence" in lowered:
        return "insufficient_evidence"
    if "choice" in lowered:
        return "choice_inconclusive"
    return "malformed_context"


def _review_error(status: int, code: str, message: str, request_id: str | None) -> JSONResponse:
    headers: dict[str, str] = {}
    if status == 422:
        headers["x-should-retry"] = "false"
    if request_id is not None:
        headers["X-Request-ID"] = request_id
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": "invalid_request_error" if status == 422 else "api_error", "code": code}},
        headers=headers,
    )


def _upstream_error_response(exc: Exception, request_id: str | None = None) -> JSONResponse:
    status = 503
    code = "jev_upstream_failure"
    message = "Jev upstream unavailable"
    headers: dict[str, str] = {}
    upstream_status: int | None = None
    if isinstance(exc, httpx.HTTPStatusError):
        upstream_status = exc.response.status_code
        if 400 <= upstream_status < 500:
            status = upstream_status
            code = "jev_upstream_request_error"
            message = "Jev upstream rejected the request"
        if upstream_status == 429:
            code = "jev_upstream_rate_limited"
            message = "Jev upstream rate limited the request"
        if upstream_status in (401, 403):
            code = "jev_upstream_authentication_error"
            message = "Jev upstream authentication failed"
        if upstream_status == 408 or upstream_status == 429 or upstream_status >= 500:
            try:
                retry_after = _retry_after_seconds(exc.response)
            except (OverflowError, TypeError, ValueError):
                retry_after = None
            if retry_after is not None:
                headers["Retry-After"] = str(math.ceil(retry_after))
    elif isinstance(exc, httpx.TimeoutException):
        status = 504
        code = "jev_upstream_timeout"
        message = "Jev upstream timed out"
    elif isinstance(exc, httpx.TransportError):
        code = "jev_upstream_connection_error"
        message = "Jev upstream connection failed"
    logger.warning(
        "Jev upstream failure upstream_status=%s mapped_status=%s exception=%s request_id=%s",
        upstream_status,
        status,
        type(exc).__name__,
        request_id,
    )
    if request_id is not None:
        headers["X-Request-ID"] = request_id
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": "api_error", "code": code}},
        headers=headers,
    )


def _review_outcome(response: dict[str, Any], user_text: str) -> str:
    content = response["choices"][0]["message"]["content"]
    ads = json.loads(content).get("ads", [])
    if not ads:
        return "rejected"
    bounds = parse_candidate_bounds(user_text)
    if bounds is None:
        return "confirmed"
    ad = ads[0]
    if abs(float(ad.get("start", bounds[0])) - bounds[0]) > 0.1 or abs(float(ad.get("end", bounds[1])) - bounds[1]) > 0.1:
        return "adjusted"
    return "confirmed"


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
