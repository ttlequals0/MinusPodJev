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
from app.services.jev import JevCategoryValidationError, _retry_after_seconds
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
    sanitize_review_range_diagnostic,
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
        except ReviewInvalidRequestError as exc:
            outcome, reason_code = "invalid_request", "invalid_request"
            diagnostics = _invalid_request_diagnostics(exc)
            return _review_error(
                422,
                "jev_review_invalid_request",
                "Review request is invalid",
                request_id,
                diagnostics["reason"],
                diagnostics,
            )
        except ReviewInconclusiveError as exc:
            diagnostics = _inconclusive_diagnostics(exc)
            outcome = "inconclusive"
            reason_code = _metric_inconclusive_reason(diagnostics)
            return _review_error(
                422,
                "jev_review_inconclusive",
                _inconclusive_message(diagnostics),
                request_id,
                diagnostics["reason"],
                diagnostics,
            )
        except ReviewUpstreamInvalidResponseError:
            outcome, reason_code = "upstream_error", "upstream_invalid_response"
            return _review_error(
                503,
                "jev_review_upstream_invalid_response",
                "Review unavailable",
                request_id,
                reason_code,
            )
        except ReviewUnavailableError:
            outcome, reason_code = "upstream_error", "upstream_failure"
            return _review_error(
                503,
                "jev_review_upstream_failure",
                "Review unavailable",
                request_id,
                reason_code,
            )
        except JevCategoryValidationError as exc:
            logger.warning(
                "category validation_failed rule=%s details=%s",
                exc.rule,
                exc.numeric_details,
            )
            return _review_error(
                503,
                "jev_category_upstream_invalid_response",
                "Category classification unavailable",
                None,
            )
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


_INCONCLUSIVE_REASONS = frozenset(
    {
        "transcript_gap",
        "ambiguous_spans",
        "insufficient_evidence",
        "no_valid_pairs",
        "too_many_boundary_options",
        "choice_inconclusive",
        "neither_complete",
        "ad_content_unconfirmed",
        "invalid_pair",
        "proposed_range_not_confirmed",
        "original_range_not_confirmed",
        "edge_content_unconfirmed",
        "adjacent_message_continues",
        "unrelated_editorial",
        "missing_boundary_coverage",
        "insufficient_boundary_text",
    }
)
_INCONCLUSIVE_STAGES = frozenset(
    {"context", "evidence", "choice_rank", "focused_validation", "boundary_coverage"}
)
_METRIC_REASONS = frozenset(
    {
        "ambiguous_spans",
        "insufficient_evidence",
        "no_valid_pairs",
        "too_many_boundary_options",
        "transcript_gap",
        "choice_inconclusive",
        "neither_complete",
        "ad_content_unconfirmed",
        "malformed_context",
        "missing_boundary_coverage",
        "insufficient_boundary_text",
        "edge_content_unconfirmed",
        "adjacent_message_continues",
        "unrelated_editorial",
    }
)


def _finite_number(value: Any) -> float | int | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    return None


def _context_diagnostics_from_dict(value: dict[str, Any]) -> dict[str, float | int]:
    details: dict[str, float | int] = {}
    for key in ("candidate_start", "candidate_end", "context_start", "context_end"):
        number = _finite_number(value.get(key))
        if number is not None:
            details[key] = number
    return details


def _inconclusive_diagnostics(exc: ReviewInconclusiveError) -> dict[str, Any]:
    return _inconclusive_diagnostics_from_dict(
        {
            key: getattr(exc, key, None)
            for key in (
                "reason",
                "stage",
                "score",
                "threshold",
                "cache_hit",
                "range_start",
                "range_end",
                "start_supported",
                "end_supported",
                "proposal",
                "fallback",
                "candidate_start",
                "candidate_end",
                "context_start",
                "context_end",
            )
        }
    )


def _inconclusive_diagnostics_from_dict(value: dict[str, Any] | None) -> dict[str, Any]:
    if not value:
        return {}
    reason = value.get("reason")
    stage = value.get("stage")
    details: dict[str, Any] = {
        "reason": reason if isinstance(reason, str) and reason in _INCONCLUSIVE_REASONS else "malformed_context",
        "stage": stage if isinstance(stage, str) and stage in _INCONCLUSIVE_STAGES else "context",
    }
    for key in ("score", "threshold", "range_start", "range_end"):
        number = _finite_number(value.get(key))
        if number is not None:
            details[key] = number
    details.update(_context_diagnostics_from_dict(value))
    for key in ("cache_hit", "start_supported", "end_supported"):
        flag = value.get(key)
        if isinstance(flag, bool):
            details[key] = flag
    for key in ("proposal", "fallback"):
        diagnostic = sanitize_review_range_diagnostic(value.get(key))
        if diagnostic is not None:
            details[key] = diagnostic
    return details


def _invalid_request_diagnostics(exc: ReviewInvalidRequestError) -> dict[str, Any]:
    return _invalid_request_diagnostics_from_dict(
        {
            "reason": getattr(exc, "reason", None),
            "candidate_start": getattr(exc, "candidate_start", None),
            "candidate_end": getattr(exc, "candidate_end", None),
            "context_start": getattr(exc, "context_start", None),
            "context_end": getattr(exc, "context_end", None),
        }
    )


def _invalid_request_diagnostics_from_dict(value: dict[str, Any] | None) -> dict[str, Any]:
    if not value:
        return {}
    reason = value.get("reason")
    details: dict[str, Any] = {
        "reason": reason
        if isinstance(reason, str) and reason in {"malformed_context", "invalid_bounds", "outside_context"}
        else "malformed_context",
        "stage": "context",
    }
    details.update(_context_diagnostics_from_dict(value))
    return details


def _inconclusive_message(diagnostics: dict[str, Any]) -> str:
    parts = [f"reason={diagnostics['reason']}", f"stage={diagnostics['stage']}"]
    parts.extend(
        f"{key}={diagnostics[key]}"
        for key in (
            "score", "threshold", "cache_hit", "range_start", "range_end",
            "start_supported", "end_supported",
        )
        if key in diagnostics
    )
    for name in ("proposal", "fallback"):
        diagnostic = sanitize_review_range_diagnostic(diagnostics.get(name))
        if diagnostic is not None:
            parts.append(
                name + "=" + ";".join(f"{key}={value}" for key, value in diagnostic.items())
            )
    return "Review is inconclusive (" + ", ".join(parts) + ")"


def _metric_inconclusive_reason(diagnostics: dict[str, Any]) -> str:
    reason = str(diagnostics["reason"])
    if reason in _METRIC_REASONS:
        return reason
    if diagnostics["stage"] == "context":
        return "malformed_context"
    return "choice_inconclusive"


def _review_error(
    status: int,
    code: str,
    message: str,
    request_id: str | None,
    reason: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> JSONResponse:
    safe_diagnostics = (
        _invalid_request_diagnostics_from_dict(diagnostics)
        if code == "jev_review_invalid_request"
        else _inconclusive_diagnostics_from_dict(diagnostics)
    )
    headers: dict[str, str] = {}
    if status == 422:
        headers["x-should-retry"] = "false"
    if request_id is not None:
        headers["X-Request-ID"] = request_id
        logger.warning(
            "review request_id=%s status=%d error_code=%s reason=%s stage=%s candidate_start=%s candidate_end=%s context_start=%s context_end=%s score=%s threshold=%s cache_hit=%s proposal=%s fallback=%s",
            request_id,
            status,
            code,
            reason or "unknown",
            safe_diagnostics.get("stage", "unknown"),
            safe_diagnostics.get("candidate_start", "unknown"),
            safe_diagnostics.get("candidate_end", "unknown"),
            safe_diagnostics.get("context_start", "unknown"),
            safe_diagnostics.get("context_end", "unknown"),
            safe_diagnostics.get("score", "unknown"),
            safe_diagnostics.get("threshold", "unknown"),
            safe_diagnostics.get("cache_hit", "unknown"),
            safe_diagnostics.get("proposal", "unknown"),
            safe_diagnostics.get("fallback", "unknown"),
        )
    error: dict[str, Any] = {
        "message": message,
        "type": "invalid_request_error" if status == 422 else "api_error",
        "code": code,
    }
    if safe_diagnostics:
        error.update(safe_diagnostics)
    return JSONResponse(
        status_code=status,
        content={"error": error},
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
