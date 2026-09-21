"""
Jev proxy endpoints.
"""

from __future__ import annotations

import math
from typing import Any

import httpx
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.api.openai import _REQUEST_SLOTS, _resolve_api_key, _upstream_error_response
from app.config import settings
from app.services.jev import jev_ask
from app.services.runtime_settings import RuntimeSettingsError, effective_from_settings

router = APIRouter()


class SegmentIn(BaseModel):
    """One transcript segment. start/end are optional timing hints."""

    sid: int
    text: str
    start: float | None = None
    end: float | None = None


class AskRequest(BaseModel):
    segments: list[SegmentIn]
    model: str | None = None
    uid: str | None = None
    enter: float | None = None
    stay: float | None = None


@router.post("/jev/ask", response_model=None)
def ask(
    request: AskRequest, authorization: str | None = Header(default=None)
) -> dict[str, Any] | JSONResponse:
    """Answer one noul question per segment against the shared transcript."""
    api_key = _resolve_api_key(authorization)

    try:
        thresholds = effective_from_settings(settings)
    except RuntimeSettingsError as exc:
        raise HTTPException(status_code=503, detail="Runtime settings unavailable") from exc
    enter = thresholds.detection_enter if request.enter is None else request.enter
    stay = thresholds.detection_stay if request.stay is None else request.stay
    if (
        not math.isfinite(enter)
        or not math.isfinite(stay)
        or not 0.0 <= enter <= 1.0
        or not 0.0 <= stay <= 1.0
        or stay > enter
    ):
        raise HTTPException(status_code=422, detail="enter and stay must be finite values in [0, 1] with stay <= enter")

    segments = [seg.model_dump(exclude_none=True) for seg in request.segments]
    if not _REQUEST_SLOTS.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Proxy is at capacity")
    try:
        return jev_ask(
            segments,
            url=settings.TYPESAFE_API_URL,
            api_key=api_key,
            timeout=settings.JEV_TIMEOUT_SECONDS,
            cache_path=settings.JEV_CACHE_PATH,
            model=request.model or settings.JEV_MODEL,
            uid=request.uid,
            enter=enter,
            stay=stay,
            max_retries=settings.JEV_MAX_RETRIES,
            cache_max_entries=settings.JEV_CACHE_MAX_ENTRIES,
            request_deadline=settings.JEV_REQUEST_DEADLINE_SECONDS,
            retry_after_max=settings.JEV_RETRY_AFTER_MAX_SECONDS,
        )
    except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.TransportError) as exc:
        return _upstream_error_response(exc)
    finally:
        _REQUEST_SLOTS.release()
