"""
Jev proxy endpoints.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.api.openai import _REQUEST_SLOTS, _resolve_api_key
from app.config import settings
from app.services.jev import jev_ask

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


@router.post("/jev/ask")
def ask(
    request: AskRequest, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    """Answer one noul question per segment against the shared transcript."""
    api_key = _resolve_api_key(authorization)

    segments = [seg.model_dump(exclude_none=True) for seg in request.segments]
    max_segments = settings.JEV_MAX_SEGMENTS
    max_chars = settings.JEV_MAX_TRANSCRIPT_CHARS
    if len(segments) > max_segments:
        raise HTTPException(status_code=413, detail="Transcript exceeds configured segment limit")
    if sum(len(str(seg.get("text", ""))) for seg in segments) > max_chars:
        raise HTTPException(status_code=413, detail="Transcript exceeds configured size limit")
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
            enter=settings.JEV_ENTER if request.enter is None else request.enter,
            stay=settings.JEV_STAY if request.stay is None else request.stay,
            max_retries=settings.JEV_MAX_RETRIES,
            cache_max_entries=settings.JEV_CACHE_MAX_ENTRIES,
            request_deadline=settings.JEV_REQUEST_DEADLINE_SECONDS,
            retry_after_max=settings.JEV_RETRY_AFTER_MAX_SECONDS,
        )
    finally:
        _REQUEST_SLOTS.release()
