"""
Jev proxy endpoints.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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
def ask(request: AskRequest) -> dict[str, Any]:
    """Answer one noul question per segment against the shared transcript."""
    try:
        api_key = settings.require_jev_api_key()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    segments = [seg.model_dump(exclude_none=True) for seg in request.segments]
    return jev_ask(
        segments,
        url=settings.TYPESAFE_API_URL,
        api_key=api_key,
        timeout=settings.JEV_TIMEOUT_SECONDS,
        cache_path=settings.JEV_CACHE_PATH,
        model=request.model or settings.JEV_MODEL,
        uid=request.uid,
        enter=request.enter,
        stay=request.stay,
    )
