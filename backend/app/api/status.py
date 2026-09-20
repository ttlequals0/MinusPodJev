"""Upstream connectivity status for MinusPod and TypeSafe Jev.

GET /api/status reports whether the two upstreams are reachable WITHOUT spending
money or tripping rate limits: it never sends a real inference, never needs a
key, and never logs in. Each probe is a cheap HEAD with a short timeout; any HTTP
response (200/401/404/405/...) counts as reachable, only a connect/DNS/timeout
error is unreachable. The response is host-only and carries no key, password,
cookie, or credentialed URL. Nothing here raises.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter

from app.config import settings
from app.services import sponsors
from app.utils.metrics import metrics

router = APIRouter()

# Short so the endpoint can never hang on a dead upstream.
_PROBE_TIMEOUT_SECONDS = 3.0


@router.get("/stats")
def stats() -> dict[str, Any]:
    """Return ephemeral metrics for this process only."""
    return metrics.snapshot()


def _host_only(url: str | None) -> str:
    """Netloc only: no scheme, credentials, or path, so no secret can leak."""
    if not url:
        return ""
    parts = urlsplit(url if "//" in url else "//" + url)
    netloc = parts.netloc or parts.path
    return netloc.rsplit("@", 1)[-1]  # drop any user:pass@ credentials


def _probe(url: str, timeout: float = _PROBE_TIMEOUT_SECONDS) -> tuple[bool, str]:
    """Cheap reachability check. Any HTTP response = reachable; only a transport
    error (connect/DNS/timeout) = unreachable. Never raises; the reason is an
    exception type name, never a secret."""
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            client.head(url)
        return True, ""
    except Exception as exc:  # noqa: BLE001 - status must never raise
        return False, type(exc).__name__


@router.get("/status")
def status_check() -> dict[str, Any]:
    """Report upstream reachability without spending money or logging in."""
    jev_url = settings.TYPESAFE_API_URL
    jev_reachable, jev_reason = _probe(jev_url)
    jev: dict[str, Any] = {
        "configured": bool(jev_url),
        "reachable": jev_reachable,
        "url": _host_only(jev_url),
    }
    if not jev_reachable:
        jev["reason"] = jev_reason

    mp_url = settings.MINUSPOD_BASE_URL
    mp_configured = bool(mp_url and settings.MINUSPOD_PASSWORD)
    if mp_url:
        mp_reachable, mp_reason = _probe(mp_url)
    else:
        mp_reachable, mp_reason = False, "not configured"
    minuspod: dict[str, Any] = {
        "configured": mp_configured,
        "reachable": mp_reachable,
        "authenticated": sponsors.has_active_session(),
        "url": _host_only(mp_url),
    }
    if not mp_reachable:
        minuspod["reason"] = mp_reason

    # Degraded only when a configured upstream cannot be reached.
    degraded = (bool(jev_url) and not jev_reachable) or (mp_configured and not mp_reachable)
    return {
        "status": "degraded" if degraded else "ok",
        "jev": jev,
        "minuspod": minuspod,
        "review": {
            "refine_boundaries": settings.JEV_REVIEW_REFINE_BOUNDARIES,
            "model": settings.JEV_MODEL,
            "evidence_threshold": settings.JEV_ENTER,
            "choice_threshold": settings.JEV_ENTER,
        },
    }
