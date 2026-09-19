"""Sponsor matching from MinusPod's live sponsor list, with a SEED fallback.

MinusPod clusters ad patterns by sponsor name. Jev never names a sponsor, so
for a real sponsor we look one up from MinusPod's GET /sponsors, and for an
unknown span we emit a unique jev-<7char> placeholder so pattern creation does
not cluster unrelated unnamed spans. The built matcher is cached with a TTL and
refetched when stale; on any error (or an unset URL) it falls back to the
vendored SEED_SPONSORS names. Nothing here raises into the request path.
"""

from __future__ import annotations

import logging
import re
import secrets
import threading
import time
from typing import Any

import httpx
from minuspod_compat import SEED_SPONSORS

from app.config import settings

logger = logging.getLogger(__name__)

# (canonical_name, term) rows, one per name/alias, sorted longest term first.
_Rows = list[tuple[str, str]]

# Short timeout: the sponsor list is a convenience, never worth stalling a call.
_FETCH_TIMEOUT_SECONDS = 5.0
_PLACEHOLDER_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"

_lock = threading.Lock()
_cache: dict[str, Any] = {"rows": None, "built_at": 0.0, "url": None}


def reset_cache() -> None:
    """Drop the cached matcher; used by tests and after a config change."""
    with _lock:
        _cache.update(rows=None, built_at=0.0, url=None)


def fetch_sponsors_json(url: str, *, token: str | None, timeout: float) -> dict[str, Any]:
    """GET the MinusPod sponsor list JSON, with an optional bearer token."""
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(url, headers=headers)
    resp.raise_for_status()
    body: dict[str, Any] = resp.json()
    return body


def _sorted(rows: _Rows) -> _Rows:
    """Longest term first so a specific name wins over a shorter substring one."""
    return sorted(rows, key=lambda r: len(r[1]), reverse=True)


def _rows_from_live(data: Any) -> _Rows:
    """Flatten {"sponsors":[{"name":..,"aliases":[..]}]} into (name, term) rows."""
    sponsors = data.get("sponsors") if isinstance(data, dict) else None
    rows: _Rows = []
    for entry in sponsors or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        for term in [name, *(str(a).strip() for a in entry.get("aliases") or [])]:
            if term:
                rows.append((name, term))
    return _sorted(rows)


def _rows_from_seed() -> _Rows:
    """SEED_SPONSORS names only: the offline fallback matcher."""
    return _sorted([(e["name"], e["name"]) for e in SEED_SPONSORS if e.get("name")])


def _load_rows(url: str | None) -> _Rows:
    """Build the matcher from the live list, else the SEED names."""
    if url:
        try:
            data = fetch_sponsors_json(
                url, token=settings.MINUSPOD_API_TOKEN, timeout=_FETCH_TIMEOUT_SECONDS
            )
            rows = _rows_from_live(data)
            if rows:
                return rows
            logger.warning("sponsors: live list empty; using SEED_SPONSORS")
        except Exception as exc:  # noqa: BLE001 - never break the request path
            logger.warning("sponsors: fetch failed (%s); using SEED_SPONSORS", exc)
    return _rows_from_seed()


def _get_rows() -> _Rows:
    """Cached matcher rows, refetched when the URL changes or the TTL lapses."""
    url = settings.MINUSPOD_SPONSORS_URL
    now = time.monotonic()
    with _lock:
        rows = _cache["rows"]
        if (
            rows is not None
            and _cache["url"] == url
            and now - _cache["built_at"] < settings.SPONSOR_CACHE_TTL_SECONDS
        ):
            return rows
    rows = _load_rows(url)  # fetch outside the lock
    with _lock:
        _cache.update(rows=rows, built_at=now, url=url)
    return rows


def match_sponsor(text: str) -> str | None:
    """Canonical name of the first sponsor term found on a word-ish boundary.

    Case-insensitive; terms are tried longest first so a specific name wins.
    """
    for name, term in _get_rows():
        pattern = rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])"
        if re.search(pattern, text, re.IGNORECASE):
            return name
    return None


def _placeholder() -> str:
    return "jev-" + "".join(secrets.choice(_PLACEHOLDER_ALPHABET) for _ in range(7))


def sponsor_for_span(text: str) -> str:
    """A matched canonical sponsor, else a fresh jev-<7char> placeholder.

    The placeholder is unique per unmatched span so MinusPod's per-sponsor
    pattern clustering never groups unrelated unnamed spans together.
    """
    return match_sponsor(text) or _placeholder()
