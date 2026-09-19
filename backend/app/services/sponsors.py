"""Sponsor matching from MinusPod's live sponsor list, with a SEED fallback.

MinusPod clusters ad patterns by sponsor name. Jev never names a sponsor, so
for a real sponsor we look one up from MinusPod's GET /api/v1/sponsors, and for
an unknown span we emit a unique jev-<7char> placeholder so pattern creation
does not cluster unrelated unnamed spans.

MinusPod requires a password login that returns session cookies (there is no
static token). We log in once, cache the cookies in-memory, reuse them for the
sponsors GET, and re-login when the session is stale or a GET returns 401. Login
is serialized behind a lock so a burst of requests never fans out into multiple
simultaneous logins (the login route is rate limited 3/min, 10/hour). The built
matcher is cached with a TTL and refetched when stale; on any error (unset
config, login failure, rate limit, network error, malformed JSON) it falls back
to the vendored SEED_SPONSORS names. Nothing here raises into the request path.
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

_LOGIN_PATH = "/api/v1/auth/login"
_SPONSORS_PATH = "/api/v1/sponsors"

# Short timeout: the sponsor list is a convenience, never worth stalling a call.
_FETCH_TIMEOUT_SECONDS = 5.0
# Login is a POST that can be slower; mirrors the benchmark reference.
_LOGIN_TIMEOUT_SECONDS = 15.0
_PLACEHOLDER_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"

_lock = threading.Lock()
_cache: dict[str, Any] = {"rows": None, "built_at": 0.0, "url": None}

_session_lock = threading.Lock()
_session: dict[str, Any] = {"cookies": None, "created_at": 0.0}


class SponsorAuthError(Exception):
    """Raised when /sponsors returns 401 so the caller can re-login once."""


def reset_cache() -> None:
    """Drop the cached matcher and session; used by tests and after a config change."""
    with _lock:
        _cache.update(rows=None, built_at=0.0, url=None)
    with _session_lock:
        _session.update(cookies=None, created_at=0.0)


def has_active_session() -> bool:
    """Whether a login session is cached and still within its TTL.

    Read-only: never logs in (login is rate limited). Used by the status endpoint.
    """
    with _session_lock:
        cookies: dict[str, str] | None = _session["cookies"]
        if cookies is None:
            return False
        age = time.monotonic() - _session["created_at"]
        return bool(age < settings.MINUSPOD_SESSION_TTL_SECONDS)


def login(base_url: str, password: str, timeout: float) -> dict[str, str]:
    """POST /api/v1/auth/login and return the session cookies. Raises on non-200."""
    url = base_url.rstrip("/") + _LOGIN_PATH
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json={"password": password})
    # Rate limited (3/min, 10/hour); surface as an error, never auto-retry.
    if resp.status_code == 429:
        raise RuntimeError("login rate-limited (HTTP 429)")
    if resp.status_code != 200:
        raise RuntimeError(f"login failed: HTTP {resp.status_code}")
    cookies = dict(resp.cookies)
    if not cookies:
        raise RuntimeError("login returned 200 but set no cookies")
    return cookies


def fetch_sponsors(url: str, cookies: dict[str, str], timeout: float) -> dict[str, Any]:
    """GET /api/v1/sponsors with session cookies. Raises SponsorAuthError on 401."""
    headers = {"Accept": "application/json"}
    with httpx.Client(timeout=timeout, cookies=cookies) as client:
        resp = client.get(url, headers=headers)
    if resp.status_code == 401:
        raise SponsorAuthError("sponsors returned 401")
    resp.raise_for_status()
    body: dict[str, Any] = resp.json()
    return body


def _acquire_session(
    base_url: str, password: str, *, stale: dict[str, str] | None = None
) -> dict[str, str]:
    """Return live session cookies, logging in under a lock when needed.

    stale: cookies just rejected with 401; force a re-login unless another thread
    already replaced them. The lock serializes logins so a request burst triggers
    at most one login (rate-limit protection); we re-check inside the lock so a
    thread that waited reuses the session a peer just created.
    """
    ttl = settings.MINUSPOD_SESSION_TTL_SECONDS
    if stale is None:
        with _session_lock:
            cookies: dict[str, str] | None = _session["cookies"]
            if cookies is not None and time.monotonic() - _session["created_at"] < ttl:
                return cookies
    with _session_lock:
        cookies = _session["cookies"]
        fresh = cookies is not None and time.monotonic() - _session["created_at"] < ttl
        if stale is None and fresh:
            return cookies  # a peer logged in while we waited
        if stale is not None and cookies is not None and cookies is not stale:
            return cookies  # a peer already replaced the rejected session
        new_cookies = login(base_url, password, _LOGIN_TIMEOUT_SECONDS)
        _session.update(cookies=new_cookies, created_at=time.monotonic())
        return new_cookies


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


def _load_rows(base_url: str, password: str) -> _Rows:
    """Build the matcher from the live list, else the SEED names."""
    sponsors_url = base_url.rstrip("/") + _SPONSORS_PATH
    try:
        cookies = _acquire_session(base_url, password)
    except Exception as exc:  # noqa: BLE001 - never break the request path
        logger.warning("sponsors: login failed (%s); using SEED_SPONSORS", exc)
        return _rows_from_seed()
    try:
        data = fetch_sponsors(sponsors_url, cookies, _FETCH_TIMEOUT_SECONDS)
    except SponsorAuthError:
        # Session rejected: re-login once, then retry the GET a single time.
        try:
            cookies = _acquire_session(base_url, password, stale=cookies)
            data = fetch_sponsors(sponsors_url, cookies, _FETCH_TIMEOUT_SECONDS)
        except Exception as exc:  # noqa: BLE001
            logger.warning("sponsors: re-login/refetch failed (%s); using SEED_SPONSORS", exc)
            return _rows_from_seed()
    except Exception as exc:  # noqa: BLE001
        logger.warning("sponsors: fetch failed (%s); using SEED_SPONSORS", exc)
        return _rows_from_seed()
    rows = _rows_from_live(data)
    if rows:
        return rows
    logger.warning("sponsors: live list empty; using SEED_SPONSORS")
    return _rows_from_seed()


def _get_rows() -> _Rows:
    """Cached matcher rows, refetched when base_url changes or the TTL lapses."""
    base_url = settings.MINUSPOD_BASE_URL
    password = settings.MINUSPOD_PASSWORD
    now = time.monotonic()
    with _lock:
        rows: list[tuple[str, str]] | None = _cache["rows"]
        if (
            rows is not None
            and _cache["url"] == base_url
            and now - _cache["built_at"] < settings.SPONSOR_CACHE_TTL_SECONDS
        ):
            return rows
    if not base_url or not password:
        rows = _rows_from_seed()  # unconfigured -> offline gazetteer
    else:
        rows = _load_rows(base_url, password)  # fetch outside the lock
    with _lock:
        _cache.update(rows=rows, built_at=now, url=base_url)
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
