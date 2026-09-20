"""Tests for GET /api/status (upstream probes mocked; no real network)."""

from typing import Any

import pytest
from app.api import status
from app.config import settings
from app.services import sponsors
from httpx import AsyncClient


@pytest.fixture(autouse=True)
def clean_session():
    """No cached MinusPod session leaking between tests."""
    sponsors.reset_cache()
    yield
    sponsors.reset_cache()


def _probe_all_reachable(url: str, timeout: float = 3.0) -> tuple[bool, str]:
    return True, ""


def _probe_jev_down(url: str, timeout: float = 3.0) -> tuple[bool, str]:
    # Only the TypeSafe upstream is unreachable.
    if "typesafe" in url:
        return False, "ConnectError"
    return True, ""


def test_host_only_strips_scheme_path_and_creds():
    assert status._host_only("https://api.typesafe.ai/v1/systemone") == "api.typesafe.ai"
    assert status._host_only("https://user:pw@mp.test/api/v1") == "mp.test"
    assert status._host_only("mp.test:8443") == "mp.test:8443"
    assert status._host_only(None) == ""


async def test_status_both_configured_and_reachable(client: AsyncClient, monkeypatch):
    monkeypatch.setattr(settings, "MINUSPOD_BASE_URL", "https://mp.test")
    monkeypatch.setattr(settings, "MINUSPOD_PASSWORD", "pw")
    monkeypatch.setattr(status, "_probe", _probe_all_reachable)
    monkeypatch.setattr(sponsors, "has_active_session", lambda: True)

    resp = await client.get("/api/status")
    assert resp.status_code == 200
    data: dict[str, Any] = resp.json()

    assert data["status"] == "ok"
    assert data["jev"] == {"configured": True, "reachable": True, "url": "api.typesafe.ai"}
    assert data["minuspod"] == {
        "configured": True,
        "reachable": True,
        "authenticated": True,
        "url": "mp.test",
    }
    # host only: never a scheme, path, or credentials
    assert "://" not in data["jev"]["url"] and "/" not in data["jev"]["url"]


@pytest.mark.parametrize("refine_boundaries", [False, True])
async def test_status_reports_effective_review_settings(client: AsyncClient, monkeypatch, refine_boundaries):
    monkeypatch.setattr(settings, "JEV_REVIEW_REFINE_BOUNDARIES", refine_boundaries)
    monkeypatch.setattr(settings, "JEV_MODEL", "jev-test-model")
    monkeypatch.setattr(settings, "JEV_ENTER", 0.9137)
    monkeypatch.setattr(status, "_probe", _probe_all_reachable)
    monkeypatch.setattr(sponsors, "has_active_session", lambda: False)

    response = await client.get("/api/status")

    assert response.json()["review"] == {
        "refine_boundaries": refine_boundaries,
        "model": "jev-test-model",
        "evidence_threshold": 0.9137,
        "choice_threshold": 0.9137,
    }


async def test_status_minuspod_unconfigured(client: AsyncClient, monkeypatch):
    monkeypatch.setattr(settings, "MINUSPOD_BASE_URL", None)
    monkeypatch.setattr(settings, "MINUSPOD_PASSWORD", None)
    monkeypatch.setattr(status, "_probe", _probe_all_reachable)
    monkeypatch.setattr(sponsors, "has_active_session", lambda: False)

    resp = await client.get("/api/status")
    data = resp.json()

    assert data["status"] == "ok"  # unconfigured upstream never degrades
    assert data["jev"]["reachable"] is True
    assert data["minuspod"]["configured"] is False
    assert data["minuspod"]["reachable"] is False
    assert data["minuspod"]["authenticated"] is False
    assert data["minuspod"]["url"] == ""
    assert data["minuspod"]["reason"] == "not configured"


async def test_status_degraded_when_upstream_unreachable(client: AsyncClient, monkeypatch):
    monkeypatch.setattr(settings, "MINUSPOD_BASE_URL", "https://mp.test")
    monkeypatch.setattr(settings, "MINUSPOD_PASSWORD", "pw")
    monkeypatch.setattr(status, "_probe", _probe_jev_down)
    monkeypatch.setattr(sponsors, "has_active_session", lambda: False)

    resp = await client.get("/api/status")
    data = resp.json()

    assert data["status"] == "degraded"
    assert data["jev"]["reachable"] is False
    assert data["jev"]["reason"] == "ConnectError"
    assert data["minuspod"]["reachable"] is True


def test_has_active_session_no_login(monkeypatch):
    """The helper reports the cached session and never logs in."""
    called: list[int] = []
    monkeypatch.setattr(sponsors, "login", lambda *a, **k: called.append(1))
    sponsors.reset_cache()
    assert sponsors.has_active_session() is False
    assert called == []  # never triggers a login
