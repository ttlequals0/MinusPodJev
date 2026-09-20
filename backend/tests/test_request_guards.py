"""Request authentication and capacity guards."""

import threading

import pytest
from app.api import openai
from app.config import settings
from app.main import app
from httpx import ASGITransport, AsyncClient


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


async def test_chat_requires_bearer_by_default(client, monkeypatch):
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "server-key")
    monkeypatch.setattr(settings, "JEV_ALLOW_UNAUTHENTICATED_FALLBACK", False)
    response = await client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 401


async def test_chat_fallback_is_explicit_opt_in(client, monkeypatch):
    monkeypatch.setattr(settings, "JEV_ALLOW_UNAUTHENTICATED_FALLBACK", True)
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "server-key")
    monkeypatch.setattr(openai, "run_chat_completion", lambda **kwargs: kwargs)
    response = await client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 200
    assert response.json()["api_key"] == "server-key"


async def test_capacity_slot_released_after_error(client, monkeypatch):
    calls = 0

    def fail_once(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("boom")
        return {"ok": True}

    monkeypatch.setattr(openai, "run_chat_completion", fail_once)
    monkeypatch.setattr(openai, "_REQUEST_SLOTS", threading.BoundedSemaphore(1))
    with pytest.raises(RuntimeError):
        openai.chat_completions(
            openai.ChatCompletionRequest(messages=[]), "Bearer caller-key"
        )
    assert openai.chat_completions(
        openai.ChatCompletionRequest(messages=[]), "Bearer caller-key"
    ) == {"ok": True}
