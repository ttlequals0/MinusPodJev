"""Tests for the Jev proxy: payload building, parsing, caching, endpoint."""

from typing import Any

import httpx
import pytest
from app.config import settings
from app.services.jev import (
    build_payload,
    call_payload,
    estimate_cost_usd,
    jev_ask,
    parse_response,
)

ANSWER_BODY: dict[str, Any] = {
    "answers": {"s1": {"noul": 0.98}, "junk": {"noul": 0.5}},
    "usage": {"input_tokens": 1000, "output_tokens": 5},
}


@pytest.fixture
def jev_env(tmp_path, monkeypatch):
    """Point settings at a test key and a throwaway cache file."""
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(settings, "JEV_CACHE_PATH", str(tmp_path / "cache.json"))


def test_build_payload_shape():
    payload = build_payload([{"sid": 1, "text": " hello "}])
    assert "model" not in payload
    assert payload["state"]["transcript"] == "L0001| hello"
    question = payload["questions"]["s1"]
    assert question["type"] == "noul"
    assert "L0001" in question["instructions"]


def test_parse_response_keeps_noul_answers_and_usage():
    entry = parse_response(ANSWER_BODY)
    assert entry["probabilities"] == {"s1": 0.98}
    assert entry["input_tokens"] == 1000
    assert entry["output_tokens"] == 5


def test_estimate_cost_bills_input_only():
    assert estimate_cost_usd(1_000_000) == 0.042


def test_call_payload_sends_bearer_header():
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["Authorization"]
        return httpx.Response(200, json=ANSWER_BODY)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    body = call_payload({"state": {}}, url="https://example.test/v1", api_key="k", client=client)
    assert seen["auth"] == "Bearer k"
    assert body == ANSWER_BODY


def test_jev_ask_caches_by_payload_hash(jev_env, tmp_path):
    calls: list[int] = []

    def fake(payload, *, url, api_key, timeout):
        calls.append(1)
        return ANSWER_BODY

    segments = [{"sid": 1, "text": "hello"}]
    cache = str(tmp_path / "jev_ask.json")
    first = jev_ask(
        segments, url="u", api_key="k", timeout=1.0, cache_path=cache, model="m", fetcher=fake
    )
    second = jev_ask(
        segments, url="u", api_key="k", timeout=1.0, cache_path=cache, model="m", fetcher=fake
    )
    assert len(calls) == 1
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["probabilities"] == {"s1": 0.98}
    assert second["estimated_cost_usd"] == pytest.approx(0.000042)
    assert second["spans"] == [{"start_id": 1, "end_id": 1, "confidence": 0.98}]


async def test_ask_endpoint_returns_probabilities(jev_env, client, monkeypatch):
    calls: list[int] = []

    def fake(payload, *, url, api_key, timeout):
        calls.append(1)
        return ANSWER_BODY

    import app.services.jev as service_module

    monkeypatch.setattr(service_module, "call_payload", fake)

    resp = await client.post("/api/v1/jev/ask", json={"segments": [{"sid": 1, "text": "hi"}]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["probabilities"] == {"s1": 0.98}
    assert data["usage"] == {"input_tokens": 1000, "output_tokens": 5}
    assert data["cache_hit"] is False

    resp2 = await client.post("/api/v1/jev/ask", json={"segments": [{"sid": 1, "text": "hi"}]})
    assert resp2.json()["cache_hit"] is True
    assert len(calls) == 1


async def test_ask_endpoint_requires_api_key(jev_env, client, monkeypatch):
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", None)
    resp = await client.post("/api/v1/jev/ask", json={"segments": [{"sid": 1, "text": "hi"}]})
    assert resp.status_code == 503
    assert "TYPESAFE_API_KEY" in resp.json()["detail"]
