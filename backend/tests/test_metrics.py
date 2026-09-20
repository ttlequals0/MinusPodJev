"""Ephemeral runtime metric regressions."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import pytest
from app.services.jev import call_payload, jev_ask
from app.utils.metrics import metrics

ANSWER: dict[str, Any] = {
    "answers": {"s1": {"noul": 0.98}},
    "usage": {"input_tokens": 1000, "output_tokens": 5},
}


@pytest.fixture(autouse=True)
def reset_metrics():
    metrics.reset()


def test_jev_attempts_include_retry_and_charge_success(monkeypatch):
    statuses = [429, 200]

    def handler(_: httpx.Request) -> httpx.Response:
        status = statuses.pop(0)
        return httpx.Response(status, json=ANSWER if status == 200 else {})

    monkeypatch.setattr("app.services.jev.time.sleep", lambda _delay: None)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert call_payload({"state": {}}, url="https://x.test", api_key="key", client=client) == ANSWER
    snapshot = metrics.snapshot()
    assert snapshot["jev_http"]["attempts"] == 2
    assert snapshot["jev_http"]["success"] == 1
    assert snapshot["jev_http"]["failure"] == 1
    assert snapshot["jev_http"]["unknown_usage"] == 0
    assert snapshot["cost"]["estimated_input_usd"] == pytest.approx(0.000042)


def test_successful_response_without_usage_is_not_charged():
    body = {"answers": {"s1": {"noul": 0.98}}}
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
    )
    assert call_payload({"state": {}}, url="https://x.test", api_key="key", client=client) == body
    snapshot = metrics.snapshot()
    assert snapshot["jev_http"]["unknown_usage"] == 1
    assert snapshot["cost"]["estimated_input_usd"] == 0.0


def test_cache_hits_are_counted_without_charging_fetcher_usage(tmp_path):
    calls = 0

    def fetcher(_payload, **_kwargs):
        nonlocal calls
        calls += 1
        return ANSWER

    kwargs = {
        "url": "u",
        "api_key": "k",
        "timeout": 1.0,
        "cache_path": str(tmp_path / "cache.json"),
        "model": "m",
        "fetcher": fetcher,
    }
    jev_ask([{"sid": 1, "text": "sponsor"}], **kwargs)
    jev_ask([{"sid": 1, "text": "sponsor"}], **kwargs)
    snapshot = metrics.snapshot()
    assert calls == 1
    assert snapshot["cache"] == {"hits": 1, "misses": 1}
    assert snapshot["cost"]["estimated_input_usd"] == 0.0


def test_failed_cache_miss_is_counted(tmp_path):
    def fetcher(_payload, **_kwargs):
        raise RuntimeError("upstream failed")

    with pytest.raises(RuntimeError, match="upstream failed"):
        jev_ask(
            [{"sid": 1, "text": "sponsor"}],
            url="u",
            api_key="k",
            timeout=1.0,
            cache_path=str(tmp_path / "cache.json"),
            model="m",
            fetcher=fetcher,
        )
    assert metrics.snapshot()["cache"] == {"hits": 0, "misses": 1}


def test_metrics_records_are_thread_safe():
    def record(_: int) -> None:
        metrics.record_proxy_request(1.0, True)
        metrics.record_cache(True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(record, range(100)))
    snapshot = metrics.snapshot()
    assert snapshot["proxy_requests"]["count"] == 100
    assert snapshot["cache"]["hits"] == 100


async def test_stats_schema_and_inference_path_scope(client):
    baseline = (await client.get("/api/stats")).json()
    assert baseline["scope"]["kind"] == "process"
    assert baseline["reset_on_restart"] is True
    assert baseline["proxy_requests"]["count"] == 0
    assert (await client.get("/api/health")).status_code == 200
    response = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "no transcript lines"}]},
        headers={"Authorization": "Bearer key"},
    )
    assert response.status_code == 200
    snapshot = (await client.get("/api/stats")).json()
    assert snapshot["proxy_requests"]["count"] == 1
    assert snapshot["proxy_requests"]["success"] == 1
    assert set(snapshot["jev_http"]) == {
        "attempts",
        "success",
        "failure",
        "unknown_usage",
        "latency_ms",
    }
    assert set(snapshot["cache"]) == {"hits", "misses"}
    assert set(snapshot["cost"]) == {"estimated_input_usd"}
    assert set(snapshot["proxy_requests"]["latency_ms"]) == {
        "count",
        "sum",
        "min",
        "max",
        "average",
    }
