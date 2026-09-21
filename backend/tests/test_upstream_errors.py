import threading
from email.utils import formatdate

import httpx
import pytest
from app.config import settings


def status_error(
    status: int,
    headers: dict[str, str] | None = None,
    body: dict[str, object] | None = None,
    sentinel: str = "upstream failure",
) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://secret.example/v1/systemone?token=sentinel")
    response = httpx.Response(status, request=request, headers=headers, json=body)
    return httpx.HTTPStatusError(sentinel, request=request, response=response)


@pytest.fixture
def jev_env(monkeypatch):
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "test-key")


@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (status_error(503), 503, "jev_upstream_failure"),
        (status_error(502), 503, "jev_upstream_failure"),
        (status_error(529), 503, "jev_upstream_failure"),
        (status_error(429, {"Retry-After": "7.1"}), 429, "jev_upstream_rate_limited"),
        (status_error(401), 401, "jev_upstream_authentication_error"),
        (status_error(403), 403, "jev_upstream_authentication_error"),
        (status_error(422), 422, "jev_upstream_request_error"),
        (httpx.ReadTimeout("timed out"), 504, "jev_upstream_timeout"),
        (httpx.ConnectError("connection failed"), 503, "jev_upstream_connection_error"),
    ],
)
async def test_chat_maps_upstream_failures(jev_env, client, monkeypatch, caplog, exc, status, code):
    import app.api.openai as api

    monkeypatch.setattr(api, "run_chat_completion", lambda **kwargs: (_ for _ in ()).throw(exc))
    response = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "[0s - 1s] hello"}]},
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert "api.typesafe.ai" not in response.text
    if status == 429:
        assert response.headers["retry-after"] == "8"


async def test_chat_failure_releases_slot(jev_env, client, monkeypatch):
    import app.api.openai as api

    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(api, "_REQUEST_SLOTS", slots)
    monkeypatch.setattr(api, "run_chat_completion", lambda **kwargs: (_ for _ in ()).throw(status_error(503)))
    response = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "[0s - 1s] hello"}]},
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 503
    assert slots.acquire(blocking=False)
    slots.release()


async def test_native_maps_upstream_failures_and_releases_slot(jev_env, client, monkeypatch):
    import app.api.v1.jev as api

    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(api, "_REQUEST_SLOTS", slots)
    monkeypatch.setattr(api, "jev_ask", lambda *args, **kwargs: (_ for _ in ()).throw(status_error(503)))
    response = await client.post(
        "/api/v1/jev/ask",
        json={"segments": [{"sid": 1, "text": "hello"}]},
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "jev_upstream_failure"
    assert slots.acquire(blocking=False)
    slots.release()


async def test_invalid_retry_after_and_upstream_details_are_safe(jev_env, client, monkeypatch, caplog):
    import app.api.openai as api

    exc = status_error(
        529,
        {"Retry-After": "not-a-delay"},
        {"detail": "secret response body"},
        "secret exception message",
    )
    monkeypatch.setattr(api, "run_chat_completion", lambda **kwargs: (_ for _ in ()).throw(exc))
    response = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "[0s - 1s] hello"}]},
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 503
    assert "retry-after" not in response.headers
    assert "secret response body" not in caplog.text
    assert "secret exception message" not in caplog.text
    assert "secret.example" not in caplog.text
    assert "upstream_status=529" in caplog.text
    assert "mapped_status=503" in caplog.text


@pytest.mark.parametrize("header", ["NaN", "Infinity", "-1"])
async def test_invalid_retry_after_headers_are_omitted(jev_env, client, monkeypatch, header):
    import app.api.openai as api

    exc = status_error(503, {"Retry-After": header})
    monkeypatch.setattr(api, "run_chat_completion", lambda **kwargs: (_ for _ in ()).throw(exc))
    response = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "[0s - 1s] hello"}]},
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 503
    assert "retry-after" not in response.headers


async def test_huge_retry_after_body_is_ignored(jev_env, client, monkeypatch):
    import app.api.openai as api

    exc = status_error(503, body={"retry_after_ms": 10**4000})
    monkeypatch.setattr(api, "run_chat_completion", lambda **kwargs: (_ for _ in ()).throw(exc))
    response = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "[0s - 1s] hello"}]},
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 503
    assert "retry-after" not in response.headers


async def test_http_date_retry_after_is_converted_safely(jev_env, client, monkeypatch):
    import app.api.openai as api
    import app.services.jev as service

    monkeypatch.setattr(service.time, "time", lambda: 1000.0)
    exc = status_error(503, {"Retry-After": formatdate(1012, usegmt=True)})
    monkeypatch.setattr(api, "run_chat_completion", lambda **kwargs: (_ for _ in ()).throw(exc))
    response = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "[0s - 1s] hello"}]},
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 503
    assert response.headers["retry-after"] == "12"


async def test_auth_retry_after_is_not_forwarded(jev_env, client, monkeypatch):
    import app.api.openai as api

    exc = status_error(401, {"Retry-After": "7"})
    monkeypatch.setattr(api, "run_chat_completion", lambda **kwargs: (_ for _ in ()).throw(exc))
    response = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "[0s - 1s] hello"}]},
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 401
    assert "retry-after" not in response.headers


async def test_unexpected_internal_error_still_raises(jev_env, client, monkeypatch):
    import app.api.openai as api

    monkeypatch.setattr(api, "run_chat_completion", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("internal")))
    with pytest.raises(RuntimeError, match="internal"):
        await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "[0s - 1s] hello"}]},
            headers={"Authorization": "Bearer test-key"},
        )
