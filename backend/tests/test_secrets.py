"""No secret (bearer token, MinusPod password) ever reaches a log, even at DEBUG."""

import logging
from typing import Any

import httpx
import pytest
from app.config import settings
from app.services import jev, sponsors
from app.utils.redact import redact
from httpx import AsyncClient
from minuspod_compat import format_window_prompt

_TOKEN = "sk-secrettoken-DEADBEEF"
_PASSWORD = "p@ss-SECRET-CAFED00D"

TS_LINES = [
    "[0.0s - 6.0s] Welcome back to the show.",
    "[6.0s - 12.0s] This episode is sponsored by BetterHelp.",
    "[12.0s - 18.0s] Use code SHOW for ten percent off today.",
    "[18.0s - 24.0s] Anyway, back to the conversation.",
]


@pytest.fixture(autouse=True)
def clean_session():
    sponsors.reset_cache()
    yield
    sponsors.reset_cache()


def _fake_fetcher(ad_sids):
    ad_sids = set(ad_sids)

    def fake(payload: dict[str, Any], *, url: str, api_key: str, timeout: float, max_retries: int = 2, **_kwargs: Any) -> dict[str, Any]:
        answers = {}
        for key in payload["questions"]:
            if key.startswith("s"):
                answers[key] = {"noul": 0.98 if int(key[1:]) in ad_sids else 0.02}
            elif question := payload["questions"][key]:
                criteria = question.get("criteria", {})
                category = next(iter(criteria))
                answers[key] = {
                    "choice": category,
                    "confidence": 0.97,
                    "probabilities": {
                        option: 0.97 if option == category else 0.005
                        for option in criteria
                    },
                }
        return {"answers": answers, "usage": {"input_tokens": 1000, "output_tokens": 5}}

    return fake


def test_redact_masks_bearer_and_password():
    assert redact("Authorization: Bearer sk-abc123") == "Authorization: Bearer [REDACTED]"
    assert redact('{"password": "hunter2"}') == '{"password": [REDACTED]}'
    assert redact("password=hunter2 next") == "password=[REDACTED] next"
    assert redact("nothing to redact here") == "nothing to redact here"


def test_http_transport_debug_logging_is_suppressed():
    assert logging.getLogger("httpx").level >= logging.INFO
    assert logging.getLogger("httpcore").level >= logging.INFO


def test_call_payload_never_logs_the_bearer(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answers": {}, "usage": {}})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    with caplog.at_level(logging.DEBUG, logger="app.services.jev"):
        jev.call_payload(
            {"state": {}, "questions": {"s0": {}}},
            url="https://api.typesafe.ai/v1/systemone",
            api_key=_TOKEN,
            client=http_client,
        )
    assert "jev upstream POST" in caplog.text  # the debug line fired
    assert _TOKEN not in caplog.text
    assert "Bearer" not in caplog.text


async def test_detection_round_trip_leaks_no_secret(
    client: AsyncClient, monkeypatch, tmp_path, caplog
):
    monkeypatch.setattr(settings, "JEV_CACHE_PATH", str(tmp_path / "cache.json"))
    monkeypatch.setattr(settings, "MINUSPOD_BASE_URL", "https://mp.test")
    monkeypatch.setattr(settings, "MINUSPOD_PASSWORD", _PASSWORD)
    monkeypatch.setattr(jev, "call_payload", _fake_fetcher({1, 2}))
    # Sponsor path uses the password via login: fake it so nothing hits the network.
    monkeypatch.setattr(sponsors, "login", lambda base_url, password, timeout: {"s": "c"})
    monkeypatch.setattr(
        sponsors,
        "fetch_sponsors",
        lambda url, cookies, timeout: {"sponsors": [{"name": "Zorptech"}]},
    )

    prompt = format_window_prompt("Pod", "Ep", "", TS_LINES, 0, 1, 0.0, 600.0)
    body = {"model": "jev-latest", "messages": [{"role": "user", "content": prompt}]}

    with caplog.at_level(logging.DEBUG):
        resp = await client.post(
            "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {_TOKEN}"}
        )
    assert resp.status_code == 200
    assert "detection:" in caplog.text  # request-path logging is active
    assert _TOKEN not in caplog.text
    assert _PASSWORD not in caplog.text
