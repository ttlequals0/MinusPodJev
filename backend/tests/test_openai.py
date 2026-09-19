"""Tests for the OpenAI-compatible chat-completions adapter (no real network)."""

import json
from typing import Any

import pytest
from app.config import settings
from app.services.openai_adapter import extract_user_text, match_sponsor, parse_transcript
from minuspod_compat import (
    AD_DETECTION_JSON_SCHEMA,
    format_window_prompt,
    parse_ads_from_response,
    parse_id_ads_from_response,
    resolve_segment_id_ads,
)

_SCHEMA_AD_KEYS = set(
    AD_DETECTION_JSON_SCHEMA["properties"]["ads"]["items"]["properties"]
)


@pytest.fixture
def jev_env(tmp_path, monkeypatch):
    """Test key plus a throwaway cache file per test."""
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(settings, "JEV_CACHE_PATH", str(tmp_path / "cache.json"))


def make_fake(ad_sids, *, cat_index=0, input_tokens=1000, output_tokens=5):
    """Fetcher that scores s<sid> detection and c<index> category questions."""
    ad_sids = set(ad_sids)

    def fake(payload: dict[str, Any], *, url: str, api_key: str, timeout: float) -> dict[str, Any]:
        answers: dict[str, Any] = {}
        for key in payload["questions"]:
            if key.startswith("s"):
                answers[key] = {"noul": 0.98 if int(key[1:]) in ad_sids else 0.02}
            elif key.startswith("c"):
                answers[key] = {"noul": 0.97 if int(key[1:]) == cat_index else 0.05}
        return {
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }

    return fake


TS_LINES = [
    "[0.0s - 6.0s] Welcome back to the show.",
    "[6.0s - 12.0s] Today we talk about hiking trips.",
    "[12.0s - 18.0s] It was a great trip in the mountains.",
    "[18.0s - 24.0s] This episode is sponsored by BetterHelp.",
    "[24.0s - 30.0s] BetterHelp offers online therapy at betterhelp.com slash show.",
    "[30.0s - 36.0s] Use code SHOW for ten percent off your first month today.",
    "[36.0s - 42.0s] Anyway, back to the trip we were on.",
    "[42.0s - 48.0s] We hiked for hours and it was tiring.",
]


# ---------------- prompt parser unit tests ----------------


def test_parse_transcript_timestamps_mode():
    text = format_window_prompt("Pod", "Ep", "", TS_LINES, 0, 1, 0.0, 600.0)
    segments, mode = parse_transcript(text)
    assert mode == "timestamps"
    assert len(segments) == 8
    assert segments[0] == {"start": 0.0, "end": 6.0, "text": "Welcome back to the show.", "sid": 0}
    assert segments[3]["sid"] == 3
    assert segments[3]["text"] == "This episode is sponsored by BetterHelp."


def test_parse_transcript_segment_ids_mode():
    lines = ["[10] first line", "[11] second line", "[12] third line"]
    text = format_window_prompt(
        "Pod", "Ep", "", lines, 0, 1, 0.0, 600.0, addressing_mode="segment_ids"
    )
    segments, mode = parse_transcript(text)
    assert mode == "segment_ids"
    assert [s["sid"] for s in segments] == [10, 11, 12]
    assert segments[0] == {"sid": 10, "text": "first line"}
    assert all("start" not in s for s in segments)


def test_parse_transcript_junk_returns_empty():
    text = "Podcast: Foo\nEpisode: Bar\nSome prose with no bracketed lines at all."
    segments, mode = parse_transcript(text)
    assert segments == []
    assert mode == "empty"


def test_extract_user_text_last_user_message():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "the transcript"},
    ]
    assert extract_user_text(messages) == "the transcript"


# ---------------- sponsor gazetteer ----------------


def test_match_sponsor_hits_seed_and_alias():
    assert match_sponsor("go to BetterHelp for therapy") == "BetterHelp"
    assert match_sponsor("try Better Help today") == "BetterHelp"  # alias
    assert match_sponsor("AG1 greens powder") == "Athletic Greens"  # alias -> canonical
    assert match_sponsor("just a normal conversation about trees") is None


def test_match_sponsor_respects_word_boundary():
    # "Ring" is a seed; a substring inside another word must not match.
    assert match_sponsor("the bell was ringing loudly") is None
    assert match_sponsor("install a Ring doorbell") == "Ring"


# ---------------- endpoint round trips ----------------


async def test_chat_completions_timestamps_round_trip(jev_env, client, monkeypatch):
    import app.services.jev as jev

    monkeypatch.setattr(jev, "call_payload", make_fake({3, 4, 5}))

    prompt = format_window_prompt("My Podcast", "Ep 1", "", TS_LINES, 0, 1, 0.0, 600.0)
    body = {
        "model": "jev-latest",
        "messages": [
            {"role": "system", "content": "detect ads"},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    resp = await client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    data = resp.json()

    assert data["object"] == "chat.completion"
    assert data["model"] == "jev-latest"
    assert isinstance(data["created"], int)
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert set(data["usage"]) == {"prompt_tokens", "completion_tokens", "total_tokens"}

    content = data["choices"][0]["message"]["content"]
    assert isinstance(content, str)
    # every emitted ad key is part of the detection schema
    for raw_ad in json.loads(content)["ads"]:
        assert set(raw_ad) <= _SCHEMA_AD_KEYS, set(raw_ad) - _SCHEMA_AD_KEYS
    ads = parse_ads_from_response(content)
    assert len(ads) == 1
    ad = ads[0]
    assert ad["start"] == 18.0
    assert ad["end"] == 36.0
    assert ad["category"] == "sponsor"
    assert ad["sponsor"] == "BetterHelp"
    assert ad["end_text"].startswith("Use code SHOW")
    assert ad["reason"] == "jev ad: 3/3 segments >= enter"


async def test_chat_completions_no_ads(jev_env, client, monkeypatch):
    import app.services.jev as jev

    monkeypatch.setattr(jev, "call_payload", make_fake(set()))

    prompt = format_window_prompt("My Podcast", "Ep 1", "", TS_LINES, 0, 1, 0.0, 600.0)
    body = {"model": "jev-latest", "messages": [{"role": "user", "content": prompt}]}
    resp = await client.post("/chat/completions", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    content = json.loads(data["choices"][0]["message"]["content"])
    assert content == {"ads": []}
    assert parse_ads_from_response(data["choices"][0]["message"]["content"]) == []


async def test_chat_completions_empty_prompt(jev_env, client):
    body = {"model": "jev-latest", "messages": [{"role": "user", "content": "no lines here"}]}
    resp = await client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert json.loads(data["choices"][0]["message"]["content"]) == {"ads": []}
    assert data["usage"]["total_tokens"] == 0


async def test_chat_completions_segment_ids_round_trip(jev_env, client, monkeypatch):
    import app.services.jev as jev

    monkeypatch.setattr(jev, "call_payload", make_fake({12, 13}))

    lines = [
        "[10] Welcome back to the show everyone.",
        "[11] We were discussing the mountains.",
        "[12] This segment is brought to you by Squarespace.",
        "[13] Build your website at squarespace.com slash pod.",
        "[14] Now back to our regular conversation.",
    ]
    prompt = format_window_prompt(
        "My Podcast", "Ep 2", "", lines, 0, 1, 0.0, 600.0, addressing_mode="segment_ids"
    )
    body = {"model": "jev-latest", "messages": [{"role": "user", "content": prompt}]}
    resp = await client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    ad = parsed["ads"][0]
    assert ad["start_id"] == 12
    assert ad["end_id"] == 13
    assert "start" not in ad and "end" not in ad
    assert isinstance(ad["start_id"], int)

    # MinusPod's id-mode parser + resolver round-trip against timed segments.
    id_ads, used_ids = parse_id_ads_from_response(content)
    assert used_ids is True
    assert id_ads[0]["start_id"] == 12 and id_ads[0]["end_id"] == 13
    window = [
        {"sid": 12, "start": 120.0, "end": 126.0},
        {"sid": 13, "start": 126.0, "end": 132.0},
    ]
    resolved = resolve_segment_id_ads(id_ads, window)
    assert len(resolved) == 1
    assert resolved[0]["start"] == 120.0 and resolved[0]["end"] == 132.0
    assert resolved[0]["sponsor"] == "Squarespace"


async def test_chat_completions_requires_api_key(jev_env, client, monkeypatch):
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", None)
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "[0.0s - 1.0s] hi"}]},
    )
    assert resp.status_code == 503
    assert "TYPESAFE_API_KEY" in resp.json()["detail"]


async def test_models_endpoints(client):
    for path in ("/models", "/v1/models"):
        resp = await client.get(path)
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert data["data"][0]["id"] == settings.JEV_PUBLIC_MODEL == "typesafe/jev"
        assert data["data"][0]["object"] == "model"


def test_category_pass_off_uses_default_category(jev_env, tmp_path, monkeypatch):
    import app.services.jev as jev
    from app.services.openai_adapter import run_chat_completion

    monkeypatch.setattr(jev, "call_payload", make_fake({3, 4, 5}))
    prompt = format_window_prompt("Pod", "Ep", "", TS_LINES, 0, 1, 0.0, 600.0)
    resp = run_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        request_model="jev-latest",
        url="u",
        api_key="k",
        timeout=1.0,
        cache_path=str(tmp_path / "c.json"),
        model="jev-latest",
        enter=0.95,
        stay=0.40,
        category_pass=False,
        category_context=2,
        default_category="sponsor",
    )
    ad = json.loads(resp["choices"][0]["message"]["content"])["ads"][0]
    assert ad["category"] == "sponsor"
    # category pass off -> only the detection call is billed
    assert resp["usage"]["prompt_tokens"] == 1000
