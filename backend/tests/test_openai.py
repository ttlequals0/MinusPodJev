"""Tests for the OpenAI-compatible chat-completions adapter (no real network)."""

import json
from typing import Any

import pytest
from app.config import settings
from app.services.openai_adapter import (
    extract_user_text,
    parse_transcript,
    run_chat_completion,
)
from minuspod_compat import (
    AD_DETECTION_JSON_SCHEMA,
    SEGMENT_CATEGORIES,
    format_window_prompt,
    parse_ads_from_response,
    parse_id_ads_from_response,
    resolve_segment_id_ads,
)
from minuspod_compat.sponsors import is_sponsor_reasoning_rationale

_SCHEMA_AD_KEYS = set(
    AD_DETECTION_JSON_SCHEMA["properties"]["ads"]["items"]["properties"]
)


@pytest.fixture
def jev_env(tmp_path, monkeypatch):
    """Test key plus a throwaway cache file per test."""
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(settings, "JEV_CACHE_PATH", str(tmp_path / "cache.json"))


def make_fake(ad_sids, *, cat_index=0, input_tokens=1000, output_tokens=5):
    """Fetcher that scores detection NouLs and category Choices."""
    ad_sids = set(ad_sids)

    def fake(payload: dict[str, Any], *, url: str, api_key: str, timeout: float, max_retries: int = 2, **_kwargs: Any) -> dict[str, Any]:
        answers: dict[str, Any] = {}
        for key in payload["questions"]:
            if key.startswith("s"):
                answers[key] = {"noul": 0.98 if int(key[1:]) in ad_sids else 0.02}
            elif question := payload["questions"][key]:
                criteria = question.get("criteria", {})
                category = list(criteria)[cat_index]
                answers[key] = {
                    "choice": category,
                    "confidence": 0.97,
                    "probabilities": {
                        option: 0.97 if option == category else 0.005
                        for option in criteria
                    },
                }
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
    text = format_window_prompt(
        "Pod", "Ep", "[10.0s - 11.0s] show-note cue", TS_LINES,
        0, 1, 0.0, 600.0,
        audio_context="\n=== AUDIO SIGNALS ===\n[12.0s - 13.0s] cue",
    )
    segments, mode = parse_transcript(text)
    assert mode == "timestamps"
    assert len(segments) == 8
    assert segments[0] == {"start": 0.0, "end": 6.0, "text": "Welcome back to the show.", "sid": 0}
    assert segments[3]["sid"] == 3
    assert segments[3]["text"] == "This episode is sponsored by BetterHelp."


def test_parse_transcript_segment_ids_mode():
    lines = ["[10] first line", "[11] second line", "[12] third line"]
    text = format_window_prompt(
        "Pod", "Ep", "[10.0s - 11.0s] show-note cue", lines, 0, 1, 0.0, 600.0,
        audio_context="\n=== AUDIO SIGNALS ===\n[12.0s - 13.0s] cue",
        addressing_mode="segment_ids",
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


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_parse_transcript_allows_blank_after_heading(newline):
    text = newline.join([
        "Transcript:", "", "[10] Host story.", "[11] Acme sponsor read.",
        "", "AUDIO SIGNALS:", "[12.0s - 13.0s] cue",
    ])
    segments, mode = parse_transcript(text)
    assert mode == "segment_ids"
    assert [segment["sid"] for segment in segments] == [10, 11]


def test_extract_user_text_last_user_message():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "the transcript"},
    ]
    assert extract_user_text(messages) == "the transcript"


# sponsor matching now lives in app.services.sponsors; see test_sponsors.py


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
    resp = await client.post(
        "/v1/chat/completions", json=body, headers={"Authorization": "Bearer test-key"}
    )
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
    # The proxy adds only its provenance label beyond MinusPod's base schema.
    for raw_ad in json.loads(content)["ads"]:
        assert set(raw_ad) <= _SCHEMA_AD_KEYS
    ads = parse_ads_from_response(content)
    assert len(ads) == 1
    ad = ads[0]
    assert ad["start"] == 18.0
    assert ad["end"] == 36.0
    assert ad["category"] == "sponsor"
    assert ad["confidence"] == 0.98
    assert ad["sponsor"] == "jev-BetterHelp"
    assert ad["end_text"].startswith("Use code SHOW")
    assert ad["reason"].startswith("This episode is sponsored by BetterHelp.")
    assert is_sponsor_reasoning_rationale(ad["reason"]) is False


@pytest.mark.parametrize("category", SEGMENT_CATEGORIES)
@pytest.mark.parametrize("addressing_mode", ["timestamps", "segment_ids"])
def test_category_choice_propagates_for_timestamp_and_segment_id_prompts(
    jev_env, tmp_path, category, addressing_mode
):
    category_index = list(SEGMENT_CATEGORIES).index(category)
    if addressing_mode == "timestamps":
        prompt = format_window_prompt("Pod", "Ep", "", TS_LINES, 0, 1, 0.0, 600.0)
    else:
        lines = [f"[{index + 10}] {line.split('] ', 1)[1]}" for index, line in enumerate(TS_LINES)]
        prompt = format_window_prompt(
            "Pod", "Ep", "", lines, 0, 1, 0.0, 600.0, addressing_mode="segment_ids"
        )
    ad_sids = {3, 4, 5} if addressing_mode == "timestamps" else {13, 14, 15}
    response = run_chat_completion(
        messages=[{"role": "system", "content": "Keep policy context."}, {"role": "user", "content": prompt}],
        request_model="jev-latest",
        url="u",
        api_key="k",
        timeout=1.0,
        cache_path=str(tmp_path / f"{addressing_mode}-{category}.json"),
        model="jev-latest",
        enter=0.95,
        stay=0.40,
        category_pass=True,
        category_context=2,
        default_category="sponsor",
        fetcher=make_fake(ad_sids, cat_index=category_index),
    )
    ad = json.loads(response["choices"][0]["message"]["content"])["ads"][0]
    assert ad["category"] == category
    assert ad["confidence"] == 0.98


def test_category_choice_preserves_verification_policy_and_context(jev_env, tmp_path):
    prompt = format_window_prompt("Pod", "Ep", "", TS_LINES, 0, 1, 0.0, 600.0)
    calls: list[dict[str, Any]] = []

    def fetcher(payload, **kwargs):
        calls.append(payload)
        return make_fake({3, 4, 5})(payload, **kwargs)

    response = run_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are reviewing a podcast episode that has ALREADY had advertisements removed. "
                    "Verify the remaining transcript and keep host banter."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        request_model="jev-latest",
        url="u",
        api_key="k",
        timeout=1.0,
        cache_path=str(tmp_path / "verification.json"),
        model="jev-latest",
        enter=0.95,
        stay=0.40,
        category_pass=True,
        category_context=2,
        default_category="sponsor",
        fetcher=fetcher,
    )
    ad = json.loads(response["choices"][0]["message"]["content"])["ads"][0]
    category_payload = next(
        payload for payload in calls if payload["questions"].get("category", {}).get("type") == "choice"
    )
    assert ad["category"] == "sponsor"
    assert (
        "You are reviewing a podcast episode that has ALREADY had advertisements removed."
        in category_payload["state"]["guidance"]
    )
    assert category_payload["state"]["focus"] == "L0003-L0005"
    assert "L0001| Today we talk about hiking trips." in category_payload["state"]["transcript"]
    assert "L0006| Anyway, back to the trip we were on." in category_payload["state"]["transcript"]


async def test_malformed_category_choice_returns_safe_503(jev_env, client, monkeypatch, tmp_path, caplog):
    import app.services.jev as jev

    monkeypatch.setattr(settings, "JEV_CACHE_PATH", str(tmp_path / "cache.json"))

    def malformed_choice(payload, **kwargs):
        answers = {}
        for key, question in payload["questions"].items():
            if key.startswith("s"):
                answers[key] = {"noul": 0.98 if int(key[1:]) in {3, 4, 5} else 0.02}
            elif question["type"] == "choice":
                answers[key] = {"choice": "sponsor", "confidence": 0.6, "probabilities": {}}
        return {"answers": answers, "usage": {"input_tokens": 7, "output_tokens": 3}}

    monkeypatch.setattr(jev, "call_payload", malformed_choice)
    prompt = format_window_prompt("Pod", "Ep", "", TS_LINES, 0, 1, 0.0, 600.0)
    with caplog.at_level("WARNING"):
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": prompt}]},
            headers={"Authorization": "Bearer test-key"},
        )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "jev_category_upstream_invalid_response"
    assert "choice_probability_keys" in caplog.text
    assert "This episode is sponsored" not in caplog.text


async def test_chat_completions_no_ads(jev_env, client, monkeypatch):
    import app.services.jev as jev

    monkeypatch.setattr(jev, "call_payload", make_fake(set()))

    prompt = format_window_prompt("My Podcast", "Ep 1", "", TS_LINES, 0, 1, 0.0, 600.0)
    body = {"model": "jev-latest", "messages": [{"role": "user", "content": prompt}]}
    resp = await client.post(
        "/chat/completions", json=body, headers={"Authorization": "Bearer test-key"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    content = json.loads(data["choices"][0]["message"]["content"])
    assert content == {"ads": []}
    assert parse_ads_from_response(data["choices"][0]["message"]["content"]) == []


async def test_chat_completions_empty_prompt(jev_env, client):
    body = {"model": "jev-latest", "messages": [{"role": "user", "content": "no lines here"}]}
    resp = await client.post(
        "/v1/chat/completions", json=body, headers={"Authorization": "Bearer test-key"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert json.loads(data["choices"][0]["message"]["content"]) == {"ads": []}
    assert data["usage"]["total_tokens"] == 0


async def test_chat_completions_forwards_large_system_policy_unchanged(
    jev_env, client, monkeypatch
):
    import app.services.jev as jev
    from minuspod_compat import get_static_system_prompt

    calls: list[dict[str, Any]] = []
    suffix = "KEEP_THIS_ENDPOINT_TAIL_RULE"
    policy = f"{get_static_system_prompt()}\n{'p' * 12_001}\n{suffix}"

    def fetcher(payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        calls.append(payload)
        return make_fake({0})(payload, **kwargs)

    monkeypatch.setattr(jev, "call_payload", fetcher)
    response = await client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": policy},
                {"role": "user", "content": "[0.0s - 1.0s] sponsored by BetterHelp"},
            ]
        },
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 200
    assert policy in calls[0]["state"]["guidance"]
    assert suffix in calls[0]["state"]["guidance"]


async def test_chat_completions_accepts_content_above_former_aggregate_cap(
    jev_env, client, monkeypatch
):
    import app.services.jev as jev

    calls: list[dict[str, Any]] = []

    def fetcher(payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        calls.append(payload)
        return make_fake({0})(payload, **kwargs)

    monkeypatch.setattr(jev, "call_payload", fetcher)
    body = {
        "model": "jev-latest",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "p" * 200_000}]},
            {"role": "user", "content": "[0.0s - 1.0s] sponsored by BetterHelp"},
        ],
    }
    resp = await client.post(
        "/v1/chat/completions", json=body, headers={"Authorization": "Bearer test-key"}
    )
    assert resp.status_code == 200
    assert calls


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
        "My Podcast", "Ep 2", "[10.0s - 11.0s] show-note cue", lines,
        0, 1, 0.0, 600.0, addressing_mode="segment_ids",
    )
    body = {"model": "jev-latest", "messages": [{"role": "user", "content": prompt}]}
    resp = await client.post(
        "/v1/chat/completions", json=body, headers={"Authorization": "Bearer test-key"}
    )
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
    assert resolved[0]["sponsor"] == "jev-Squarespace"


async def test_chat_completions_requires_api_key(jev_env, client, monkeypatch):
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", None)
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "[0.0s - 1.0s] hi"}]},
    )
    assert resp.status_code == 503
    assert "TypeSafe API key" in resp.json()["detail"]


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


def test_policy_is_forwarded_and_separates_cache_entries(jev_env, tmp_path):
    prompt = format_window_prompt("Pod", "Ep", "", TS_LINES, 0, 1, 0.0, 600.0)
    calls: list[dict[str, Any]] = []

    def fetcher(payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        calls.append(payload)
        return make_fake({3, 4, 5})(payload, **kwargs)

    common = {
        "request_model": "jev-latest",
        "url": "u",
        "api_key": "k",
        "timeout": 1.0,
        "cache_path": str(tmp_path / "c.json"),
        "model": "jev-latest",
        "enter": 0.95,
        "stay": 0.40,
        "category_pass": False,
        "category_context": 2,
        "default_category": "sponsor",
        "fetcher": fetcher,
    }
    run_chat_completion(
        messages=[{"role": "system", "content": "Keep host banter."}, {"role": "user", "content": prompt}],
        **common,
    )
    run_chat_completion(
        messages=[{"role": "system", "content": "Keep intros."}, {"role": "user", "content": prompt}],
        **common,
    )
    assert len(calls) == 2
    assert "Keep host banter." in calls[0]["state"]["guidance"]
    assert "Keep intros." in calls[1]["state"]["guidance"]
    assert set(calls[0]["questions"]) == {f"s{i}" for i in range(len(TS_LINES))}


def test_large_default_policy_is_forwarded_unchanged(jev_env, tmp_path):
    from minuspod_compat import get_static_system_prompt

    calls: list[dict[str, Any]] = []
    suffix = "KEEP_THIS_TAIL_RULE"
    policy = f"{get_static_system_prompt()}\n{'p' * 12_001}\n{suffix}"

    def fetcher(payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        calls.append(payload)
        return make_fake({0})(payload, **kwargs)

    run_chat_completion(
        messages=[
            {"role": "system", "content": policy},
            {"role": "user", "content": "[0.0s - 1.0s] sponsored by BetterHelp"},
        ],
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
        fetcher=fetcher,
    )
    assert policy in calls[0]["state"]["guidance"]
    assert suffix in calls[0]["state"]["guidance"]


def test_direct_adapter_accepts_content_above_former_cap(jev_env, tmp_path):
    calls: list[dict[str, Any]] = []

    def fetcher(payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        calls.append(payload)
        return make_fake({0})(payload, **kwargs)

    run_chat_completion(
        messages=[
            {"role": "system", "content": "p" * 200_001},
            {"role": "user", "content": "[0.0s - 1.0s] sponsored by BetterHelp"},
        ],
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
        fetcher=fetcher,
    )
    assert calls


def test_unknown_sponsor_uses_grounded_reason_without_minting_label(jev_env, tmp_path, monkeypatch):
    import app.services.openai_adapter as adapter

    monkeypatch.setattr(adapter, "matched_sponsor_for_span", lambda *_args, **_kwargs: None)
    prompt = format_window_prompt(
        "Pod",
        "Ep",
        "",
        ["[0.0s - 10.0s] Sponsored by Unknown Brand, visit unknown.example today."],
        0,
        1,
        0.0,
        600.0,
    )
    response = run_chat_completion(
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
        fetcher=make_fake({0}),
    )
    content = response["choices"][0]["message"]["content"]
    raw_ad = json.loads(content)["ads"][0]
    assert "sponsor" not in raw_ad
    assert raw_ad["reason"] == "Based on transcript: Sponsored by Unknown Brand, visit unknown.example today."
    parsed = parse_ads_from_response(content)
    assert "sponsor" not in parsed[0]
