"""Tests for the Jev proxy: payload building, parsing, caching, endpoint."""

import json
import multiprocessing
import time
from typing import Any

import httpx
import pytest
from app.config import Settings, settings
from app.services.jev import (
    JevCategoryValidationError,
    JevReviewValidationError,
    _retry_after_seconds,
    _review_answers,
    build_category_payload,
    build_payload,
    call_payload,
    estimate_cost_usd,
    jev_ask,
    jev_category,
    jev_review_questions,
    parse_response,
)
from app.utils.cache import JsonCache, hash_payload
from app.utils.spans import spans_from_probabilities

ANSWER_BODY: dict[str, Any] = {
    "answers": {"s1": {"noul": 0.98}, "junk": {"noul": 0.5}},
    "usage": {"input_tokens": 1000, "output_tokens": 5},
}

REVIEW_QUESTIONS = {
    "evidence": {"type": "noul"},
    "start_0": {"type": "choice", "criteria": {"unknown": "none", "w0": "word"}},
}

CATEGORY_LABELS = [
    "sponsor",
    "cross_promo",
    "self_promo",
    "interaction",
    "intro",
    "outro",
    "recap",
]

LEGACY_CATEGORY_DESCRIPTIONS = {
    "sponsor": "a paid sponsor read or product advertisement for an outside advertiser",
    "cross_promo": "a cross-promotion for another podcast or show",
    "self_promo": "the host promoting their own show, Patreon, merch, membership, or back catalog",
    "interaction": "a call to action to like, subscribe, rate, review, follow, or comment",
    "intro": "an intro segment opening the episode",
    "outro": "an outro segment closing the episode",
    "recap": "a recap or summary of earlier content",
}


def _valid_review_body() -> dict[str, Any]:
    return {
        "answers": {
            "evidence": {"noul": 0.96},
            "start_0": {
                "choice": "w0",
                "confidence": 0.94,
                "probabilities": {"unknown": 0.1, "w0": 0.9},
            },
        },
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }


def _store_cache_entry(cache_path: str, payload: dict[str, Any]) -> None:
    JsonCache(cache_path).get_or_fetch(payload, lambda: {"value": payload["state"]})


@pytest.fixture
def jev_env(tmp_path, monkeypatch):
    """Point settings at a test key and a throwaway cache file."""
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(settings, "JEV_CACHE_PATH", str(tmp_path / "cache.json"))


def test_build_payload_shape():
    payload = build_payload([{"sid": 1, "text": " hello "}], model="jev-latest")
    assert payload["model"] == "jev-latest"
    assert payload["state"]["transcript"] == "L0001| hello"
    question = payload["questions"]["s1"]
    assert question["type"] == "noul"
    assert "L0001" in question["instructions"]


def test_build_payload_keeps_caller_context_separate_and_hashes_it():
    segments = [{"sid": 1, "text": " hello "}]
    base = build_payload(segments, model="jev-latest")
    empty = build_payload(segments, model="jev-latest", caller_context=" \n ")
    contextual = build_payload(
        segments,
        model="jev-latest",
        caller_context="Podcast description: show notes and cue metadata",
    )

    assert empty == base
    assert "caller_context" not in base["state"]
    assert contextual["state"]["caller_context"] == "Podcast description: show notes and cue metadata"
    assert contextual["state"]["transcript"] == base["state"]["transcript"]
    assert contextual["state"]["guidance"] == base["state"]["guidance"]
    assert hash_payload(contextual) != hash_payload(base)


def test_parse_response_keeps_noul_answers_and_usage():
    entry = parse_response(ANSWER_BODY)
    assert entry["probabilities"] == {"s1": 0.98}
    assert entry["input_tokens"] == 1000
    assert entry["output_tokens"] == 5


def test_review_validation_error_exposes_only_fixed_rule_and_numeric_details():
    body = _valid_review_body()
    body["answers"]["start_0"]["probabilities"] = {"unknown": 0.2, "w0": 0.2}
    with pytest.raises(JevReviewValidationError) as raised:
        _review_answers(body, REVIEW_QUESTIONS)
    error = raised.value
    assert error.code == "jev_upstream_invalid_response"
    assert error.rule == "choice_probability_sum"
    assert error.numeric_details == {
        "expected_total": 1.0,
        "actual_total": 0.4,
        "tolerance": 0.01,
    }
    assert "unknown" not in str(error)


@pytest.mark.parametrize("total", [0.99, 1.01, 0.9900000000000001, 1.0099999999999998])
def test_review_validation_accepts_rounded_choice_probability_totals(total):
    body = _valid_review_body()
    body["answers"]["start_0"]["probabilities"] = {
        "unknown": max(0.0, total - 1.0),
        "w0": min(1.0, total),
    }
    parsed = _review_answers(body, REVIEW_QUESTIONS)
    assert parsed["answers"]["start_0"]["probabilities"] == {
        "unknown": max(0.0, total - 1.0),
        "w0": min(1.0, total),
    }


@pytest.mark.parametrize("total", [0.989, 1.011, 0.0, 0.4])
def test_review_validation_rejects_materially_drifted_choice_probability_totals(total):
    body = _valid_review_body()
    body["answers"]["start_0"]["probabilities"] = {
        "unknown": max(0.0, total - 1.0),
        "w0": min(1.0, total),
    }
    with pytest.raises(JevReviewValidationError) as raised:
        _review_answers(body, REVIEW_QUESTIONS)
    assert raised.value.rule == "choice_probability_sum"


def test_review_choice_cache_preserves_rounded_distribution(tmp_path):
    questions = {
        "evidence": {"type": "noul"},
        "start_0": {
            "type": "choice",
            "criteria": {"unknown": "none", "w0": "word"},
        },
    }
    calls = 0

    def fetcher(_payload, **_kwargs):
        nonlocal calls
        calls += 1
        return {
            "answers": {
                "evidence": {"noul": 0.96},
                "start_0": {
                    "choice": "w0",
                    "confidence": 0.94,
                    "probabilities": {"unknown": 0.0, "w0": 0.99},
                },
            },
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }

    kwargs = {
        "state": {"transcript": "transcript"},
        "questions": questions,
        "url": "u",
        "api_key": "k",
        "timeout": 1.0,
        "cache_path": str(tmp_path / "review-cache.json"),
        "model": "jev-latest",
        "fetcher": fetcher,
    }
    first = jev_review_questions(**kwargs)
    second = jev_review_questions(**kwargs)
    assert calls == 1
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["answers"]["start_0"]["probabilities"] == {"unknown": 0.0, "w0": 0.99}
    assert second["answers"]["start_0"]["confidence"] == 0.94


@pytest.mark.parametrize(
    "probabilities",
    [{"unknown": 0.0, "w0": 0.98}, {"w0": 1.0}],
)
def test_review_choice_invalid_distribution_is_not_cached(tmp_path, probabilities):
    questions = {
        "evidence": {"type": "noul"},
        "start_0": {
            "type": "choice",
            "criteria": {"unknown": "none", "w0": "word"},
        },
    }
    calls = 0

    def fetcher(_payload, **_kwargs):
        nonlocal calls
        calls += 1
        return {
            "answers": {
                "evidence": {"noul": 0.96},
                "start_0": {
                    "choice": "w0",
                    "confidence": 0.94,
                    "probabilities": probabilities,
                },
            },
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }

    kwargs = {
        "state": {"transcript": "transcript"},
        "questions": questions,
        "url": "u",
        "api_key": "k",
        "timeout": 1.0,
        "cache_path": str(tmp_path / "review-cache.json"),
        "model": "jev-latest",
        "fetcher": fetcher,
    }
    with pytest.raises(ValueError):
        jev_review_questions(**kwargs)
    with pytest.raises(ValueError):
        jev_review_questions(**kwargs)
    assert calls == 2


def test_review_validation_allowlist_and_confidence_are_independent():
    error = JevReviewValidationError(
        "not-a-rule", {"unexpected": 1, "actual_count": 2}
    )
    assert error.rule == "response_object"
    assert error.numeric_details == {"actual_count": 2}

    body = _valid_review_body()
    body["answers"]["start_0"]["confidence"] = 0.1
    assert _review_answers(body, REVIEW_QUESTIONS)["answers"]["start_0"]["confidence"] == 0.1


@pytest.mark.parametrize(
    ("body", "rule"),
    [
        (None, "response_object"),
        ({"answers": {}}, "answers_keys"),
        ({"answers": {"evidence": {"noul": float("nan")}, "start_0": {}}}, "evidence_probability"),
        (
            {
                "answers": {
                    "evidence": {"noul": 0.9},
                    "start_0": {"choice": "w0", "confidence": 0.9},
                }
            },
            "choice_answer_shape",
        ),
        (
            {
                "answers": {
                    "evidence": {"noul": 0.9},
                    "start_0": {
                        "choice": "w0",
                        "confidence": 0.9,
                        "probabilities": {"w0": 1.0},
                    },
                }
            },
            "choice_probability_keys",
        ),
        (
            {
                "answers": {
                    "evidence": {"noul": 0.9},
                    "start_0": {
                        "choice": "w0",
                        "confidence": float("inf"),
                        "probabilities": {"unknown": 0.1, "w0": 0.9},
                    },
                }
            },
            "choice_confidence",
        ),
        (
            {
                "answers": {
                    "evidence": {"noul": 0.9},
                    "start_0": {
                        "choice": "unknown",
                        "confidence": 0.9,
                        "probabilities": {"unknown": 0.1, "w0": 0.9},
                    },
                }
            },
            "choice_winner",
        ),
    ],
)
def test_review_validation_reports_malformed_rules(body, rule):
    with pytest.raises(JevReviewValidationError) as raised:
        _review_answers(body, REVIEW_QUESTIONS)
    assert raised.value.code == "jev_upstream_invalid_response"
    assert raised.value.rule == rule
    assert all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in raised.value.numeric_details.values())


def test_review_validation_reports_usage_rule_without_echoing_value():
    body = _valid_review_body()
    body["usage"]["input_tokens"] = "secret-like-invalid-value"
    with pytest.raises(JevReviewValidationError) as raised:
        _review_answers(body, REVIEW_QUESTIONS)
    assert raised.value.rule == "usage_input_tokens"
    assert "secret-like" not in str(raised.value)


def test_review_validation_reports_choice_criteria_shape():
    body = _valid_review_body()
    questions = {**REVIEW_QUESTIONS, "start_0": {"type": "choice", "criteria": []}}
    with pytest.raises(JevReviewValidationError) as raised:
        _review_answers(body, questions)
    assert raised.value.rule == "choice_criteria_object"


@pytest.mark.parametrize("usage", [[], False, 0, ""])
def test_review_validation_reports_usage_object_shape(usage):
    body = _valid_review_body()
    body["usage"] = usage
    with pytest.raises(JevReviewValidationError) as raised:
        _review_answers(body, REVIEW_QUESTIONS)
    assert raised.value.rule == "usage_object"


@pytest.mark.parametrize("value", [True, float("nan"), float("inf")])
def test_parse_response_excludes_non_finite_probabilities(value):
    assert parse_response({"answers": {"s1": {"noul": value}}})["probabilities"] == {}


def test_estimate_cost_bills_input_only():
    assert estimate_cost_usd(1_000_000) == 0.042


def test_retry_configuration_rejects_negative_values():
    with pytest.raises(ValueError):
        Settings(JEV_MAX_RETRIES=-1)


def test_call_payload_sends_bearer_header():
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["Authorization"]
        return httpx.Response(200, json=ANSWER_BODY)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    body = call_payload({"state": {}}, url="https://example.test/v1", api_key="k", client=client)
    assert seen["auth"] == "Bearer k"
    assert body == ANSWER_BODY


def test_call_payload_retries_on_429_then_succeeds(monkeypatch):
    statuses = [429, 200]

    def handler(request: httpx.Request) -> httpx.Response:
        code = statuses.pop(0)
        if code == 429:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json=ANSWER_BODY)

    monkeypatch.setattr(time, "sleep", lambda _s: None)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    body = call_payload({"state": {}}, url="https://x.test/v1", api_key="k", client=client)
    assert body == ANSWER_BODY
    assert statuses == []


def test_call_payload_raises_after_retries_exhausted(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    monkeypatch.setattr(time, "sleep", lambda _s: None)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        call_payload(
            {"state": {}}, url="https://x.test/v1", api_key="k", max_retries=1, client=client
        )


def test_call_payload_does_not_retry_on_422(monkeypatch):
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(422)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        call_payload({"state": {}}, url="https://x.test/v1", api_key="k", client=client)
    assert len(calls) == 1


def test_call_payload_retries_transport_error(monkeypatch):
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("upstream timed out")
        return httpx.Response(200, json=ANSWER_BODY)

    monkeypatch.setattr(time, "sleep", lambda _s: None)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert call_payload({"state": {}}, url="https://x.test", api_key="k", client=client) == ANSWER_BODY
    assert attempts == 2


def test_retry_after_falls_back_to_body_when_header_is_invalid():
    response = httpx.Response(429, headers={"Retry-After": "not-a-date"}, json={"retry_after_ms": 250})
    assert _retry_after_seconds(response) == 0.25


def test_jev_ask_caches_by_payload_hash(jev_env, tmp_path):
    calls: list[int] = []

    def fake(payload, *, url, api_key, timeout, max_retries=2, **_kwargs):
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


def test_invalid_cached_entry_is_evicted_and_refetched(jev_env, tmp_path):
    cache_path = tmp_path / "answers.json"
    payload = build_payload([{"sid": 1, "text": "hello"}], model="m")
    cache_path.write_text(
        json.dumps(
            {
                hash_payload(payload): {
                    "probabilities": {},
                    "input_tokens": 1,
                    "output_tokens": 1,
                }
            }
        )
    )

    calls = 0

    def valid_response(payload, **_kwargs):
        nonlocal calls
        calls += 1
        return ANSWER_BODY

    result = jev_ask(
        [{"sid": 1, "text": "hello"}],
        url="u",
        api_key="k",
        timeout=1.0,
        cache_path=str(cache_path),
        model="m",
        fetcher=valid_response,
    )
    assert calls == 1
    assert result["probabilities"] == {"s1": 0.98}


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_detection_rejects_out_of_range_answers(tmp_path, value):
    with pytest.raises(ValueError, match="valid answers"):
        jev_ask(
            [{"sid": 1, "text": "ad"}],
            url="u",
            api_key="k",
            timeout=1.0,
            cache_path=str(tmp_path / "answers.json"),
            model="m",
            fetcher=lambda *_args, **_kwargs: {"answers": {"s1": {"noul": value}}},
        )


def _category_body(category: str = "sponsor") -> dict[str, Any]:
    return {
        "answers": {
            "category": {
                "choice": category,
                "confidence": 0.6,
                "probabilities": {
                    label: 0.4 if label == category else 0.1 for label in CATEGORY_LABELS
                },
            }
        },
        "usage": {"input_tokens": 7, "output_tokens": 3},
    }


def test_category_payload_is_one_choice_with_focus_context_and_uid():
    payload = build_category_payload(
        [{"sid": 3, "text": "focus"}],
        [{"sid": 1, "text": "before"}, {"sid": 5, "text": "after"}],
        CATEGORY_LABELS,
        model="m",
        uid="uid",
        guidance="policy",
    )
    assert payload["state"] == {
        "guidance": "policy",
        "transcript": "L0001| before\nL0003| focus\nL0005| after",
        "focus": "L0003",
        "uid": "uid",
    }
    assert set(payload["questions"]) == {"category"}
    question = payload["questions"]["category"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == set(CATEGORY_LABELS)
    assert "paid ad for another podcast or show" in question["criteria"]["sponsor"]
    assert "unpaid promotion" in question["criteria"]["cross_promo"]


@pytest.mark.parametrize("category", CATEGORY_LABELS)
def test_category_returns_selected_choice_and_cache_usage(tmp_path, category):
    calls = 0

    def fetcher(_payload, **_kwargs):
        nonlocal calls
        calls += 1
        return _category_body(category)

    kwargs = {
        "url": "u",
        "api_key": "k",
        "timeout": 1.0,
        "cache_path": str(tmp_path / "categories.json"),
        "model": "m",
        "fetcher": fetcher,
    }
    first = jev_category([{"sid": 1, "text": "ad"}], [], CATEGORY_LABELS, **kwargs)
    second = jev_category([{"sid": 1, "text": "ad"}], [], CATEGORY_LABELS, **kwargs)
    assert calls == 1
    assert first["category"] == second["category"] == category
    assert first["confidence"] == second["confidence"] == 0.6
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["usage"] == {"input_tokens": 7, "output_tokens": 3}


@pytest.mark.parametrize(
    ("body", "rule"),
    [
        ({"answers": {}}, "answers_keys"),
        (_category_body("missing"), "choice_probability_keys"),
        (
            {
                "answers": {
                    "category": {
                        "choice": "sponsor",
                        "confidence": 0.6,
                        "probabilities": {label: float("nan") for label in CATEGORY_LABELS},
                    }
                }
            },
            "choice_probability",
        ),
        (
            {
                "answers": {
                    "category": {
                        "choice": "sponsor",
                        "confidence": 0.6,
                        "probabilities": dict.fromkeys(CATEGORY_LABELS, 0.1),
                    }
                }
            },
            "choice_probability_sum",
        ),
    ],
)
def test_category_rejects_malformed_choice_without_caching(tmp_path, body, rule):
    calls = 0

    def fetcher(_payload, **_kwargs):
        nonlocal calls
        calls += 1
        return body

    kwargs = {
        "url": "u",
        "api_key": "k",
        "timeout": 1.0,
        "cache_path": str(tmp_path / "categories.json"),
        "model": "m",
        "fetcher": fetcher,
    }
    for _ in range(2):
        with pytest.raises(JevCategoryValidationError) as raised:
            jev_category([{"sid": 1, "text": "ad"}], [], CATEGORY_LABELS, **kwargs)
        assert raised.value.rule == rule
    assert calls == 2


def test_category_payload_does_not_reuse_legacy_noul_cache(tmp_path):
    cache_path = str(tmp_path / "categories.json")
    current_payload = build_category_payload(
        [{"sid": 1, "text": "ad"}], [], CATEGORY_LABELS, model="m"
    )
    legacy_payload = {
        "state": current_payload["state"],
        "model": current_payload["model"],
        "questions": {
            f"c{index}": {
                "type": "noul",
                "instructions": (
                    "The advertising break at lines L0001 is "
                    f"{LEGACY_CATEGORY_DESCRIPTIONS[category]}."
                ),
            }
            for index, category in enumerate(CATEGORY_LABELS)
        },
    }
    JsonCache(cache_path).get_or_fetch(
        legacy_payload,
        lambda: {
            "probabilities": {f"c{index}": 0.5 for index in range(len(CATEGORY_LABELS))},
            "input_tokens": 1,
            "output_tokens": 1,
        },
    )
    calls = 0

    def fetcher(_payload, **_kwargs):
        nonlocal calls
        calls += 1
        return _category_body()

    result = jev_category(
        [{"sid": 1, "text": "ad"}],
        [],
        CATEGORY_LABELS,
        url="u",
        api_key="k",
        timeout=1.0,
        cache_path=cache_path,
        model="m",
        fetcher=fetcher,
    )
    assert hash_payload(legacy_payload) != hash_payload(current_payload)
    assert calls == 1
    assert result["cache_hit"] is False


def test_category_accepts_rounded_distribution_without_changing_confidence(tmp_path):
    body = _category_body()
    body["answers"]["category"]["confidence"] = 0.6
    body["answers"]["category"]["probabilities"] = {
        label: 0.39 if label == "sponsor" else 0.1 for label in CATEGORY_LABELS
    }
    result = jev_category(
        [{"sid": 1, "text": "ad"}],
        [],
        CATEGORY_LABELS,
        url="u",
        api_key="k",
        timeout=1.0,
        cache_path=str(tmp_path / "categories.json"),
        model="m",
        fetcher=lambda *_args, **_kwargs: body,
    )
    assert sum(result["probabilities"].values()) == pytest.approx(0.99)
    assert result["confidence"] == 0.6


def test_cache_keeps_entries_from_concurrent_processes(tmp_path):
    cache_path = str(tmp_path / "shared.json")
    payloads = [{"state": {"window": "one"}}, {"state": {"window": "two"}}]
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_store_cache_entry, args=(cache_path, payload)) for payload in payloads
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    for payload in payloads:
        entry, hit = JsonCache(cache_path).get_or_fetch(payload, lambda: pytest.fail("cache miss"))
        assert hit is True
        assert entry["value"] == payload["state"]


def test_cache_evicts_old_entries_at_the_configured_limit(tmp_path):
    cache = JsonCache(tmp_path / "bounded.json", max_entries=1)
    first = {"state": {"window": "one"}}
    second = {"state": {"window": "two"}}
    cache.get_or_fetch(first, lambda: {"value": "one"})
    cache.get_or_fetch(second, lambda: {"value": "two"})
    _, hit = JsonCache(tmp_path / "bounded.json", max_entries=1).get_or_fetch(
        first, lambda: {"value": "refetched"}
    )
    assert hit is False


def test_segment_id_runs_are_not_bridged_without_timings():
    segments = [{"sid": index, "text": "segment"} for index in range(5)]
    probabilities = {"s0": 0.99, "s1": 0.01, "s2": 0.01, "s3": 0.99, "s4": 0.01}
    assert spans_from_probabilities(segments, probabilities) == [
        {"start_id": 0, "end_id": 0, "confidence": 0.99},
        {"start_id": 3, "end_id": 3, "confidence": 0.99},
    ]


async def test_ask_endpoint_returns_probabilities(jev_env, client, monkeypatch):
    calls: list[int] = []
    seen: dict[str, Any] = {}

    def fake(payload, *, url, api_key, timeout, max_retries=2, **_kwargs):
        calls.append(1)
        seen.update(max_retries=max_retries, **_kwargs)
        return ANSWER_BODY

    import app.services.jev as service_module

    monkeypatch.setattr(service_module, "call_payload", fake)
    monkeypatch.setattr(settings, "JEV_ENTER", 0.88)
    monkeypatch.setattr(settings, "JEV_STAY", 0.22)
    monkeypatch.setattr(settings, "JEV_MAX_RETRIES", 7)

    resp = await client.post(
        "/api/v1/jev/ask",
        json={"segments": [{"sid": 1, "text": "hi"}]},
        headers={"Authorization": "Bearer test-key"},
    )
    assert resp.status_code == 200
    assert seen["max_retries"] == 7
    data = resp.json()
    assert data["probabilities"] == {"s1": 0.98}
    assert data["usage"] == {"input_tokens": 1000, "output_tokens": 5}
    assert data["cache_hit"] is False

    resp2 = await client.post(
        "/api/v1/jev/ask",
        json={"segments": [{"sid": 1, "text": "hi"}]},
        headers={"Authorization": "Bearer test-key"},
    )
    assert resp2.json()["cache_hit"] is True
    assert len(calls) == 1


async def test_ask_endpoint_requires_api_key(jev_env, client, monkeypatch):
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", None)
    resp = await client.post("/api/v1/jev/ask", json={"segments": [{"sid": 1, "text": "hi"}]})
    assert resp.status_code == 503


def test_native_ask_uses_configured_thresholds_when_omitted(jev_env, monkeypatch):
    import app.api.v1.jev as api

    seen: dict[str, Any] = {}

    def fake_ask(segments, **kwargs):
        seen.update(kwargs)
        return {"probabilities": {}, "spans": [], "usage": {}}

    monkeypatch.setattr(api, "jev_ask", fake_ask)
    monkeypatch.setattr(settings, "JEV_ENTER", 0.88)
    monkeypatch.setattr(settings, "JEV_STAY", 0.22)
    monkeypatch.setattr(settings, "JEV_MAX_RETRIES", 7)
    result = api.ask(
        api.AskRequest(segments=[{"sid": 1, "text": "hi"}]),
        "Bearer test-key",
    )
    assert result["spans"] == []
    assert seen["enter"] == 0.88
    assert seen["stay"] == 0.22
    assert seen["max_retries"] == 7


def test_native_ask_accepts_more_than_the_former_segment_cap(jev_env, monkeypatch):
    import app.api.v1.jev as api

    seen: dict[str, Any] = {}

    def fake_ask(segments, **_kwargs):
        seen["segments"] = segments
        return {"probabilities": {}, "spans": [], "usage": {}}

    monkeypatch.setattr(api, "jev_ask", fake_ask)
    api.ask(
        api.AskRequest(segments=[{"sid": index, "text": "segment"} for index in range(301)]),
        "Bearer test-key",
    )
    assert len(seen["segments"]) == 301
