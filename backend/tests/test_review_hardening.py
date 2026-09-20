"""Regression coverage for typed review calls and review route accounting."""

import json
from pathlib import Path
from typing import Any

import pytest
from app.config import settings
from app.services.jev import jev_review_questions
from app.services.openai_adapter import _select_word
from app.utils.cache import hash_payload
from app.utils.metrics import metrics

from tests.test_openai_review import build_review_prompt, make_text_fake
from tests.test_review_corpus import _rows_for_range, _segments

REAL_TRANSCRIPT = _segments("ep-daily-tech-news-show-c1904b8605f7")[0]["text"]


@pytest.fixture(autouse=True)
def reset_review_metrics():
    metrics.reset()


@pytest.fixture
def jev_env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(settings, "JEV_CACHE_PATH", str(tmp_path / "cache.json"))


def _review_questions() -> dict[str, dict[str, Any]]:
    return {
        "evidence": {
            "type": "noul",
            "instructions": "The candidate contains advertising language.",
        }
    }


def _review_call(tmp_path: Path, fetcher, questions=None):
    return jev_review_questions(
        state={"transcript": REAL_TRANSCRIPT},
        questions=questions or _review_questions(),
        url="u",
        api_key="k",
        timeout=1.0,
        cache_path=str(tmp_path / "review-cache.json"),
        model="jev-latest",
        fetcher=fetcher,
    )


def test_review_helper_does_not_cache_poisoned_typed_entry(tmp_path):
    questions = _review_questions()
    payload = {"state": {"transcript": REAL_TRANSCRIPT}, "model": "jev-latest", "questions": questions}
    (tmp_path / "review-cache.json").write_text(
        json.dumps({hash_payload(payload): {"answers": {"evidence": "bad"}, "input_tokens": 1, "output_tokens": 1}})
    )
    calls = 0

    def fetcher(_payload, **_kwargs):
        nonlocal calls
        calls += 1
        return {"answers": {"evidence": {"noul": 0.99}}, "usage": {"input_tokens": 3, "output_tokens": 2}}

    result = _review_call(tmp_path, fetcher, questions)
    assert calls == 1
    assert result["cache_hit"] is False
    assert result["answers"]["evidence"] == 0.99


def test_review_helper_accepts_normalized_evidence_cache_and_hits_again(tmp_path):
    questions = _review_questions()
    payload = {"state": {"transcript": REAL_TRANSCRIPT}, "model": "jev-latest", "questions": questions}
    (tmp_path / "review-cache.json").write_text(
        json.dumps({hash_payload(payload): {"answers": {"evidence": {"noul": 0.99}}, "input_tokens": 1, "output_tokens": 1}})
    )
    calls = 0

    def fetcher(_payload, **_kwargs):
        nonlocal calls
        calls += 1
        return {"answers": {"evidence": {"noul": 0.99}}, "usage": {"input_tokens": 3, "output_tokens": 2}}

    result = _review_call(tmp_path, fetcher, questions)
    assert result["cache_hit"] is False
    assert calls == 1
    def cache_miss(_payload, **_kwargs):
        raise AssertionError("cache miss")

    second = _review_call(tmp_path, cache_miss, questions)
    assert second["cache_hit"] is True
    assert calls == 1


def test_review_helper_rejects_cache_entry_with_missing_usage(tmp_path):
    questions = _review_questions()
    payload = {"state": {"transcript": REAL_TRANSCRIPT}, "model": "jev-latest", "questions": questions}
    (tmp_path / "review-cache.json").write_text(
        json.dumps({hash_payload(payload): {"answers": {"evidence": 0.99}, "usage": None}})
    )
    calls = 0

    def fetcher(_payload, **_kwargs):
        nonlocal calls
        calls += 1
        return {"answers": {"evidence": {"noul": 0.99}}, "usage": {"input_tokens": 3, "output_tokens": 2}}

    result = _review_call(tmp_path, fetcher, questions)
    assert result["cache_hit"] is False
    assert calls == 1


async def test_review_gap_is_rejected_before_upstream(jev_env, client, monkeypatch):
    import app.services.jev as jev

    called = False

    def fetcher(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("a candidate in a transcript gap must not call Jev")

    monkeypatch.setattr(jev, "call_payload", fetcher)
    rows = _segments("ep-oxide-and-friends-ce789ff5b62e")
    prompt = _corpus_prompt(8.0, 9.0, rows[:1], [], rows[1:2])
    response = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": prompt}]},
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "jev_review_invalid_request"
    assert called is False


@pytest.mark.parametrize("bad", [float("nan"), -0.01, 1.01])
def test_review_helper_rejects_invalid_upstream_probability(tmp_path, bad):
    def fetcher(_payload, **_kwargs):
        return {"answers": {"evidence": {"noul": bad}}, "usage": {"input_tokens": 3, "output_tokens": 2}}

    with pytest.raises(ValueError, match="evidence"):
        _review_call(tmp_path, fetcher)


def test_choice_hierarchy_retains_all_real_corpus_word_candidates():
    rows = _segments("ep-daily-tech-news-show-c1904b8605f7")
    words = [
        {"start": word["start"], "end": word["end"], "text": word["word"].strip()}
        for row in rows
        for word in row["words"]
    ][:255]
    assert len(words) == 255

    seen: list[dict[str, Any]] = []

    def request(questions, _stage):
        seen.append(questions)
        answers = {}
        for key, question in questions.items():
            choice = next(option for option in question["criteria"] if option != "unknown")
            probabilities = {option: (1.0 if option == choice else 0.0) for option in question["criteria"]}
            answers[key] = {"choice": choice, "confidence": 0.99, "probabilities": probabilities}
        return {"answers": answers}

    selected = _select_word(prefix="start", words=words, enter=0.95, request=request)
    assert selected == words[0]
    assert len(seen[0]) == 2
    assert max(len(question["criteria"]) for question in seen[0].values()) == 255
    assert sum(len(question["criteria"]) - 1 for question in seen[0].values()) == 255


@pytest.mark.parametrize("answer", [
    {"choice": "unknown", "confidence": 0.99},
    {"choice": "w0000", "confidence": 0.94},
])
def test_choice_unknown_or_low_confidence_is_inconclusive(answer):
    words = [
        {"start": word["start"], "end": word["end"], "text": word["word"].strip()}
        for word in _segments("ep-daily-tech-news-show-c1904b8605f7")[0]["words"][:1]
    ]

    def request(questions, _stage):
        key, question = next(iter(questions.items()))
        choice = answer["choice"] if answer["choice"] in question["criteria"] else next(option for option in question["criteria"] if option != "unknown")
        probabilities = {option: (1.0 if option == choice else 0.0) for option in question["criteria"]}
        return {"answers": {key: {"choice": choice, "confidence": answer["confidence"], "probabilities": probabilities}}}

    assert _select_word(prefix="start", words=words, enter=0.95, request=request) is None


def _corpus_prompt(start: float, end: float, before: list[dict], candidate: list[dict], after: list[dict]) -> str:
    def as_rows(rows):
        return [(row["start"], row["end"], row["text"]) for row in rows]

    return build_review_prompt(start, end, as_rows(before), as_rows(candidate), as_rows(after))


async def test_review_api_metrics_and_request_ids(jev_env, client, monkeypatch, tmp_path):
    import app.services.jev as jev

    monkeypatch.setattr(jev, "call_payload", make_text_fake())
    monkeypatch.setattr(settings, "JEV_CACHE_PATH", str(tmp_path / "api-cache.json"))
    segments = _segments("ep-daily-tech-news-show-c1904b8605f7")
    confirmed_start, confirmed_end = 821.4, 889.7
    adjusted_start, adjusted_end = 13 * 60 + 50.0, 17 * 60
    confirmed = _rows_for_range(segments, confirmed_start, confirmed_end)
    confirmed_before = _rows_for_range(segments, confirmed_start - 20, confirmed_start)
    confirmed_after = _rows_for_range(segments, confirmed_end, confirmed_end + 20)
    adjusted = _rows_for_range(segments, adjusted_start, adjusted_end)
    adjusted_before = _rows_for_range(segments, adjusted_start - 20, adjusted_start)
    adjusted_after = _rows_for_range(segments, adjusted_end, adjusted_end + 20)
    no_ad = _segments("ep-oxide-and-friends-ce789ff5b62e")
    cases = [
        _corpus_prompt(confirmed_start, confirmed_end, confirmed_before, confirmed, confirmed_after),
        _corpus_prompt(adjusted_start, adjusted_end, adjusted_before, adjusted, adjusted_after),
        _corpus_prompt(no_ad[0]["start"], no_ad[0]["end"], [], no_ad[:1], no_ad[1:2]),
    ]
    responses = []
    for prompt in cases:
        responses.append(await client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": prompt}]}, headers={"Authorization": "Bearer test-key"}))

    assert [response.status_code for response in responses] == [200, 200, 200]
    ids = [response.headers["x-request-id"] for response in responses]
    assert all(ids) and len(set(ids)) == 3
    snapshot = metrics.snapshot()["review"]
    assert snapshot["outcomes"]["confirmed"] == 1
    assert snapshot["outcomes"]["adjusted"] == 1
    assert snapshot["outcomes"]["rejected"] == 1
