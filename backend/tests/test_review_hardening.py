"""Regression coverage for typed review calls and review route accounting."""

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from app.config import settings
from app.services.jev import jev_review_questions
from app.services.openai_adapter import _focused_range_question, _review_state
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


def test_review_helper_accepts_focused_noul_and_validates_its_cache(tmp_path):
    questions = {
        "proposed_range": {
            "type": "noul",
            "instructions": "The proposed complete range is advertising.",
        }
    }

    def fetcher(_payload, **_kwargs):
        return {"answers": {"proposed_range": {"noul": 0.99}}, "usage": {"input_tokens": 3, "output_tokens": 2}}

    result = _review_call(tmp_path, fetcher, questions)

    assert result["answers"]["proposed_range"] == 0.99


@pytest.mark.parametrize("pool", ["accepted", "resurrection"])
async def test_review_gap_is_inconclusive_before_upstream(
    jev_env, client, monkeypatch, caplog, pool
):
    import app.services.jev as jev

    called = False

    def fetcher(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("a candidate in a transcript gap must not call Jev")

    monkeypatch.setattr(jev, "call_payload", fetcher)
    rows = _segments("ep-oxide-and-friends-ce789ff5b62e")
    prompt = _corpus_prompt(8.0, 9.0, rows[:1], [], rows[1:2], pool=pool)
    with caplog.at_level(logging.INFO):
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": prompt}]},
            headers={"Authorization": "Bearer test-key"},
        )

    assert response.status_code == 422
    assert response.headers["x-should-retry"] == "false"
    request_id = response.headers["x-request-id"]
    assert response.json()["error"]["code"] == "jev_review_inconclusive"
    assert called is False
    review_metrics = metrics.snapshot()["review"]
    assert review_metrics["outcomes"]["inconclusive"] == 1
    assert review_metrics["outcomes"]["invalid_request"] == 0
    assert review_metrics["outcomes"]["rejected"] == 0
    refinement = review_metrics["refinement"]
    assert refinement["attempted"] == 0
    assert refinement["skipped"]["transcript_gap"] == 1
    assert review_metrics["reasons"]["transcript_gap"] == 1
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        f"request_id={request_id} unavailable reason=candidate lies in a transcript gap" in message
        for message in messages
    )
    assert any(
        f"request_id={request_id} status=422 error_code=jev_review_inconclusive reason=transcript_gap"
        in message
        for message in messages
    )
    assert any(
        f"request_id={request_id} outcome=inconclusive reason=transcript_gap" in message
        for message in messages
    )


@pytest.mark.parametrize(
    "case",
    ["before", "after", "zero_width", "negative"],
)
async def test_review_outside_or_invalid_bounds_remain_invalid_request(
    jev_env, client, monkeypatch, case
):
    import app.services.jev as jev

    called = False

    def fetcher(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("invalid review input must not call Jev")

    monkeypatch.setattr(jev, "call_payload", fetcher)
    rows = _segments("ep-oxide-and-friends-ce789ff5b62e")
    context_start = float(rows[0]["start"])
    context_end = float(rows[1]["end"])
    candidates = {
        "before": (context_start - 2.0, context_start - 1.0),
        "after": (context_end + 1.0, context_end + 2.0),
        "zero_width": (context_start, context_start),
        "negative": (-1.0, 1.0),
    }
    prompt = _corpus_prompt(*candidates[case], rows[:1], [], rows[1:2])
    response = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": prompt}]},
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 422
    assert response.headers["x-should-retry"] == "false"
    assert response.headers["x-request-id"]
    assert response.json()["error"]["code"] == "jev_review_invalid_request"
    assert called is False


@pytest.mark.parametrize("bad", [float("nan"), -0.01, 1.01])
def test_review_helper_rejects_invalid_upstream_probability(tmp_path, bad):
    def fetcher(_payload, **_kwargs):
        return {"answers": {"evidence": {"noul": bad}}, "usage": {"input_tokens": 3, "output_tokens": 2}}

    with pytest.raises(ValueError, match="evidence"):
        _review_call(tmp_path, fetcher)


def test_pair_choice_keeps_all_supplied_words_in_shared_state():
    rows = _segments("ep-daily-tech-news-show-c1904b8605f7")
    words = [
        {"start": word["start"], "end": word["end"], "text": word["word"].strip()}
        for row in rows
        for word in row["words"]
    ][:255]
    assert len(words) == 255

    segments = [{"sid": 0, "start": words[0]["start"], "end": words[-1]["end"], "text": "context"}]
    state = _review_state(
        segments,
        {"start": words, "end": words},
        (words[0]["start"], words[-1]["end"]),
        "guidance",
    )
    assert len(words) == 255
    assert state["boundary_words"]["start"] == words
    assert state["boundary_words"]["end"] == words


def test_focused_range_question_requires_complete_observed_support():
    question = _focused_range_question("proposed_range", (10.0, 20.0))

    assert question["proposed_range"]["type"] == "noul"
    assert "entire interval" in question["proposed_range"]["instructions"]
    assert "interior portion lacks observed transcript evidence" in question["proposed_range"]["instructions"]


def _corpus_prompt(
    start: float,
    end: float,
    before: list[dict],
    candidate: list[dict],
    after: list[dict],
    *,
    pool: str = "accepted",
) -> str:
    def as_rows(rows):
        return [(row["start"], row["end"], row["text"]) for row in rows]

    return build_review_prompt(
        start,
        end,
        as_rows(before),
        as_rows(candidate),
        as_rows(after),
        pool=pool,
    )


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
