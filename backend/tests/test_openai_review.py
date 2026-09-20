"""Tests for the ad-review route of the OpenAI-compatible adapter (no network).

The review verdicts emitted here are round-tripped through MinusPod's own
review-response parser (minuspod_compat.extract_json_ads_array, the same call
ad_reviewer._review_single makes) plus a faithful copy of its verdict decision
(ad_reviewer.py:1281-1345), so a shape drift fails the test.
"""

import json
from typing import Any

import pytest
from app.config import settings
from app.services.openai_adapter import (
    ReviewInconclusiveError,
    ReviewUnavailableError,
    is_review_request,
    parse_candidate_bounds,
    parse_review_context,
    parse_review_segments,
    run_review,
)
from app.utils.metrics import metrics
from minuspod_compat import extract_json_ads_array, format_window_prompt

_AD_KW = ("sponsor", "betterhelp", "promo code", "brought to you by", "acast")


@pytest.fixture
def jev_env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(settings, "JEV_CACHE_PATH", str(tmp_path / "cache.json"))


@pytest.fixture(autouse=True)
def reset_review_refinement_metrics():
    metrics.reset()
    yield


def make_text_fake(*, keywords=_AD_KW, input_tokens=1200, output_tokens=6, raises=False):
    """Fetcher that scores s<sid> high when that line's text names an ad keyword.

    It reads the state transcript ("L0000| text") so it does not depend on the
    proxy's internal sid assignment.
    """

    def fake(payload: dict[str, Any], *, url: str, api_key: str, timeout: float, max_retries: int = 2, **_kwargs: Any) -> dict[str, Any]:
        if raises:
            raise RuntimeError("upstream boom")
        sid_text: dict[str, str] = {}
        for line in payload["state"]["transcript"].splitlines():
            tag, _, body = line.partition("| ")
            sid_text[f"s{int(tag[1:])}"] = body.lower()
        answers: dict[str, Any] = {}
        for key in payload["questions"]:
            if key == "evidence":
                answers[key] = {"noul": 0.98 if any(kw in text for text in sid_text.values() for kw in _AD_KW) else 0.02}
                continue
            question = payload["questions"][key]
            criteria = question.get("criteria", {})
            if question.get("type") == "choice":
                option = next(name for name in criteria if name != "unknown")
                remainder = 0.01 / max(len(criteria) - 1, 1)
                answers[key] = {
                    "choice": option,
                    "confidence": 0.98,
                    "probabilities": {name: 0.99 if name == option else remainder for name in criteria},
                }
                continue
            if not key.startswith("s"):
                continue
            text = sid_text.get(key, "")
            hot = any(kw in text for kw in keywords)
            answers[key] = {"noul": 0.98 if hot else 0.02}
        return {
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }

    return fake


def _ts_lines(rows):
    return "\n".join(f"[{s:.1f}s-{e:.1f}s] {t}" for s, e, t in rows)


def build_review_prompt(cand_start, cand_end, before, ad, after, *, pool="accepted",
                        with_timestamps=True):
    """Mirror ad_reviewer._build_user_prompt output shape."""
    if pool == "resurrection":
        framing = (
            "This segment was rejected for low confidence by the validator. "
            "Decide whether it should be cut after all.\n"
            f"Original boundaries: {cand_start:.2f}s - {cand_end:.2f}s.\n"
        )
    else:
        framing = (
            "This is the candidate ad to review.\n"
            f"Original boundaries: {cand_start:.2f}s - {cand_end:.2f}s.\n"
        )
    render = _ts_lines if with_timestamps else (lambda rows: "\n".join(t for _, _, t in rows))
    return (
        "Podcast: My Podcast\n"
        "Episode: Ep 1\n"
        "\n"
        f"{framing}\n"
        "Transcript (60s before, the candidate ad, 60s after; all lines "
        "carry [start-end] second timestamps):\n"
        f"{render(before)}\n"
        f">>> CANDIDATE AD START [{cand_start:.1f}s] >>>\n"
        f"{render(ad)}\n"
        f"<<< CANDIDATE AD END [{cand_end:.1f}s] <<<\n"
        f"{render(after)}\n"
    )


def minuspod_review_verdict(content, original_start, original_end, *, pool="accepted", tol=0.1):
    """Faithful copy of ad_reviewer._review_single's decision over a parsed
    review response, driven by MinusPod's own extract_json_ads_array."""
    ads, method = extract_json_ads_array(content)
    assert ads is not None, "MinusPod could not parse the review response"
    if not ads:  # empty array -> reject (ad_reviewer.py:1246-1258)
        return "reject", None, None, method
    kept = ads[0]
    assert isinstance(kept, dict)
    if kept.get("is_ad") is False:  # structured whole-span reject (1281-1293)
        return "reject", None, None, method
    new_start = float(kept.get("start", original_start))
    new_end = float(kept.get("end", original_end))
    if pool == "resurrection":  # resurrection pool always resurrects (1330-1331)
        return "resurrect", new_start, new_end, method
    unchanged = abs(new_start - original_start) <= tol and abs(new_end - original_end) <= tol
    return ("confirmed" if unchanged else "adjust"), new_start, new_end, method


def _run(
    prompt,
    fake,
    tmp_path,
    *,
    refine_boundaries=False,
    review_request_id=None,
    enter=0.95,
    review_evidence_enter=None,
    review_choice_enter=None,
):
    return run_review(
        messages=[
            {"role": "system", "content": "review ads"},
            {"role": "user", "content": prompt},
        ],
        request_model="typesafe/jev",
        url="u",
        api_key="k",
        timeout=1.0,
        cache_path=str(tmp_path / "c.json"),
        model="jev-latest",
        enter=enter,
        stay=0.40,
        refine_boundaries=refine_boundaries,
        review_evidence_enter=review_evidence_enter,
        review_choice_enter=review_choice_enter,
        review_request_id=review_request_id,
        fetcher=fake,
    )


def test_review_evidence_threshold_is_independent(jev_env, tmp_path):
    prompt = build_review_prompt(
        100.0,
        120.0,
        [(94.0, 100.0, "back to the topic")],
        [(100.0, 110.0, "This episode is sponsored by BetterHelp"),
         (110.0, 120.0, "Use promo code SHOW")],
        [(120.0, 126.0, "and we are back")],
    )
    with pytest.raises(ReviewInconclusiveError, match="sufficient advertising evidence"):
        _run(
            prompt,
            make_text_fake(),
            tmp_path,
            review_evidence_enter=0.99,
        )


def test_review_choice_threshold_is_independent(jev_env, tmp_path):
    prompt = build_review_prompt(
        100.0,
        120.0,
        [(94.0, 100.0, "back to the topic")],
        [(100.0, 110.0, "This episode is sponsored by BetterHelp"),
         (110.0, 120.0, "Use promo code SHOW")],
        [(120.0, 126.0, "and we are back")],
    )
    with pytest.raises(ReviewInconclusiveError, match="boundary Choice"):
        _run(
            _with_word_edges(prompt),
            make_text_fake(),
            tmp_path,
            refine_boundaries=True,
            review_choice_enter=0.99,
        )


def test_lower_evidence_threshold_reuses_cached_review_answer(jev_env, tmp_path):
    prompt = build_review_prompt(
        100.0,
        120.0,
        [(94.0, 100.0, "back to the topic")],
        [(100.0, 110.0, "This episode is sponsored by BetterHelp"),
         (110.0, 120.0, "Use promo code SHOW")],
        [(120.0, 126.0, "and we are back")],
    )
    calls = []
    base = make_text_fake()

    def fake(payload, **kwargs):
        calls.append(payload["questions"])
        body = base(payload, **kwargs)
        if "evidence" in body["answers"]:
            body["answers"]["evidence"]["noul"] = 0.86
        return body

    with pytest.raises(ReviewInconclusiveError, match="sufficient advertising evidence"):
        _run(prompt, fake, tmp_path, review_evidence_enter=0.95)
    assert len(calls) == 2

    response = _run(prompt, fake, tmp_path, review_evidence_enter=0.85)
    assert json.loads(response["choices"][0]["message"]["content"])["ads"]
    assert len(calls) == 2


def _with_word_edges(prompt, start=(99.5, 100.0, "This"), end=(119.5, 120.5, "BetterHelp")):
    return prompt + (
        "Boundary word timing, use these timestamps for corrections:\n"
        "Start edge:\n"
        f"[{start[0]:.1f}s-{start[1]:.1f}s] {start[2]}\n"
        "End edge:\n"
        f"[{end[0]:.1f}s-{end[1]:.1f}s] {end[2]}\n"
    )


# ---------------- discriminator ----------------


def test_review_prompt_is_recognized():
    prompt = build_review_prompt(
        100.0, 120.0,
        [(94.0, 100.0, "so anyway back to the topic")],
        [(100.0, 110.0, "This episode is sponsored by BetterHelp"),
         (110.0, 120.0, "Use promo code SHOW at betterhelp.com")],
        [(120.0, 126.0, "and we are back")],
    )
    assert is_review_request(prompt) is True
    assert parse_candidate_bounds(prompt) == (100.0, 120.0)


def test_detection_window_is_not_a_review():
    lines = [
        "[0.0s - 6.0s] Welcome back to the show.",
        "[6.0s - 12.0s] This episode is sponsored by BetterHelp.",
    ]
    prompt = format_window_prompt("Pod", "Ep", "", lines, 0, 1, 0.0, 600.0)
    assert is_review_request(prompt) is False


def test_review_system_prompt_routes_without_markers():
    # No CANDIDATE markers in the user text; only the system prompt signals review.
    user = "[100.0s - 110.0s] This episode is sponsored by BetterHelp."
    assert ">>> CANDIDATE AD START [" not in user
    review_system = (
        "You are reviewing a candidate advertisement that has already been "
        "detected in a podcast episode."
    )
    resurrect_system = (
        "You are taking a second look at a segment that the validator already "
        "rejected for low confidence."
    )
    assert is_review_request(user, review_system) is True
    assert is_review_request(user, resurrect_system) is True
    assert is_review_request(user, review_system.upper()) is True  # case-insensitive
    assert is_review_request(user) is False  # no markers, no system signal


def test_detection_prompt_never_routes_to_review():
    from minuspod_compat import get_static_system_prompt

    lines = ["[0.0s - 6.0s] Welcome back to the show.", "[6.0s - 12.0s] Today we discuss hiking."]
    user = format_window_prompt("Pod", "Ep", "", lines, 0, 1, 0.0, 600.0)
    system = get_static_system_prompt()  # opens "Analyze this podcast transcript..."
    assert is_review_request(user, system) is False


def test_dedupe_overlapping_context_and_candidate_lines():
    # A straddling line rendered in both context and candidate is scored once.
    prompt = build_review_prompt(
        100.0, 120.0,
        [(96.0, 100.0, "let's take a quick break"),
         (100.0, 110.0, "This episode is sponsored by BetterHelp")],
        [(100.0, 110.0, "This episode is sponsored by BetterHelp"),
         (110.0, 120.0, "Use promo code SHOW at betterhelp.com")],
        [(120.0, 126.0, "and we are back")],
    )
    segs = parse_review_segments(prompt)
    starts = [s["start"] for s in segs]
    assert starts == sorted(starts)
    assert len(segs) == len({(s["start"], s["end"], s["text"]) for s in segs})
    assert [s["sid"] for s in segs] == list(range(len(segs)))


# ---------------- verdicts, verified against MinusPod's parser ----------------


def test_confirm_keeps_candidate_with_jev_boundaries(jev_env, tmp_path):
    prompt = build_review_prompt(
        100.0, 120.0,
        [(94.0, 100.0, "so anyway back to the topic")],
        [(100.0, 110.0, "This episode is sponsored by BetterHelp"),
         (110.0, 120.0, "Use promo code SHOW at betterhelp.com")],
        [(120.0, 126.0, "and we are back to the conversation")],
    )
    resp = _run(prompt, make_text_fake(), tmp_path)
    content = resp["choices"][0]["message"]["content"]
    ad = json.loads(content)["ads"][0]
    assert ad["is_ad"] is True
    assert (ad["start"], ad["end"]) == (100.0, 120.0)  # Jev span edges == candidate

    verdict, start, end, method = minuspod_review_verdict(content, 100.0, 120.0)
    assert verdict == "confirmed"
    assert (start, end) == (100.0, 120.0)
    assert method == "json_object_ads_key"


def test_adjust_when_jev_span_extends_into_context(jev_env, tmp_path):
    # The ad actually starts at 94.0 (a context line reads promotional), so the
    # Jev span extends past the candidate start -> a boundary adjustment.
    prompt = build_review_prompt(
        100.0, 120.0,
        [(88.0, 94.0, "so anyway back to the topic"),
         (94.0, 100.0, "and now a word from our sponsor")],
        [(100.0, 110.0, "This episode is sponsored by BetterHelp"),
         (110.0, 120.0, "Use promo code SHOW at betterhelp.com")],
        [(120.0, 126.0, "and we are back to the conversation")],
    )
    resp = _run(prompt, make_text_fake(), tmp_path)
    content = resp["choices"][0]["message"]["content"]
    verdict, start, end, _ = minuspod_review_verdict(content, 100.0, 120.0)
    assert verdict == "adjust"
    assert start == 94.0 and end == 120.0


def test_reject_when_jev_finds_no_ad(jev_env, tmp_path):
    prompt = build_review_prompt(
        50.0, 70.0,
        [(45.0, 50.0, "Have you followed the antitrust case?")],
        [(50.0, 62.0, "The DOJ argued Apple's policies harm developers"),
         (62.0, 70.0, "They want the court to change the commission")],
        [(70.0, 74.0, "what is your take on the remedies?")],
    )
    resp = _run(prompt, make_text_fake(), tmp_path)
    content = resp["choices"][0]["message"]["content"]
    assert json.loads(content) == {"ads": []}
    verdict, start, end, _ = minuspod_review_verdict(content, 50.0, 70.0)
    assert verdict == "reject"


def test_resurrection_hit_resurrects_with_ad_signal(jev_env, tmp_path):
    prompt = build_review_prompt(
        666.7, 674.3,
        [(660.0, 666.7, "so that's our take on the case")],
        [(666.7, 674.3, "Hosted on Acast. See acast dot com slash privacy")],
        [(674.3, 680.0, "welcome back, today we talk about the launch")],
        pool="resurrection",
    )
    resp = _run(prompt, make_text_fake(), tmp_path)
    content = resp["choices"][0]["message"]["content"]
    verdict, start, end, _ = minuspod_review_verdict(content, 666.7, 674.3, pool="resurrection")
    assert verdict == "resurrect"
    assert json.loads(content)["ads"][0]["is_ad"] is True


def test_resurrection_miss_keeps_rejected(jev_env, tmp_path):
    prompt = build_review_prompt(
        200.0, 215.0,
        [(195.0, 200.0, "we've been talking about the privacy framework")],
        [(200.0, 215.0, "Apple says the framework gives users more control")],
        [(215.0, 220.0, "but critics argue Apple has too much power")],
        pool="resurrection",
    )
    resp = _run(prompt, make_text_fake(), tmp_path)
    content = resp["choices"][0]["message"]["content"]
    assert json.loads(content) == {"ads": []}
    verdict, *_ = minuspod_review_verdict(content, 200.0, 215.0, pool="resurrection")
    assert verdict == "reject"  # keep-rejected: no content cut


def test_ambiguous_overlapping_spans_are_unavailable(jev_env, tmp_path):
    prompt = build_review_prompt(
        100.0,
        200.0,
        [(94.0, 100.0, "context before")],
        [
            (100.0, 105.0, "This is sponsored by BetterHelp"),
            (105.0, 155.0, "The hosts return to their discussion"),
            (155.0, 200.0, "Use promo code SHOW at betterhelp.com"),
        ],
        [(200.0, 206.0, "context after")],
    )
    with pytest.raises(ReviewUnavailableError, match="unambiguous"):
        _run(prompt, make_text_fake(), tmp_path)


def test_high_score_without_advertising_cue_is_unavailable(jev_env, tmp_path):
    prompt = build_review_prompt(
        100.0,
        110.0,
        [(94.0, 100.0, "context before")],
        [(100.0, 110.0, "This unrelated editorial sentence has no promotion")],
        [(110.0, 116.0, "context after")],
    )
    with pytest.raises(ReviewUnavailableError, match="sufficient advertising evidence"):
        _run(prompt, make_text_fake(keywords=("unrelated",)), tmp_path)


# ---------------- safe degrade ----------------


def test_degrade_no_transcript_is_unavailable(jev_env, tmp_path):
    # Markers present (routes to review) but no [start-end] lines to score.
    prompt = build_review_prompt(
        100.0, 120.0,
        [(94.0, 100.0, "context before")],
        [(100.0, 120.0, "This episode is sponsored by BetterHelp")],
        [(120.0, 126.0, "context after")],
        with_timestamps=False,
    )
    assert is_review_request(prompt) is True
    assert parse_review_segments(prompt) == []
    with pytest.raises(ReviewUnavailableError, match="could not be parsed"):
        _run(prompt, make_text_fake(), tmp_path)


def test_degrade_no_transcript_resurrection_is_unavailable(jev_env, tmp_path):
    prompt = build_review_prompt(
        100.0, 120.0,
        [(94.0, 100.0, "context before")],
        [(100.0, 120.0, "This episode is sponsored by BetterHelp")],
        [(120.0, 126.0, "context after")],
        pool="resurrection",
        with_timestamps=False,
    )
    with pytest.raises(ReviewUnavailableError, match="could not be parsed"):
        _run(prompt, make_text_fake(), tmp_path)


async def test_review_routes_through_the_endpoint(jev_env, client, monkeypatch):
    import app.services.jev as jev

    monkeypatch.setattr(jev, "call_payload", make_text_fake())
    prompt = build_review_prompt(
        100.0, 120.0,
        [(94.0, 100.0, "so anyway back to the topic")],
        [(100.0, 110.0, "This episode is sponsored by BetterHelp"),
         (110.0, 120.0, "Use promo code SHOW at betterhelp.com")],
        [(120.0, 126.0, "and we are back to the conversation")],
    )
    body = {
        "model": "typesafe/jev",
        "messages": [
            {"role": "system", "content": "You are reviewing a candidate advertisement."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    resp = await client.post(
        "/v1/chat/completions", json=body, headers={"Authorization": "Bearer test-key"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "typesafe/jev"
    content = data["choices"][0]["message"]["content"]
    verdict, start, end, method = minuspod_review_verdict(content, 100.0, 120.0)
    assert verdict == "confirmed"
    assert (start, end) == (100.0, 120.0)
    assert method == "json_object_ads_key"


async def test_review_failure_returns_503(jev_env, client, monkeypatch):
    import app.services.jev as jev

    monkeypatch.setattr(jev, "call_payload", make_text_fake(raises=True))
    prompt = build_review_prompt(
        100.0,
        120.0,
        [(94.0, 100.0, "context before")],
        [(100.0, 110.0, "This episode is sponsored by BetterHelp")],
        [
            (110.0, 120.0, "Use promo code SHOW at betterhelp.com"),
            (120.0, 126.0, "context after"),
        ],
    )
    response = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": prompt}]},
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "jev_review_upstream_failure"
    assert "x-should-retry" not in response.headers


async def test_inconclusive_review_returns_non_retryable_422(jev_env, client, monkeypatch):
    import app.services.jev as jev

    monkeypatch.setattr(jev, "call_payload", make_text_fake())
    prompt = build_review_prompt(
        100.0, 200.0,
        [(94.0, 100.0, "context before")],
        [(100.0, 105.0, "This is sponsored by BetterHelp"),
         (105.0, 155.0, "The hosts return to their discussion"),
         (155.0, 200.0, "Use promo code SHOW at betterhelp.com")],
        [(200.0, 206.0, "context after")],
    )
    response = await client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": prompt}]},
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 422
    assert response.headers["x-should-retry"] == "false"
    assert response.headers["x-request-id"]
    assert response.json()["error"]["code"] == "jev_review_inconclusive"


def test_word_timing_is_not_mixed_into_coarse_review_segments():
    prompt = build_review_prompt(
        100.0, 120.0,
        [(94.0, 100.0, "context before")],
        [(100.0, 120.0, "This episode is sponsored by BetterHelp")],
        [(120.0, 126.0, "context after")],
    ) + (
        "Boundary word timing, use these timestamps for corrections:\n"
        "Start edge:\n[99.9s-100.0s] This\n"
        "End edge:\n[120.0s-120.0s] end\n"
    )
    segments, words = parse_review_context(prompt)
    assert all(segment["text"] not in {"This", "end"} for segment in segments)
    assert words["start"][0]["text"] == "This"
    assert words["end"][0]["end"] == 120.0


def test_opt_in_refinement_uses_word_edges(jev_env, tmp_path):
    prompt = build_review_prompt(
        100.0, 120.0,
        [(94.0, 100.0, "context before")],
        [(100.0, 120.0, "This episode is sponsored by BetterHelp")],
        [(120.0, 126.0, "context after")],
    ) + (
        "Boundary word timing, use these timestamps for corrections:\n"
        "Start edge:\n[99.5s-100.0s] This\n"
        "End edge:\n[119.5s-120.5s] BetterHelp\n"
    )
    response = _run(prompt, make_text_fake(), tmp_path, refine_boundaries=True)
    ad = json.loads(response["choices"][0]["message"]["content"])["ads"][0]
    assert (ad["start"], ad["end"]) == (99.5, 120.5)


def test_refinement_metrics_changed_and_unchanged(jev_env, tmp_path):
    base = build_review_prompt(
        100.0,
        120.0,
        [(94.0, 100.0, "context before")],
        [(100.0, 120.0, "This episode is sponsored by BetterHelp")],
        [(120.0, 126.0, "context after")],
    )
    _run(_with_word_edges(base), make_text_fake(), tmp_path, refine_boundaries=True)
    _run(
        _with_word_edges(base, start=(100.0, 100.0, "This"), end=(120.0, 120.0, "BetterHelp")),
        make_text_fake(),
        tmp_path,
        refine_boundaries=True,
    )

    refinement = metrics.snapshot()["review"]["refinement"]
    assert refinement["attempted"] == 2
    assert refinement["completed"] == 2
    assert refinement["changed"] == 1
    assert refinement["unchanged"] == 1


def test_refinement_gates_never_enter_choice(jev_env, tmp_path):
    base = build_review_prompt(
        100.0,
        120.0,
        [(94.0, 100.0, "context before")],
        [(100.0, 120.0, "This episode is sponsored by BetterHelp")],
        [(120.0, 126.0, "context after")],
    )
    _run(base, make_text_fake(), tmp_path, refine_boundaries=False)
    _run(base, make_text_fake(), tmp_path, refine_boundaries=True)

    low_evidence = build_review_prompt(
        100.0,
        120.0,
        [(94.0, 100.0, "context before")],
        [(100.0, 120.0, "This unrelated editorial sentence")],
        [(120.0, 126.0, "context after")],
    )
    with pytest.raises(ReviewInconclusiveError):
        _run(_with_word_edges(low_evidence), make_text_fake(keywords=("unrelated",)), tmp_path, refine_boundaries=True)

    outside = build_review_prompt(
        100.0,
        120.0,
        [(88.0, 94.0, "context before")],
        [(100.0, 120.0, "editorial content")],
        [(126.0, 132.0, "context after")],
    )
    outside_cache = tmp_path / "outside"
    outside_cache.mkdir()
    _run(outside, make_text_fake(keywords=("context",)), outside_cache, refine_boundaries=True)

    ambiguous = build_review_prompt(
        100.0,
        200.0,
        [(94.0, 100.0, "context before")],
        [
            (100.0, 105.0, "This is sponsored by BetterHelp"),
            (105.0, 155.0, "The hosts return to their discussion"),
            (155.0, 200.0, "Use promo code SHOW at betterhelp.com"),
        ],
        [(200.0, 206.0, "context after")],
    )
    ambiguous_cache = tmp_path / "ambiguous"
    ambiguous_cache.mkdir()
    with pytest.raises(ReviewInconclusiveError):
        _run(ambiguous, make_text_fake(), ambiguous_cache, refine_boundaries=True)

    refinement = metrics.snapshot()["review"]["refinement"]
    assert refinement["attempted"] == 0
    assert refinement["skipped"] == {
        "disabled": 1,
        "missing_word_timings": 1,
        "insufficient_evidence": 1,
        "ambiguous_spans": 1,
        "no_overlapping_span": 1,
    }


def test_refinement_choice_failures_are_counted(jev_env, tmp_path):
    base = build_review_prompt(
        100.0,
        120.0,
        [(94.0, 100.0, "context before")],
        [(100.0, 120.0, "This episode is sponsored by BetterHelp")],
        [(120.0, 126.0, "context after")],
    )
    prompt = _with_word_edges(base)
    normal = make_text_fake()

    def uncertain(payload, **kwargs):
        result = normal(payload, **kwargs)
        if any(question.get("type") == "choice" for question in payload["questions"].values()):
            for answer in result["answers"].values():
                answer["confidence"] = 0.5
        return result

    with pytest.raises(ReviewInconclusiveError):
        uncertain_cache = tmp_path / "uncertain"
        uncertain_cache.mkdir()
        _run(prompt, uncertain, uncertain_cache, refine_boundaries=True)

    def upstream_choice(payload, **kwargs):
        if any(question.get("type") == "choice" for question in payload["questions"].values()):
            raise RuntimeError("choice upstream failure")
        return normal(payload, **kwargs)

    with pytest.raises(ReviewUnavailableError):
        upstream_cache = tmp_path / "upstream"
        upstream_cache.mkdir()
        _run(prompt, upstream_choice, upstream_cache, refine_boundaries=True)

    def malformed_choice(payload, **kwargs):
        result = normal(payload, **kwargs)
        if any(question.get("type") == "choice" for question in payload["questions"].values()):
            result["answers"][next(iter(result["answers"]))]["probabilities"] = {"bad": 1.0}
        return result

    with pytest.raises(ReviewUnavailableError):
        malformed_cache = tmp_path / "malformed"
        malformed_cache.mkdir()
        _run(prompt, malformed_choice, malformed_cache, refine_boundaries=True)

    invalid = _with_word_edges(base, end=(90.0, 95.0, "wrong"))
    with pytest.raises(ReviewInconclusiveError):
        invalid_cache = tmp_path / "invalid"
        invalid_cache.mkdir()
        _run(invalid, normal, invalid_cache, refine_boundaries=True)

    refinement = metrics.snapshot()["review"]["refinement"]
    assert refinement["attempted"] == 4
    assert refinement["inconclusive"] == 2
    assert refinement["upstream_error"] == 2


@pytest.mark.parametrize("mode", ["low_confidence", "unknown"])
def test_inconclusive_start_choice_does_not_request_end(jev_env, tmp_path, mode):
    base = build_review_prompt(
        100.0,
        120.0,
        [(94.0, 100.0, "context before")],
        [(100.0, 120.0, "This episode is sponsored by BetterHelp")],
        [(120.0, 126.0, "context after")],
    )
    normal = make_text_fake()
    stages = []

    def fake(payload, **kwargs):
        choice_questions = [
            key for key, question in payload["questions"].items() if question.get("type") == "choice"
        ]
        if not choice_questions:
            return normal(payload, **kwargs)
        stage = choice_questions[0]
        stages.append(stage)
        if stage.startswith("end_"):
            raise AssertionError("end Choice must not run after an inconclusive start")
        result = normal(payload, **kwargs)
        for key in choice_questions:
            answer = result["answers"][key]
            if mode == "low_confidence":
                answer["confidence"] = 0.5
            else:
                criteria = payload["questions"][key]["criteria"]
                remainder = 0.01 / (len(criteria) - 1)
                answer["choice"] = "unknown"
                answer["confidence"] = 0.98
                answer["probabilities"] = {
                    option: 0.99 if option == "unknown" else remainder
                    for option in criteria
                }
        return result

    with pytest.raises(ReviewInconclusiveError):
        _run(_with_word_edges(base), fake, tmp_path, refine_boundaries=True)

    assert stages == ["start_0"]
    refinement = metrics.snapshot()["review"]["refinement"]
    assert refinement["attempted"] == 1
    assert refinement["inconclusive"] == 1
    assert refinement["upstream_error"] == 0


async def test_inconclusive_start_choice_returns_422_without_end_request(
    jev_env, client, monkeypatch
):
    import app.services.jev as jev

    monkeypatch.setattr(settings, "JEV_REVIEW_REFINE_BOUNDARIES", True)
    normal = make_text_fake()
    stages = []

    def fake(payload, **kwargs):
        choice_questions = [
            key for key, question in payload["questions"].items() if question.get("type") == "choice"
        ]
        if not choice_questions:
            return normal(payload, **kwargs)
        stage = choice_questions[0]
        stages.append(stage)
        if stage.startswith("end_"):
            raise RuntimeError("end Choice must not run after an inconclusive start")
        result = normal(payload, **kwargs)
        for answer in result["answers"].values():
            answer["confidence"] = 0.5
        return result

    monkeypatch.setattr(jev, "call_payload", fake)
    prompt = build_review_prompt(
        100.0,
        120.0,
        [(94.0, 100.0, "context before")],
        [(100.0, 120.0, "This episode is sponsored by BetterHelp")],
        [(120.0, 126.0, "context after")],
    )
    response = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": _with_word_edges(prompt)}]},
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "jev_review_inconclusive"
    assert stages == ["start_0"]


async def test_valid_start_choice_still_requests_end_and_preserves_503(
    jev_env, client, monkeypatch
):
    import app.services.jev as jev

    monkeypatch.setattr(settings, "JEV_REVIEW_REFINE_BOUNDARIES", True)
    normal = make_text_fake()
    stages = []

    def fake(payload, **kwargs):
        choice_questions = [
            key for key, question in payload["questions"].items() if question.get("type") == "choice"
        ]
        if choice_questions:
            stages.append(choice_questions[0])
            if choice_questions[0].startswith("end_"):
                raise RuntimeError("end Choice upstream failure")
        return normal(payload, **kwargs)

    monkeypatch.setattr(jev, "call_payload", fake)
    prompt = build_review_prompt(
        100.0,
        120.0,
        [(94.0, 100.0, "context before")],
        [(100.0, 120.0, "This episode is sponsored by BetterHelp")],
        [(120.0, 126.0, "context after")],
    )
    response = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": _with_word_edges(prompt)}]},
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "jev_review_upstream_failure"
    assert stages == ["start_0", "end_0"]


async def test_malformed_choice_returns_503_with_safe_validation_diagnostic(
    jev_env, client, monkeypatch, caplog
):
    import logging

    import app.services.jev as jev

    monkeypatch.setattr(settings, "JEV_REVIEW_REFINE_BOUNDARIES", True)
    normal = make_text_fake()

    def malformed(payload, **kwargs):
        result = normal(payload, **kwargs)
        if any(question.get("type") == "choice" for question in payload["questions"].values()):
            result["answers"][next(iter(result["answers"]))]["probabilities"] = {"bad": 1.0}
        return result

    monkeypatch.setattr(jev, "call_payload", malformed)
    prompt = build_review_prompt(
        100.0,
        120.0,
        [(94.0, 100.0, "context before")],
        [(100.0, 120.0, "This episode is sponsored by BetterHelp")],
        [(120.0, 126.0, "context after")],
    )
    with caplog.at_level(logging.WARNING, logger="app.services.openai_adapter"):
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": _with_word_edges(prompt)}]},
            headers={"Authorization": "Bearer test-key"},
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "jev_review_upstream_invalid_response"
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "stage=choice_start validation_failed reason=choice_probability_keys" in message
        for message in messages
    )
    assert all("bad" not in message for message in messages)


async def test_foreign_validation_error_uses_generic_safe_diagnostic(
    jev_env, client, monkeypatch, caplog
):
    import logging

    import app.services.jev as jev

    monkeypatch.setattr(settings, "JEV_REVIEW_REFINE_BOUNDARIES", True)
    normal = make_text_fake()

    def invalid(payload, **kwargs):
        if any(question.get("type") == "choice" for question in payload["questions"].values()):
            raise ValueError("sensitive upstream response")
        return normal(payload, **kwargs)

    monkeypatch.setattr(jev, "call_payload", invalid)
    prompt = build_review_prompt(
        100.0,
        120.0,
        [(94.0, 100.0, "context before")],
        [(100.0, 120.0, "This episode is sponsored by BetterHelp")],
        [(120.0, 126.0, "context after")],
    )
    with caplog.at_level(logging.WARNING, logger="app.services.openai_adapter"):
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": _with_word_edges(prompt)}]},
            headers={"Authorization": "Bearer test-key"},
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "jev_review_upstream_invalid_response"
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "stage=choice_start validation_failed reason=invalid_response details={}" in message
        for message in messages
    )
    assert all("sensitive upstream response" not in message for message in messages)


def test_refinement_logs_effective_context_and_precise_thresholds(jev_env, tmp_path, caplog):
    import logging

    prompt = build_review_prompt(
        100.0,
        120.0,
        [(94.0, 100.0, "context before")],
        [(100.0, 120.0, "This episode is sponsored by BetterHelp")],
        [(120.0, 126.0, "context after")],
    )
    with caplog.at_level(logging.INFO, logger="app.services.openai_adapter"):
        _run(
            _with_word_edges(prompt),
            make_text_fake(),
            tmp_path,
            refine_boundaries=True,
            review_request_id="request-123",
            enter=0.9496,
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any("request_id=request-123 stage=context" in message and "start_word_count=1" in message for message in messages)
    assert any("stage=evidence score=0.98 threshold=0.9496" in message for message in messages)
    assert any("stage=choice_start" in message and "confidence=0.98" in message for message in messages)
    assert all("BetterHelp" not in message for message in messages)


def test_review_upstream_failure_is_unavailable(jev_env, tmp_path):
    prompt = build_review_prompt(
        100.0, 120.0,
        [(94.0, 100.0, "context before")],
        [(100.0, 110.0, "This episode is sponsored by BetterHelp"),
         (110.0, 120.0, "Use promo code SHOW at betterhelp.com")],
        [(120.0, 126.0, "context after")],
    )
    with pytest.raises(ReviewUnavailableError, match="Jev review request failed"):
        _run(prompt, make_text_fake(raises=True), tmp_path)
