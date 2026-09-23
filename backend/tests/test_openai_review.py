"""Tests for the ad-review route of the OpenAI-compatible adapter (no network).

The review verdicts emitted here are round-tripped through MinusPod's own
review-response parser (minuspod_compat.extract_json_ads_array, the same call
ad_reviewer._review_single makes) plus a faithful copy of its verdict decision
(ad_reviewer.py:1281-1345), so a shape drift fails the test.
"""

import json
from typing import Any

import app.services.jev as jev
import httpx
import pytest
from app.config import settings
from app.services.openai_adapter import (
    ReviewInconclusiveError,
    ReviewUnavailableError,
    _boundary_candidates,
    _pair_questions,
    _shortlist_boundaries,
    _valid_pairs,
    is_review_request,
    parse_candidate_bounds,
    parse_review_context,
    parse_review_segments,
    run_review,
)
from app.utils.metrics import metrics
from minuspod_compat import extract_json_ads_array, format_window_prompt

_AD_KW = ("sponsor", "betterhelp", "promo code", "brought to you by", "acast")


def _upstream_status_error(
    status: int, retry_after: str = "7"
) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.example/v1/systemone")
    response = httpx.Response(
        status, request=request, headers={"Retry-After": retry_after}
    )
    return httpx.HTTPStatusError("upstream failure", request=request, response=response)


@pytest.fixture
def jev_env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(settings, "JEV_CACHE_PATH", str(tmp_path / "cache.json"))


@pytest.fixture(autouse=True)
def reset_review_refinement_metrics():
    metrics.reset()
    yield


def test_pair_candidates_mix_meaningful_inward_and_outward_edges():
    segments = [
        {"start": 90.0, "end": 100.0, "text": "Editorial."},
        {"start": 100.0, "end": 120.0, "text": "Sponsor message."},
        {"start": 120.0, "end": 130.0, "text": "Editorial return."},
    ]
    words = {
        "start": [
            {"start": 90.0, "end": 91.0, "text": "Editorial."},
            {"start": 100.0, "end": 101.0, "text": "Sponsor.\""},
            {"start": 106.0, "end": 107.0, "text": "message"},
        ],
        "end": [
            {"start": 111.0, "end": 112.0, "text": "Offer"},
            {"start": 114.0, "end": 115.0, "text": "ends."},
            {"start": 120.0, "end": 121.0, "text": "Return"},
            {"start": 129.0, "end": 130.0, "text": "Editorial."},
        ],
    }

    starts, ends = _boundary_candidates(segments, words, (100.0, 120.0))
    candidate_pairs = _valid_pairs(starts, ends, (100.0, 120.0), (90.0, 130.0), 90.0, 130.0)
    questions, pairs = _pair_questions(words, (100.0, 120.0), candidate_pairs)

    assert (100.0, 120.0) in pairs.values()
    assert (90.0, 115.0) in pairs.values()
    assert (106.0, 130.0) in pairs.values()
    assert len(set(pairs.values())) == len(pairs)
    assert all(
        end > start
        and min(end, 120.0) > max(start, 100.0)
        and min(end, 130.0) > max(start, 90.0)
        and 90.0 <= start < end <= 130.0
        for start, end in pairs.values()
    )
    assert set(questions["boundary_pair"]["criteria"]) == {"unknown", *pairs}


def test_pair_candidates_keep_only_when_no_meaningful_timed_edge_exists():
    segments = [{"start": 94.0, "end": 126.0, "text": "Sponsor message"}]
    words = {
        "start": [{"start": 99.5, "end": 100.0, "text": "Sponsor"}],
        "end": [{"start": 119.5, "end": 120.5, "text": "message"}],
    }

    starts, ends = _boundary_candidates(segments, words, (100.0, 120.0))
    _, pairs = _pair_questions(
        words,
        (100.0, 120.0),
        _valid_pairs(starts, ends, (100.0, 120.0), (100.0, 120.0), 94.0, 126.0),
    )

    assert pairs == {"pair_00": (100.0, 120.0)}


def test_boundary_candidates_snap_two_second_grid_within_thirty_seconds():
    segments = [{"start": 90.0, "end": 170.0, "text": "context"}]
    words = {
        "start": [
            {"start": value, "end": value + 0.1, "text": f"w{index}"}
            for index, value in enumerate((100.0, 101.1, 109.9, 129.8, 130.1, 131.0, 161.0))
        ],
        "end": [
            {"start": value - 0.1, "end": value, "text": f"w{index}"}
            for index, value in enumerate((89.0, 90.0, 90.2, 110.1, 118.9, 120.0))
        ],
    }

    starts, ends = _boundary_candidates(segments, words, (100.0, 120.0))

    assert 100.0 in starts and 130.1 not in starts and 161.0 not in starts
    assert 120.0 in ends and 89.0 not in ends
    assert len(starts) == len(set(starts)) and len(ends) == len(set(ends))


def test_shortlist_uses_two_positive_scores_and_retains_original():
    values = {"start_00": 90.0, "start_01": 100.0, "start_02": 106.0, "start_03": 110.0}
    answer = {
        "choice": "start_02",
        "confidence": 0.2,
        "probabilities": {"unknown": 0.4, "start_00": 0.1, "start_01": 0.0, "start_02": 0.3, "start_03": 0.2},
    }

    assert _shortlist_boundaries(answer, values, 100.0) == [100.0, 106.0, 110.0]


def test_no_valid_pair_is_inconclusive_without_choice_request(jev_env, tmp_path):
    prompt = build_review_prompt(
        90.0,
        110.0,
        [],
        [(94.0, 100.0, "This episode is sponsored by BetterHelp")],
        [(100.0, 120.0, "context after")],
    ) + (
        "Boundary word timing, use these timestamps for corrections:\n"
        "Start edge:\n[90.0s-90.0s] candidate\n"
        "End edge:\n[109.0s-110.0s] candidate\n"
    )
    normal = make_text_fake()
    choice_requests = 0

    def fake(payload, **kwargs):
        nonlocal choice_requests
        if any(question.get("type") == "choice" for question in payload["questions"].values()):
            choice_requests += 1
        return normal(payload, **kwargs)

    with pytest.raises(ReviewInconclusiveError, match="no valid pair"):
        _run(prompt, fake, tmp_path, refine_boundaries=True)

    assert choice_requests == 0
    refinement = metrics.snapshot()["review"]["refinement"]
    assert refinement["attempted"] == 0
    assert refinement["inconclusive"] == 0
    assert refinement["upstream_error"] == 0


def _meaningful_pair_prompt() -> str:
    return build_review_prompt(
        100.0,
        120.0,
        [(90.0, 100.0, "Editorial before.")],
        [(100.0, 120.0, "This episode is sponsored by BetterHelp")],
        [(120.0, 130.0, "Editorial return.")],
    ) + (
        "Boundary word timing, use these timestamps for corrections:\n"
        "Start edge:\n[90.0s-91.0s] Editorial.\n[100.0s-101.0s] Sponsor.\n[106.0s-107.0s] Offer\n"
        "End edge:\n[111.0s-112.0s] Offer\n[114.0s-115.0s] ends.\n[129.0s-130.0s] Editorial.\n"
    )


def _select_pair_fake(start: float, end: float):
    normal = make_text_fake()

    def fake(payload, **kwargs):
        result = normal(payload, **kwargs)
        for key, question in payload["questions"].items():
            if question.get("type") != "choice":
                continue
            if key == "boundary_pair":
                option = next(
                    name
                    for name, description in question["criteria"].items()
                    if description.startswith(f"Ad {start:.2f}s-{end:.2f}s")
                )
            else:
                target = start if key == "boundary_start" else end
                option = next(name for name, description in question["criteria"].items() if f"{target:.2f}s" in description)
            result["answers"][key]["choice"] = option
            result["answers"][key]["probabilities"] = {
                name: 0.99 if name == option else 0.01 / (len(question["criteria"]) - 1)
                for name in question["criteria"]
            }
        return result

    return fake


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
                candidate = payload["state"].get("candidate", {})
                if key == "boundary_start":
                    option = next(name for name, text in criteria.items() if f"{candidate.get('start'):.2f}s" in text)
                elif key == "boundary_end":
                    option = next(name for name, text in criteria.items() if f"{candidate.get('end'):.2f}s" in text)
                elif key == "boundary_pair":
                    option = next(
                        name for name, text in criteria.items()
                        if text.startswith(f"Ad {candidate.get('start'):.2f}s-{candidate.get('end'):.2f}s")
                    )
                else:
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
    kwargs = {}
    if fake is not None:
        kwargs["fetcher"] = fake
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
        **kwargs,
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


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code", "retry_after"),
    [
        (_upstream_status_error(429), 429, "jev_upstream_rate_limited", "7"),
        (_upstream_status_error(401), 401, "jev_upstream_authentication_error", None),
        (_upstream_status_error(403), 403, "jev_upstream_authentication_error", None),
        (httpx.ReadTimeout("timed out"), 504, "jev_upstream_timeout", None),
    ],
)
async def test_review_detection_preserves_upstream_errors(
    jev_env,
    client,
    monkeypatch,
    error,
    expected_status,
    expected_code,
    retry_after,
):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(jev, "call_payload", fail)
    prompt = build_review_prompt(
        100.0,
        120.0,
        [(94.0, 100.0, "context before")],
        [(100.0, 110.0, "This episode is sponsored by BetterHelp")],
        [(110.0, 120.0, "Use promo code SHOW"), (120.0, 126.0, "context after")],
    )

    response = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": prompt}]},
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == expected_status
    assert response.json()["error"]["code"] == expected_code
    assert response.headers.get("x-request-id")
    if retry_after is None:
        assert "retry-after" not in response.headers
    else:
        assert response.headers["retry-after"] == retry_after


async def test_review_boundary_preserves_upstream_status_and_retry_after(
    jev_env, client, monkeypatch
):
    monkeypatch.setattr(settings, "JEV_REVIEW_REFINE_BOUNDARIES", True)
    normal = make_text_fake()
    error = _upstream_status_error(503, "11")

    def fake(payload, **kwargs):
        if any(
            question.get("type") == "choice"
            for question in payload["questions"].values()
        ):
            raise error
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
    assert response.json()["error"]["code"] == "jev_upstream_failure"
    assert response.headers["retry-after"] == "11"
    assert response.headers.get("x-request-id")
    assert metrics.snapshot()["review"]["refinement"]["upstream_error"] == 1


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


@pytest.mark.parametrize(
    ("selected", "expected"),
    [
        ((106.0, 120.0), (106.0, 120.0)),
        ((100.0, 115.0), (100.0, 115.0)),
        ((90.0, 120.0), (90.0, 120.0)),
        ((100.0, 130.0), (100.0, 130.0)),
        ((90.0, 115.0), (90.0, 115.0)),
    ],
)
def test_opt_in_refinement_applies_selected_meaningful_pair(jev_env, tmp_path, selected, expected):
    response = _run(
        _meaningful_pair_prompt(),
        _select_pair_fake(*selected),
        tmp_path,
        refine_boundaries=True,
    )
    ad = json.loads(response["choices"][0]["message"]["content"])["ads"][0]
    assert (ad["start"], ad["end"]) == expected
    verdict, start, end, _ = minuspod_review_verdict(
        response["choices"][0]["message"]["content"], 100.0, 120.0
    )
    assert verdict == "adjust"
    assert (start, end) == expected


def test_refinement_metrics_changed_and_unchanged(jev_env, tmp_path):
    prompt = _meaningful_pair_prompt()
    _run(prompt, _select_pair_fake(90.0, 115.0), tmp_path, refine_boundaries=True)
    keep_cache = tmp_path / "keep"
    keep_cache.mkdir()
    _run(prompt, _select_pair_fake(100.0, 120.0), keep_cache, refine_boundaries=True)

    refinement = metrics.snapshot()["review"]["refinement"]
    assert refinement["attempted"] == 2
    assert refinement["completed"] == 2
    assert refinement["changed"] == 1
    assert refinement["unchanged"] == 1


def test_refinement_rank_and_pair_use_warm_cache(jev_env, tmp_path):
    calls = 0
    base = _select_pair_fake(100.0, 120.0)

    def fake(payload, **kwargs):
        nonlocal calls
        calls += 1
        return base(payload, **kwargs)

    first = _run(_meaningful_pair_prompt(), fake, tmp_path, refine_boundaries=True)
    second = _run(_meaningful_pair_prompt(), fake, tmp_path, refine_boundaries=True)

    assert calls == 4
    assert first["choices"][0]["message"]["content"] == second["choices"][0]["message"]["content"]


def test_refinement_stages_share_one_deadline(jev_env, tmp_path, monkeypatch):
    deadlines = []
    base = make_text_fake()

    def fake(payload, **kwargs):
        deadlines.append(kwargs["deadline_at"])
        return base(payload, **kwargs)

    monkeypatch.setattr(jev, "call_payload", fake)
    _run(_meaningful_pair_prompt(), None, tmp_path, refine_boundaries=True)

    assert len(deadlines) == 4
    assert len(set(deadlines)) == 1


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
        "no_valid_pairs": 0,
        "transcript_gap": 0,
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

    refinement = metrics.snapshot()["review"]["refinement"]
    assert refinement["attempted"] == 3
    assert refinement["inconclusive"] == 1
    assert refinement["upstream_error"] == 2


def test_unknown_rank_boundary_is_inconclusive_before_pair_choice(jev_env, tmp_path):
    prompt = _meaningful_pair_prompt()
    normal = make_text_fake()
    stages = []

    def fake(payload, **kwargs):
        result = normal(payload, **kwargs)
        if "boundary_start" not in payload["questions"]:
            return result
        stages.append(tuple(payload["questions"]))
        criteria = payload["questions"]["boundary_start"]["criteria"]
        result["answers"]["boundary_start"] = {
            "choice": "unknown",
            "confidence": 0.98,
            "probabilities": {
                option: 0.99 if option == "unknown" else 0.01 / (len(criteria) - 1)
                for option in criteria
            },
        }
        return result

    with pytest.raises(ReviewInconclusiveError, match="Choice ranking"):
        _run(prompt, fake, tmp_path, refine_boundaries=True)

    assert stages == [("boundary_start", "boundary_end")]
    refinement = metrics.snapshot()["review"]["refinement"]
    assert refinement["attempted"] == 1 and refinement["inconclusive"] == 1


async def test_post_rank_empty_pairs_are_inconclusive(jev_env, client, monkeypatch, tmp_path):
    import app.services.openai_adapter as adapter

    monkeypatch.setattr(settings, "JEV_REVIEW_REFINE_BOUNDARIES", True)
    monkeypatch.setattr(settings, "JEV_CACHE_PATH", str(tmp_path / "post-rank-cache.json"))
    monkeypatch.setattr(jev, "call_payload", make_text_fake())

    def inverted_shortlist(_answer, _values, original):
        return [110.0] if original == 100.0 else [100.0]

    monkeypatch.setattr(adapter, "_shortlist_boundaries", inverted_shortlist)
    prompt = _meaningful_pair_prompt() + (
        "Boundary word timing, use these timestamps for corrections:\n"
        "Start edge:\n[110.0s-111.0s] Later\n"
        "End edge:\n[99.0s-100.0s] Earlier\n"
    )
    response = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": prompt}]},
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "jev_review_inconclusive"
    refinement = metrics.snapshot()["review"]["refinement"]
    assert refinement["attempted"] == 1 and refinement["inconclusive"] == 1
    assert refinement["skipped"]["no_valid_pairs"] == 0
    assert metrics.snapshot()["review"]["reasons"]["no_valid_pairs"] == 1


@pytest.mark.parametrize("mode", ["low_confidence", "unknown"])
def test_inconclusive_boundary_pair_follows_ranking(jev_env, tmp_path, mode):
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
        result = normal(payload, **kwargs)
        if stage != "boundary_pair":
            return result
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

    assert stages == ["boundary_start", "boundary_pair"]
    refinement = metrics.snapshot()["review"]["refinement"]
    assert refinement["attempted"] == 1
    assert refinement["inconclusive"] == 1
    assert refinement["upstream_error"] == 0


async def test_inconclusive_boundary_pair_returns_422(
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
        result = normal(payload, **kwargs)
        if stage != "boundary_pair":
            return result
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
    assert stages == ["boundary_start", "boundary_pair"]


async def test_boundary_pair_upstream_failure_preserves_503(
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
        if choice_questions and choice_questions[0] == "boundary_pair":
            stages.append(choice_questions[0])
            raise RuntimeError("boundary pair upstream failure")
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
    assert stages == ["boundary_pair"]


async def test_malformed_choice_returns_503_with_safe_validation_diagnostic(
    jev_env, client, monkeypatch, caplog
):
    import logging

    import app.services.jev as jev

    monkeypatch.setattr(settings, "JEV_REVIEW_REFINE_BOUNDARIES", True)
    normal = make_text_fake()

    def malformed(payload, **kwargs):
        result = normal(payload, **kwargs)
        if "boundary_pair" in payload["questions"]:
            result["answers"]["boundary_pair"]["probabilities"] = {"bad": 1.0}
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
        "stage=choice_pair validation_failed reason=choice_probability_keys" in message
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
        if "boundary_pair" in payload["questions"]:
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
        "stage=choice_pair validation_failed reason=invalid_response details={}" in message
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
    assert any("stage=choice_pair" in message and "confidence=0.98" in message for message in messages)
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
