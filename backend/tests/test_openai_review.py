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
    is_review_request,
    parse_candidate_bounds,
    parse_review_segments,
    run_review,
)
from minuspod_compat import extract_json_ads_array, format_window_prompt

_AD_KW = ("sponsor", "betterhelp", "promo code", "brought to you by", "acast")


@pytest.fixture
def jev_env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(settings, "JEV_CACHE_PATH", str(tmp_path / "cache.json"))


def make_text_fake(*, keywords=_AD_KW, input_tokens=1200, output_tokens=6, raises=False):
    """Fetcher that scores s<sid> high when that line's text names an ad keyword.

    It reads the state transcript ("L0000| text") so it does not depend on the
    proxy's internal sid assignment.
    """

    def fake(payload: dict[str, Any], *, url: str, api_key: str, timeout: float) -> dict[str, Any]:
        if raises:
            raise RuntimeError("upstream boom")
        sid_text: dict[str, str] = {}
        for line in payload["state"]["transcript"].splitlines():
            tag, _, body = line.partition("| ")
            sid_text[f"s{int(tag[1:])}"] = body.lower()
        answers: dict[str, Any] = {}
        for key in payload["questions"]:
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


def _run(prompt, fake, tmp_path):
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
        enter=0.95,
        stay=0.40,
        fetcher=fake,
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


# ---------------- safe degrade ----------------


def test_degrade_no_transcript_confirms_original_accepted(jev_env, tmp_path):
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
    resp = _run(prompt, make_text_fake(), tmp_path)
    content = resp["choices"][0]["message"]["content"]
    verdict, start, end, _ = minuspod_review_verdict(content, 100.0, 120.0)
    assert verdict == "confirmed"  # original span left untouched
    assert (start, end) == (100.0, 120.0)


def test_degrade_no_transcript_keeps_rejected_resurrection(jev_env, tmp_path):
    prompt = build_review_prompt(
        100.0, 120.0,
        [(94.0, 100.0, "context before")],
        [(100.0, 120.0, "This episode is sponsored by BetterHelp")],
        [(120.0, 126.0, "context after")],
        pool="resurrection",
        with_timestamps=False,
    )
    resp = _run(prompt, make_text_fake(), tmp_path)
    content = resp["choices"][0]["message"]["content"]
    assert json.loads(content) == {"ads": []}


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
    resp = await client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "typesafe/jev"
    content = data["choices"][0]["message"]["content"]
    verdict, start, end, method = minuspod_review_verdict(content, 100.0, 120.0)
    assert verdict == "confirmed"
    assert (start, end) == (100.0, 120.0)
    assert method == "json_object_ads_key"


def test_degrade_on_upstream_failure_does_not_crash(jev_env, tmp_path):
    prompt = build_review_prompt(
        100.0, 120.0,
        [(94.0, 100.0, "context before")],
        [(100.0, 110.0, "This episode is sponsored by BetterHelp"),
         (110.0, 120.0, "Use promo code SHOW at betterhelp.com")],
        [(120.0, 126.0, "context after")],
    )
    resp = _run(prompt, make_text_fake(raises=True), tmp_path)
    content = resp["choices"][0]["message"]["content"]
    verdict, start, end, _ = minuspod_review_verdict(content, 100.0, 120.0)
    assert verdict == "confirmed"  # Jev failure -> no-change, not a crash
    assert (start, end) == (100.0, 120.0)
