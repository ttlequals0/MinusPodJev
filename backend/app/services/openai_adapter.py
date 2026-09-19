"""OpenAI chat-completions adapter over the Jev detection path.

MinusPod points a stock OpenAI client at this proxy. The user message carries a
transcript rendered by MinusPod's format_window_prompt; we parse its segment
lines, run the Jev detection + category passes, match sponsors, and emit the
ads JSON MinusPod's parse_ads_from_response reads, wrapped in a chat.completion
envelope with the ads JSON as message.content (a string).
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Callable, Sequence
from typing import Any

from minuspod_compat import SEGMENT_CATEGORIES, SPONSOR_PRIORITY_FIELDS

from app.services.jev import jev_ask, jev_category
from app.services.sponsors import sponsor_for_span

logger = logging.getLogger(__name__)

# timestamps mode: [123.4s - 130.0s] some text
_TS_LINE = re.compile(r"^\s*\[(\d+(?:\.\d+)?)s\s*-\s*(\d+(?:\.\d+)?)s\]\s?(.*)$")
# segment_ids mode: [12] some text
_ID_LINE = re.compile(r"^\s*\[(\d+)\]\s?(.*)$")

# Candidate markers: ad_reviewer._build_user_prompt wraps every review candidate in
# these; detection never emits them, so they tell a review request from detection.
_REVIEW_START_MARK = ">>> CANDIDATE AD START ["
_REVIEW_END_MARK = "<<< CANDIDATE AD END ["
_REVIEW_START_RE = re.compile(r">>> CANDIDATE AD START \[(\d+(?:\.\d+)?)s\] >>>")
_REVIEW_END_RE = re.compile(r"<<< CANDIDATE AD END \[(\d+(?:\.\d+)?)s\] <<<")
# ad_reviewer._build_user_prompt framing: "Original boundaries: {start:.2f}s - {end:.2f}s."
_REVIEW_BOUNDS_RE = re.compile(
    r"Original boundaries:\s*(\d+(?:\.\d+)?)s\s*-\s*(\d+(?:\.\d+)?)s"
)
# resurrection-pool framing (ad_reviewer._build_user_prompt). Its degrade path
# must keep the segment rejected, not confirm a cut.
_REVIEW_RESURRECT_MARK = "rejected for low confidence"

# Fallback signal (case-insensitive): opening lines of DEFAULT_REVIEW_PROMPT /
# DEFAULT_RESURRECT_PROMPT, in case the candidate markers ever change. Detection
# opens "Analyze this podcast transcript..." and matches neither.
_REVIEW_SYSTEM_SIGNATURES = (
    "reviewing a candidate advertisement that has already been detected",
    "taking a second look at a segment that the validator already rejected",
)


def extract_user_text(messages: Sequence[dict[str, Any]]) -> str:
    """The last user-role message content, list-of-parts flattened to text."""
    for msg in reversed(list(messages)):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type", "text") == "text"
            )
    return ""


def extract_system_text(messages: Sequence[dict[str, Any]]) -> str:
    """All system-role message contents, list-of-parts flattened, joined."""
    parts: list[str] = []
    for msg in messages:
        if msg.get("role") != "system":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.append(
                "\n".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict) and part.get("type", "text") == "text"
                )
            )
    return "\n".join(parts)


def parse_transcript(text: str) -> tuple[list[dict[str, Any]], str]:
    """Parse transcript lines into segments and the detected addressing mode.

    Timestamps mode wins when any timestamped line is present; sid is assigned
    by 0-based order. Otherwise segment_ids mode uses the bracket id. Non-line
    text (header, rules, podcast/description) is ignored. Empty -> ([], "empty").
    """
    ts: list[dict[str, Any]] = []
    ids: list[dict[str, Any]] = []
    for line in text.splitlines():
        m = _TS_LINE.match(line)
        if m:
            ts.append(
                {"start": float(m.group(1)), "end": float(m.group(2)), "text": m.group(3).strip()}
            )
            continue
        m = _ID_LINE.match(line)
        if m:
            ids.append({"sid": int(m.group(1)), "text": m.group(2).strip()})
    if ts:
        for i, seg in enumerate(ts):
            seg["sid"] = i
        return ts, "timestamps"
    if ids:
        return ids, "segment_ids"
    return [], "empty"


def _members(ordered: list[dict[str, Any]], start_id: int, end_id: int) -> list[dict[str, Any]]:
    idx = {int(s["sid"]): i for i, s in enumerate(ordered)}
    return ordered[idx[start_id] : idx[end_id] + 1]


def _empty_envelope(model: str) -> dict[str, Any]:
    return _envelope(model, {"ads": []}, {"input_tokens": 0, "output_tokens": 0})


def _envelope(model: str, content: dict[str, Any], usage: dict[str, int]) -> dict[str, Any]:
    prompt_tokens = int(usage["input_tokens"])
    completion_tokens = int(usage["output_tokens"])
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": json.dumps(content)},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def run_chat_completion(
    *,
    messages: Sequence[dict[str, Any]],
    request_model: str | None,
    url: str,
    api_key: str,
    timeout: float,
    cache_path: str,
    model: str,
    enter: float,
    stay: float,
    category_pass: bool,
    category_context: int,
    default_category: str,
    uid: str | None = None,
    fetcher: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Parse the prompt, detect ad spans, classify them, and build the envelope."""
    echo_model = request_model or model
    user_text = extract_user_text(messages)
    system_text = extract_system_text(messages)
    if is_review_request(user_text, system_text):
        return run_review(
            messages=messages,
            request_model=request_model,
            url=url,
            api_key=api_key,
            timeout=timeout,
            cache_path=cache_path,
            model=model,
            enter=enter,
            stay=stay,
            uid=uid,
            fetcher=fetcher,
        )
    segments, mode = parse_transcript(user_text)
    if not segments:
        logger.warning("chat_completions: no transcript lines parsed; returning empty ads")
        return _empty_envelope(echo_model)

    detection = jev_ask(
        segments,
        url=url,
        api_key=api_key,
        timeout=timeout,
        cache_path=cache_path,
        model=model,
        uid=uid,
        enter=enter,
        stay=stay,
        fetcher=fetcher,
    )
    probabilities: dict[str, float] = detection["probabilities"]
    input_tokens = int(detection["usage"]["input_tokens"])
    output_tokens = int(detection["usage"]["output_tokens"])
    logger.info("detection: %d segments -> %d spans", len(segments), len(detection["spans"]))

    ordered = sorted(segments, key=lambda s: int(s["sid"]))
    categories = list(SEGMENT_CATEGORIES)
    ads: list[dict[str, Any]] = []

    for span in detection["spans"]:
        start_id, end_id = int(span["start_id"]), int(span["end_id"])
        members = _members(ordered, start_id, end_id)
        n = len(members)
        k = sum(1 for m in members if probabilities.get(f"s{int(m['sid'])}", 0.0) >= enter)
        span_text = " ".join(str(m.get("text", "")) for m in members)

        if category_pass:
            lo_i = ordered.index(members[0])
            hi_i = ordered.index(members[-1])
            context = ordered[max(0, lo_i - category_context) : lo_i] + ordered[
                hi_i + 1 : hi_i + 1 + category_context
            ]
            cat_result = jev_category(
                members,
                context,
                categories,
                url=url,
                api_key=api_key,
                timeout=timeout,
                cache_path=cache_path,
                uid=uid,
                fetcher=fetcher,
            )
            category = cat_result["category"]
            input_tokens += int(cat_result["usage"]["input_tokens"])
            output_tokens += int(cat_result["usage"]["output_tokens"])
        else:
            category = default_category

        ad: dict[str, Any] = {}
        if mode == "timestamps":
            ad["start"] = float(members[0]["start"])
            ad["end"] = float(members[-1]["end"])
        else:
            ad["start_id"] = start_id
            ad["end_id"] = end_id
        ad["category"] = category
        ad["confidence"] = float(span["confidence"])
        # "ad" keeps mentions_advertising() true so MinusPod's evidence gate
        # never drops a long span that named no sponsor.
        ad["reason"] = f"jev ad: {k}/{n} segments >= enter"
        ad["end_text"] = str(members[-1].get("text", ""))
        # Always populate sponsor: real name if matched, else a unique jev-<7char>.
        # An empty field lets MinusPod mint one from the reason and cluster unnamed ads.
        sponsor = sponsor_for_span(span_text)
        ad[SPONSOR_PRIORITY_FIELDS[0]] = sponsor
        logger.debug("sponsor: %s", sponsor)
        ads.append(ad)

    return _envelope(
        echo_model, {"ads": ads}, {"input_tokens": input_tokens, "output_tokens": output_tokens}
    )


# --- Review route --------------------------------------------------------
# ad_reviewer sends one candidate ad (CANDIDATE markers + context). We run Jev
# over the window and answer with its ads-wrapped schema (ad_reviewer.py:62-83):
# empty array = reject, one {is_ad,start,end,confidence,reason} = keep. MinusPod
# reads the verdict from the boundary delta; a non-empty resurrection answer
# resurrects.


def is_review_request(text: str, system_text: str = "") -> bool:
    """Whether the request is an ad-review prompt, not a detection window.

    Primary signal: the code-generated CANDIDATE markers in the user text.
    Fallback: a review/resurrect signature in the system prompt, so a marker
    change still routes correctly. Detection is neither.
    """
    if _REVIEW_START_MARK in text and _REVIEW_END_MARK in text:
        return True
    lowered = system_text.lower()
    return any(sig in lowered for sig in _REVIEW_SYSTEM_SIGNATURES)


def _review_pool(text: str) -> str:
    """"resurrection" when the framing names a validator rejection, else "accepted"."""
    return "resurrection" if _REVIEW_RESURRECT_MARK in text else "accepted"


def parse_candidate_bounds(text: str) -> tuple[float, float] | None:
    """Candidate [start, end] from the CANDIDATE markers, else the framing line."""
    start_m = _REVIEW_START_RE.search(text)
    end_m = _REVIEW_END_RE.search(text)
    if start_m and end_m:
        return float(start_m.group(1)), float(end_m.group(1))
    bounds_m = _REVIEW_BOUNDS_RE.search(text)
    if bounds_m:
        return float(bounds_m.group(1)), float(bounds_m.group(2))
    return None


def parse_review_segments(text: str) -> list[dict[str, Any]]:
    """Unique timestamped transcript segments from a review prompt, sid by order.

    get_timestamped_transcript_for_range renders overlapping segments in both the
    context and candidate sections, so the same line can appear twice; dedupe on
    (start, end, text) and reassign sids so Jev scores each segment once.
    """
    seen: set[tuple[float, float, str]] = set()
    segs: list[dict[str, Any]] = []
    for line in text.splitlines():
        m = _TS_LINE.match(line)
        if not m:
            continue
        start, end, body = float(m.group(1)), float(m.group(2)), m.group(3).strip()
        key = (start, end, body)
        if key in seen:
            continue
        seen.add(key)
        segs.append({"start": start, "end": end, "text": body})
    segs.sort(key=lambda s: (s["start"], s["end"]))
    for i, seg in enumerate(segs):
        seg["sid"] = i
    return segs


def _review_reject_envelope(model: str) -> dict[str, Any]:
    """Empty ads array: reject a candidate / keep a rejected segment out."""
    return _empty_envelope(model)


def _review_degrade(model: str, pool: str, cand: tuple[float, float] | None) -> dict[str, Any]:
    """Schema-valid no-change verdict for a review Jev cannot answer.

    Accepted pool with known bounds: confirm the original span so MinusPod's ad
    is left exactly as detected. Resurrection pool (or unknown bounds): reject,
    which keeps the segment out of the cut and never deletes extra content.
    """
    if pool == "accepted" and cand is not None:
        start, end = cand
        verdict = {
            "is_ad": True,
            "start": start,
            "end": end,
            "confidence": 0.5,
            "reason": "jev review unavailable; keeping original boundaries",
        }
        return _envelope(model, {"ads": [verdict]}, {"input_tokens": 0, "output_tokens": 0})
    return _review_reject_envelope(model)


def run_review(
    *,
    messages: Sequence[dict[str, Any]],
    request_model: str | None,
    url: str,
    api_key: str,
    timeout: float,
    cache_path: str,
    model: str,
    enter: float,
    stay: float,
    uid: str | None = None,
    fetcher: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Review one candidate ad with Jev and emit the ads-wrapped review verdict."""
    echo_model = request_model or model
    text = extract_user_text(messages)
    pool = _review_pool(text)
    cand = parse_candidate_bounds(text)
    segments = parse_review_segments(text)
    if cand is None or not segments:
        logger.warning("review: no candidate bounds/transcript parsed; degrading (pool=%s)", pool)
        return _review_degrade(echo_model, pool, cand)

    cand_start, cand_end = cand
    try:
        detection = jev_ask(
            segments,
            url=url,
            api_key=api_key,
            timeout=timeout,
            cache_path=cache_path,
            model=model,
            uid=uid,
            enter=enter,
            stay=stay,
            fetcher=fetcher,
        )
    except Exception as exc:  # noqa: BLE001 - never crash the review; degrade safely
        logger.warning("review: jev call failed (%s); degrading (pool=%s)", exc, pool)
        return _review_degrade(echo_model, pool, cand)

    probabilities: dict[str, float] = detection["probabilities"]
    usage = detection["usage"]
    by_sid = {int(s["sid"]): s for s in segments}

    # The candidate is an ad when a Jev span overlaps its region; pick the span
    # with the most overlap and take its segment-edge bounds.
    best_bounds: tuple[float, float, float] | None = None
    best_overlap = 0.0
    for span in detection["spans"]:
        span_start = float(by_sid[int(span["start_id"])]["start"])
        span_end = float(by_sid[int(span["end_id"])]["end"])
        overlap = min(span_end, cand_end) - max(span_start, cand_start)
        if overlap > 0.0 and overlap > best_overlap:
            best_overlap = overlap
            best_bounds = (span_start, span_end, float(span["confidence"]))

    if best_bounds is None:
        # No ad signal in the candidate region: reject / keep-rejected.
        logger.info("review: verdict=reject (no overlapping span, pool=%s)", pool)
        return _envelope(echo_model, {"ads": []}, usage)

    ad_start, ad_end, confidence = best_bounds
    cand_segs = [
        s for s in segments if float(s["end"]) > cand_start and float(s["start"]) < cand_end
    ]
    k = sum(1 for s in cand_segs if probabilities.get(f"s{int(s['sid'])}", 0.0) >= enter)
    n = len(cand_segs)
    verdict = {
        "is_ad": True,
        "start": ad_start,
        "end": ad_end,
        "confidence": confidence,
        "reason": f"jev review: candidate is advertising ({k}/{n} segments >= enter)",
    }
    logger.info("review: verdict=keep %.1fs-%.1fs (pool=%s)", ad_start, ad_end, pool)
    return _envelope(echo_model, {"ads": [verdict]}, usage)
