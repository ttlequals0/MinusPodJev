"""
Jev System One proxy logic: payload building, answer parsing, cost estimate.

Ported from the bench reference and restructured into the service layer.
One request carries the whole transcript window: state is sent once, each
segment gets one short noul question keyed s<sid>, and all questions are
evaluated in parallel against the shared state.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import httpx

from app.utils.cache import JsonCache
from app.utils.spans import spans_from_probabilities

# $ per million input tokens (docs.typesafe.ai/models). Output is not billed.
INPUT_COST_PER_MTOK = 0.042

# Defining "advertising" once in the shared state keeps a 300-segment window
# affordable: each per-question instruction is then a single short sentence.
GUIDANCE = (
    "Each line of `transcript` is one segment of a podcast episode, prefixed "
    "with its line id. A line is ADVERTISING when it is a sponsor read, a "
    "produced ad spot, a dynamically inserted ad, a hosting-platform pre-roll "
    "or post-roll, a cross-promotion for another show, or a produced segment "
    "asking listeners to subscribe, rate, or follow. Signs of advertising: a "
    "sponsor or brand name, a URL, a promo code, a product pitch, a call to "
    "action, or concentrated marketing copy that is tonally separate from the "
    "conversation. A line is EDITORIAL CONTENT when it is the host or a guest "
    "discussing the episode's subject. A guest talking about their own book or "
    "project, and the host mentioning their own show or Patreon in passing "
    "during conversation, are editorial content, not advertising."
)

NOUL_INSTRUCTIONS = "Line {lid} of `transcript` is advertising, not editorial content."


def line_id(sid: int) -> str:
    return f"L{sid:04d}"


def build_state(segments: Sequence[dict[str, Any]]) -> str:
    """Window transcript as ID-prefixed numbered lines."""
    return "\n".join(
        f"{line_id(int(seg['sid']))}| {str(seg.get('text', '')).strip()}" for seg in segments
    )


def build_questions(segments: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """One noul question per segment, keyed s<sid>.

    No per-question criteria: the definition lives in the shared state.
    """
    return {
        f"s{seg['sid']}": {
            "type": "noul",
            "instructions": NOUL_INSTRUCTIONS.format(lid=line_id(int(seg["sid"]))),
        }
        for seg in segments
    }


def build_payload(
    segments: Sequence[dict[str, Any]], *, uid: str | None = None
) -> dict[str, Any]:
    """One request covering a whole window.

    No model field: System One selects the model, and omitting it keeps the
    payload (and the cache key) independent of any caller-supplied model id.
    """
    state: dict[str, Any] = {"guidance": GUIDANCE, "transcript": build_state(segments)}
    if uid is not None:
        state["uid"] = uid
    return {"state": state, "questions": build_questions(segments)}


def parse_response(body: dict[str, Any]) -> dict[str, Any]:
    """Normalize an upstream reply into cache-entry shape."""
    answers = body.get("answers") or {}
    probs: dict[str, float] = {}
    for key, ans in answers.items():
        if not key.startswith("s") or not isinstance(ans, dict):
            continue
        value = ans.get("noul")
        if isinstance(value, (int, float)):
            probs[key] = float(value)
    usage = body.get("usage") or {}
    return {
        "probabilities": probs,
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
    }


def call_payload(
    payload: dict[str, Any],
    *,
    url: str,
    api_key: str,
    timeout: float = 60.0,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """POST the payload to the upstream endpoint and return the decoded body."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if client is not None:
        resp = client.post(url, json=payload, headers=headers)
    else:
        with httpx.Client(timeout=timeout) as own_client:
            resp = own_client.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    result: dict[str, Any] = resp.json()
    return result


def estimate_cost_usd(input_tokens: int) -> float:
    """Input tokens at INPUT_COST_PER_MTOK per Mtok; output is not billed."""
    return round(input_tokens / 1_000_000 * INPUT_COST_PER_MTOK, 6)


def jev_ask(
    segments: Sequence[dict[str, Any]],
    *,
    url: str,
    api_key: str,
    timeout: float,
    cache_path: str,
    model: str,
    uid: str | None = None,
    enter: float | None = None,
    stay: float | None = None,
    fetcher: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the payload, serve from cache or upstream, and shape the reply."""
    payload = build_payload(segments, uid=uid)
    cache = JsonCache(cache_path)
    send = fetcher or call_payload

    def _fetch() -> dict[str, Any]:
        body = send(payload, url=url, api_key=api_key, timeout=timeout)
        return parse_response(body)

    entry, hit = cache.get_or_fetch(payload, _fetch)
    probabilities: dict[str, float] = entry["probabilities"]
    input_tokens = int(entry["input_tokens"])
    return {
        "model": model,
        "cache_hit": hit,
        "probabilities": probabilities,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": int(entry["output_tokens"]),
        },
        "estimated_cost_usd": estimate_cost_usd(input_tokens),
        "spans": spans_from_probabilities(segments, probabilities, enter=enter, stay=stay),
    }


# --- Category second pass -------------------------------------------------
# One upstream call per assembled span: a single state carrying the span's
# segments plus a little context, and one noul per category. The argmax noul
# is the span's category. Keyed c<index> so it never collides with the s<sid>
# detection questions, and cached through the same JsonCache.

CATEGORY_GUIDANCE = (
    "Each line of `transcript` is one segment around a single advertising "
    "break in a podcast, prefixed with its line id. The lines named in "
    "`focus` are the break itself; the others are nearby context. Decide what "
    "KIND of break the focus lines are."
)

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "sponsor": "a paid sponsor read or product advertisement for an outside advertiser",
    "cross_promo": "a cross-promotion for another podcast or show",
    "self_promo": "the host promoting their own show, Patreon, merch, membership, or back catalog",
    "interaction": "a call to action to like, subscribe, rate, review, follow, or comment",
    "intro": "an intro segment opening the episode",
    "outro": "an outro segment closing the episode",
    "recap": "a recap or summary of earlier content",
}

CATEGORY_NOUL = "The advertising break at lines {ids} is {desc}."


def _focus_ids(focus: Sequence[dict[str, Any]]) -> str:
    ordered = sorted(focus, key=lambda s: int(s["sid"]))
    lo, hi = int(ordered[0]["sid"]), int(ordered[-1]["sid"])
    return line_id(lo) if lo == hi else f"{line_id(lo)}-{line_id(hi)}"


def build_category_payload(
    focus: Sequence[dict[str, Any]],
    context: Sequence[dict[str, Any]],
    categories: Sequence[str],
    *,
    uid: str | None = None,
) -> dict[str, Any]:
    """One request: span + context state, one noul per category keyed c<index>."""
    merged = {int(s["sid"]): s for s in [*context, *focus]}
    ordered = [merged[k] for k in sorted(merged)]
    ids = _focus_ids(focus)
    state: dict[str, Any] = {
        "guidance": CATEGORY_GUIDANCE,
        "transcript": build_state(ordered),
        "focus": ids,
    }
    if uid is not None:
        state["uid"] = uid
    questions = {
        f"c{i}": {
            "type": "noul",
            "instructions": CATEGORY_NOUL.format(
                ids=ids, desc=CATEGORY_DESCRIPTIONS.get(cat, cat)
            ),
        }
        for i, cat in enumerate(categories)
    }
    return {"state": state, "questions": questions}


def parse_category_response(body: dict[str, Any]) -> dict[str, Any]:
    """Keep the c<index> noul answers and usage from an upstream reply."""
    answers = body.get("answers") or {}
    probs: dict[str, float] = {}
    for key, ans in answers.items():
        if not key.startswith("c") or not isinstance(ans, dict):
            continue
        value = ans.get("noul")
        if isinstance(value, (int, float)):
            probs[key] = float(value)
    usage = body.get("usage") or {}
    return {
        "probabilities": probs,
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
    }


def jev_category(
    focus: Sequence[dict[str, Any]],
    context: Sequence[dict[str, Any]],
    categories: Sequence[str],
    *,
    url: str,
    api_key: str,
    timeout: float,
    cache_path: str,
    uid: str | None = None,
    fetcher: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Argmax category for one span, cached and shaped like jev_ask."""
    payload = build_category_payload(focus, context, categories, uid=uid)
    cache = JsonCache(cache_path)
    send = fetcher or call_payload

    def _fetch() -> dict[str, Any]:
        body = send(payload, url=url, api_key=api_key, timeout=timeout)
        return parse_category_response(body)

    entry, hit = cache.get_or_fetch(payload, _fetch)
    probs: dict[str, float] = entry["probabilities"]
    best_cat = categories[0] if categories else ""
    best_p = -1.0
    for i, cat in enumerate(categories):
        p = probs.get(f"c{i}", 0.0)
        if p > best_p:
            best_p, best_cat = p, cat
    return {
        "category": best_cat,
        "confidence": max(best_p, 0.0),
        "cache_hit": hit,
        "probabilities": probs,
        "usage": {
            "input_tokens": int(entry["input_tokens"]),
            "output_tokens": int(entry["output_tokens"]),
        },
    }
