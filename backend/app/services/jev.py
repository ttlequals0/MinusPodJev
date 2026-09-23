"""
Jev System One proxy logic: payload building, answer parsing, cost estimate.

Ported from the bench reference and restructured into the service layer.
One request carries the whole transcript window: state is sent once, each
segment gets one short noul question keyed s<sid>, and all questions are
evaluated in parallel against the shared state.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from app.utils.cache import JsonCache
from app.utils.metrics import metrics
from app.utils.spans import spans_from_probabilities

logger = logging.getLogger(__name__)


class JevReviewValidationError(ValueError):
    """Safe, machine-readable validation failure for a review response."""

    code = "jev_upstream_invalid_response"
    _RULES = frozenset(
        {
            "response_object",
            "answers_object",
            "answers_keys",
            "answer_object",
            "evidence_probability",
            "choice_criteria_object",
            "choice_criteria_empty",
            "choice_answer_shape",
            "choice_confidence",
            "choice_probability",
            "choice_probability_keys",
            "choice_probability_sum",
            "choice_winner",
            "usage_object",
            "usage_input_tokens",
            "usage_output_tokens",
        }
    )
    _DETAILS = frozenset(
        {
            "expected_count",
            "actual_count",
            "actual",
            "expected_total",
            "actual_total",
            "tolerance",
            "selected_probability",
            "max_probability",
        }
    )

    def __init__(self, rule: str, numeric_details: dict[str, int | float] | None = None):
        self.rule = rule if rule in self._RULES else "response_object"
        self.numeric_details = self._safe_details(numeric_details or {})
        super().__init__(f"upstream review response validation failed: {self.rule}")

    @classmethod
    def _safe_details(cls, details: dict[str, int | float]) -> dict[str, int | float]:
        safe: dict[str, int | float] = {}
        for name, value in details.items():
            if name not in cls._DETAILS:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if isinstance(value, int):
                safe[name] = value
                continue
            try:
                numeric = float(value)
            except (OverflowError, TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                safe[name] = numeric
        return safe


class JevCategoryValidationError(JevReviewValidationError):
    """Safe validation failure for a category Choice response."""


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None

# $ per million input tokens (docs.typesafe.ai/models). Output is not billed.
INPUT_COST_PER_MTOK = 0.042
# Accept observed serialized Choice totals down to 0.99.
CHOICE_PROBABILITY_SUM_TOLERANCE = 0.01
CHOICE_PROBABILITY_SUM_EPSILON = 1e-12

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
    segments: Sequence[dict[str, Any]],
    *,
    model: str,
    uid: str | None = None,
    guidance: str = GUIDANCE,
) -> dict[str, Any]:
    """One request covering a whole window. System One requires the model field
    (omitting it returns 422 Unprocessable Entity)."""
    state: dict[str, Any] = {"guidance": guidance, "transcript": build_state(segments)}
    if uid is not None:
        state["uid"] = uid
    return {"state": state, "model": model, "questions": build_questions(segments)}


def parse_response(body: dict[str, Any]) -> dict[str, Any]:
    """Normalize an upstream reply into cache-entry shape."""
    if not isinstance(body, dict):
        raise ValueError("upstream response must be an object")
    answers = body.get("answers") or {}
    if not isinstance(answers, dict):
        raise ValueError("upstream answers must be an object")
    probs: dict[str, float] = {}
    for key, ans in answers.items():
        if not key.startswith("s") or not isinstance(ans, dict):
            continue
        value = ans.get("noul")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            probs[key] = float(value)
    usage = body.get("usage") or {}
    if not isinstance(usage, dict):
        raise ValueError("upstream usage must be an object")
    return {
        "probabilities": probs,
        "input_tokens": _usage_tokens(usage, "input_tokens"),
        "output_tokens": _usage_tokens(usage, "output_tokens"),
    }


# Transient upstream statuses worth retrying (SDK RetryPolicy: 408/429/5xx).
RETRY_STATUSES = frozenset({408, 429, *range(500, 600)})
RETRY_BACKOFF_INITIAL = 0.5
RETRY_BACKOFF_MAX = 5.0


def _retry_after_seconds(resp: httpx.Response) -> float | None:
    """Retry-After as a non-negative delay from a header or Jev response body."""
    header = resp.headers.get("Retry-After")
    if header:
        try:
            value = float(header)
            if math.isfinite(value) and value >= 0:
                return value
        except ValueError:
            try:
                value = parsedate_to_datetime(header).timestamp() - time.time()
                if math.isfinite(value):
                    return max(value, 0.0)
            except (TypeError, ValueError, IndexError, OverflowError):
                pass
    try:
        ms = resp.json().get("retry_after_ms")
    except Exception:
        return None
    if not isinstance(ms, (int, float)) or isinstance(ms, bool):
        return None
    value = float(ms) / 1000.0
    return value if math.isfinite(value) and value >= 0 else None


def call_payload(
    payload: dict[str, Any],
    *,
    url: str,
    api_key: str,
    timeout: float = 60.0,
    max_retries: int = 2,
    retry_after_max: float = 5.0,
    deadline_at: float | None = None,
    request_deadline: float = 75.0,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """POST upstream, retrying transient failures within one total deadline."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    own = client or httpx.Client(timeout=timeout)
    end = deadline_at if deadline_at is not None else time.monotonic() + request_deadline
    try:
        for attempt in range(max_retries + 1):
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise httpx.TimeoutException("Jev request deadline exceeded")
            start = time.monotonic()
            try:
                resp = own.post(url, json=payload, headers=headers, timeout=min(timeout, remaining))
            except httpx.TransportError as exc:
                metrics.record_jev_attempt((time.monotonic() - start) * 1000, False)
                if attempt >= max_retries:
                    raise
                delay = min(RETRY_BACKOFF_INITIAL * 2**attempt, RETRY_BACKOFF_MAX, retry_after_max)
                delay = min(delay, max(0.0, end - time.monotonic()))
                logger.warning(
                    "jev upstream transport error (%s), retry %d/%d in %.1fs",
                    type(exc).__name__,
                    attempt + 1,
                    max_retries,
                    delay,
                )
                if delay > 0:
                    time.sleep(delay)
                continue
            # URL, status, latency, and payload size only: never headers or the key.
            logger.debug(
                "jev upstream POST %s -> %d (%d questions) in %.0f ms",
                url,
                resp.status_code,
                len(payload.get("questions", {})),
                (time.monotonic() - start) * 1000,
            )
            duration_ms = (time.monotonic() - start) * 1000
            if 200 <= resp.status_code < 300:
                try:
                    result: dict[str, Any] = resp.json()
                    input_tokens = _input_tokens(result)
                    metrics.record_jev_attempt(
                        duration_ms,
                        True,
                        estimate_cost_usd(input_tokens) if input_tokens is not None else None,
                    )
                except Exception:
                    metrics.record_jev_attempt(duration_ms, True)
                    raise
            else:
                metrics.record_jev_attempt(duration_ms, False)
            if time.monotonic() > end:
                raise httpx.TimeoutException("Jev request deadline exceeded")
            if resp.status_code in RETRY_STATUSES and attempt < max_retries:
                backoff = min(RETRY_BACKOFF_INITIAL * 2**attempt, RETRY_BACKOFF_MAX)
                retry_after = _retry_after_seconds(resp)
                delay = retry_after if retry_after is not None else backoff
                remaining = max(0.0, end - time.monotonic())
                if retry_after is not None and (delay > retry_after_max or delay > remaining):
                    resp.raise_for_status()
                delay = min(delay, remaining)
                logger.warning(
                    "jev upstream %d, retry %d/%d in %.1fs",
                    resp.status_code,
                    attempt + 1,
                    max_retries,
                    delay,
                )
                if delay > 0:
                    time.sleep(delay)
                continue
            resp.raise_for_status()
            return result
        raise RuntimeError("unreachable")  # loop always returns or raises
    finally:
        if client is None:
            own.close()


def estimate_cost_usd(input_tokens: int) -> float:
    """Input tokens at INPUT_COST_PER_MTOK per Mtok; output is not billed."""
    return round(input_tokens / 1_000_000 * INPUT_COST_PER_MTOK, 6)


def _usage_tokens(usage: dict[str, Any], key: str) -> int:
    value = usage.get(key, 0)
    if value is None:
        return 0
    valid = False
    if isinstance(value, int) and not isinstance(value, bool):
        valid = value >= 0
    elif isinstance(value, float):
        valid = math.isfinite(value) and value >= 0 and int(value) == value
    if not valid:
        rule = "usage_input_tokens" if key == "input_tokens" else "usage_output_tokens"
        raise JevReviewValidationError(rule)
    return int(value)


def _input_tokens(body: dict[str, Any]) -> int | None:
    usage = body.get("usage")
    if not isinstance(usage, dict) or usage.get("input_tokens") is None:
        return None
    try:
        return _usage_tokens(usage, "input_tokens")
    except ValueError:
        return None


def _valid_answers(entry: dict[str, Any], expected_keys: set[str]) -> bool:
    probabilities = entry.get("probabilities")
    if not isinstance(probabilities, dict):
        return False
    for key in expected_keys:
        value = probabilities.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return False
    return all(
        isinstance(entry.get(key), int) and not isinstance(entry[key], bool) and entry[key] >= 0
        for key in ("input_tokens", "output_tokens")
    )


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
    max_retries: int = 2,
    retry_after_max: float = 5.0,
    request_deadline: float = 75.0,
    deadline_at: float | None = None,
    cache_max_entries: int = 10_000,
    guidance: str = GUIDANCE,
    fetcher: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the payload, serve from cache or upstream, and shape the reply."""
    end = deadline_at if deadline_at is not None else time.monotonic() + request_deadline
    payload = build_payload(segments, model=model, uid=uid, guidance=guidance)
    cache = JsonCache(cache_path, max_entries=cache_max_entries)
    send = fetcher or call_payload
    expected_keys = set(payload["questions"])

    def _fetch() -> dict[str, Any]:
        metrics.record_cache(False)
        kwargs: dict[str, Any] = {
            "url": url,
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if fetcher is None:
            kwargs.update(retry_after_max=retry_after_max, deadline_at=end)
        body = send(payload, **kwargs)
        return parse_response(body)

    entry, hit = cache.get_or_fetch(
        payload, _fetch, valid=lambda entry: _valid_answers(entry, expected_keys)
    )
    if hit:
        metrics.record_cache(True)
    if time.monotonic() > end:
        raise httpx.TimeoutException("Jev request deadline exceeded")
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


def _review_answers(body: dict[str, Any], questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Validate a mixed NouL/Choice review reply before it can enter the cache."""
    if not isinstance(body, dict):
        raise JevReviewValidationError("response_object")
    answers = body.get("answers")
    expected_keys = set(questions)
    if not isinstance(answers, dict):
        raise JevReviewValidationError("answers_object")
    if set(answers) != expected_keys:
        raise JevReviewValidationError(
            "answers_keys",
            {"expected_count": len(expected_keys), "actual_count": len(answers)},
        )
    parsed: dict[str, Any] = {}
    for key, answer in answers.items():
        if not isinstance(answer, dict):
            raise JevReviewValidationError("answer_object")
        if questions[key].get("type") == "noul":
            value = answer.get("noul")
            numeric = _finite_float(value)
            if numeric is None or not 0 <= numeric <= 1:
                details = {"actual": numeric} if numeric is not None else None
                raise JevReviewValidationError("evidence_probability", details)
            parsed[key] = numeric
            continue
        criteria = questions[key].get("criteria")
        if not isinstance(criteria, dict):
            raise JevReviewValidationError("choice_criteria_object")
        if not criteria:
            raise JevReviewValidationError(
                "choice_criteria_empty", {"expected_count": 1, "actual_count": 0}
            )
        choice = answer.get("choice")
        confidence = answer.get("confidence")
        probabilities = answer.get("probabilities")
        if not isinstance(choice, str) or not isinstance(probabilities, dict):
            raise JevReviewValidationError("choice_answer_shape")
        confidence_numeric = _finite_float(confidence)
        if confidence_numeric is None or not 0 <= confidence_numeric <= 1:
            details = {"actual": confidence_numeric} if confidence_numeric is not None else None
            raise JevReviewValidationError("choice_confidence", details)
        parsed_probs: dict[str, float] = {}
        for option, value in probabilities.items():
            numeric = _finite_float(value)
            if not isinstance(option, str) or numeric is None or not 0 <= numeric <= 1:
                details = {"actual": numeric} if numeric is not None else None
                raise JevReviewValidationError("choice_probability", details)
            parsed_probs[option] = numeric
        if set(parsed_probs) != set(criteria) or choice not in parsed_probs:
            raise JevReviewValidationError(
                "choice_probability_keys",
                {"expected_count": len(criteria), "actual_count": len(parsed_probs)},
            )
        total = math.fsum(parsed_probs.values())
        if not math.isclose(
            total,
            1.0,
            rel_tol=0.0,
            abs_tol=CHOICE_PROBABILITY_SUM_TOLERANCE + CHOICE_PROBABILITY_SUM_EPSILON,
        ):
            raise JevReviewValidationError(
                "choice_probability_sum",
                {
                    "expected_total": 1.0,
                    "actual_total": total,
                    "tolerance": CHOICE_PROBABILITY_SUM_TOLERANCE,
                },
            )
        winner = max(parsed_probs.values())
        if parsed_probs[choice] != winner:
            raise JevReviewValidationError(
                "choice_winner",
                {"selected_probability": parsed_probs[choice], "max_probability": winner},
            )
        parsed[key] = {
            "choice": choice,
            "confidence": confidence_numeric,
            "probabilities": parsed_probs,
        }
    usage = body.get("usage", {})
    if usage is None:
        usage = {}
    elif not isinstance(usage, dict):
        raise JevReviewValidationError("usage_object")
    return {
        "answers": parsed,
        "input_tokens": _usage_tokens(usage, "input_tokens"),
        "output_tokens": _usage_tokens(usage, "output_tokens"),
    }


def jev_review_questions(
    *,
    state: dict[str, Any],
    questions: dict[str, dict[str, Any]],
    url: str,
    api_key: str,
    timeout: float,
    cache_path: str,
    model: str,
    uid: str | None = None,
    max_retries: int = 2,
    retry_after_max: float = 5.0,
    request_deadline: float = 75.0,
    deadline_at: float | None = None,
    cache_max_entries: int = 10_000,
    fetcher: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate focused review questions through the validated shared cache."""
    end = deadline_at if deadline_at is not None else time.monotonic() + request_deadline
    full_state = dict(state)
    if uid is not None:
        full_state["uid"] = uid
    payload = {"state": full_state, "model": model, "questions": questions}
    cache = JsonCache(cache_path, max_entries=cache_max_entries)
    send = fetcher or call_payload

    def _fetch() -> dict[str, Any]:
        metrics.record_cache(False)
        kwargs: dict[str, Any] = {"url": url, "api_key": api_key, "timeout": timeout, "max_retries": max_retries}
        if fetcher is None:
            kwargs.update(retry_after_max=retry_after_max, deadline_at=end)
        return _review_answers(send(payload, **kwargs), questions)

    def _valid(entry: dict[str, Any]) -> bool:
        if not isinstance(entry.get("input_tokens"), int) or isinstance(entry.get("input_tokens"), bool) or entry["input_tokens"] < 0:
            return False
        if not isinstance(entry.get("output_tokens"), int) or isinstance(entry.get("output_tokens"), bool) or entry["output_tokens"] < 0:
            return False
        answers = entry.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(questions):
            return False
        normalized = {"answers": {}, "usage": {"input_tokens": entry.get("input_tokens"), "output_tokens": entry.get("output_tokens")}}
        for key, value in answers.items():
            if questions[key].get("type") == "noul" and isinstance(value, (int, float)) and not isinstance(value, bool):
                normalized["answers"][key] = {"noul": value}
            elif questions[key].get("type") == "choice" and isinstance(value, dict):
                normalized["answers"][key] = value
            else:
                return False
        try:
            _review_answers(normalized, questions)
        except (TypeError, ValueError):
            return False
        return True

    entry, hit = cache.get_or_fetch(payload, _fetch, valid=_valid)
    if hit:
        metrics.record_cache(True)
    if time.monotonic() > end:
        raise httpx.TimeoutException("Jev request deadline exceeded")
    return {
        "answers": entry["answers"],
        "cache_hit": hit,
        "usage": {"input_tokens": int(entry["input_tokens"]), "output_tokens": int(entry["output_tokens"])},
    }


# --- Category second pass -------------------------------------------------
# One call per span: the span's segments plus a little context and one Choice
# over MinusPod's category labels. Validation, caching, retries, and deadline
# handling are shared with focused review questions.

CATEGORY_GUIDANCE = (
    "Each line of `transcript` is one segment around a single advertising "
    "break in a podcast, prefixed with its line id. The lines named in "
    "`focus` are the break itself; the others are nearby context. Decide what "
    "KIND of break the focus lines are."
)

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "sponsor": "a paid sponsor read, product advertisement, or paid ad for another podcast or show",
    "cross_promo": "an unpaid promotion for another podcast or show in the same network or by the same host",
    "self_promo": "the host promoting their own show, Patreon, merch, membership, or back catalog",
    "interaction": "a call to action to like, subscribe, rate, review, follow, or comment",
    "intro": "an intro segment opening the episode",
    "outro": "an outro segment closing the episode",
    "recap": "a recap or summary of earlier content",
}

CATEGORY_CHOICE_KEY = "category"
CATEGORY_CHOICE_INSTRUCTIONS = (
    "Choose the one category that best describes the advertising break at lines {ids}. "
    "Use the supplied transcript context."
)


def _focus_ids(focus: Sequence[dict[str, Any]]) -> str:
    ordered = sorted(focus, key=lambda s: int(s["sid"]))
    lo, hi = int(ordered[0]["sid"]), int(ordered[-1]["sid"])
    return line_id(lo) if lo == hi else f"{line_id(lo)}-{line_id(hi)}"


def build_category_payload(
    focus: Sequence[dict[str, Any]],
    context: Sequence[dict[str, Any]],
    categories: Sequence[str],
    *,
    model: str,
    uid: str | None = None,
    guidance: str = CATEGORY_GUIDANCE,
) -> dict[str, Any]:
    """One request: span plus context state and one category Choice."""
    merged = {int(s["sid"]): s for s in [*context, *focus]}
    ordered = [merged[k] for k in sorted(merged)]
    ids = _focus_ids(focus)
    state: dict[str, Any] = {
        "guidance": guidance,
        "transcript": build_state(ordered),
        "focus": ids,
    }
    if uid is not None:
        state["uid"] = uid
    questions = {
        CATEGORY_CHOICE_KEY: {
            "type": "choice",
            "instructions": CATEGORY_CHOICE_INSTRUCTIONS.format(ids=ids),
            "criteria": {
                category: CATEGORY_DESCRIPTIONS.get(category, category) for category in categories
            },
        }
    }
    return {"state": state, "model": model, "questions": questions}


def jev_category(
    focus: Sequence[dict[str, Any]],
    context: Sequence[dict[str, Any]],
    categories: Sequence[str],
    *,
    url: str,
    api_key: str,
    timeout: float,
    cache_path: str,
    model: str,
    uid: str | None = None,
    max_retries: int = 2,
    retry_after_max: float = 5.0,
    request_deadline: float = 75.0,
    deadline_at: float | None = None,
    cache_max_entries: int = 10_000,
    guidance: str = CATEGORY_GUIDANCE,
    fetcher: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Classify one span with a validated category Choice."""
    payload = build_category_payload(
        focus, context, categories, model=model, uid=uid, guidance=guidance
    )
    try:
        result = jev_review_questions(
            state=payload["state"],
            questions=payload["questions"],
            url=url,
            api_key=api_key,
            timeout=timeout,
            cache_path=cache_path,
            model=model,
            max_retries=max_retries,
            retry_after_max=retry_after_max,
            request_deadline=request_deadline,
            deadline_at=deadline_at,
            cache_max_entries=cache_max_entries,
            fetcher=fetcher,
        )
    except JevReviewValidationError as exc:
        raise JevCategoryValidationError(exc.rule, exc.numeric_details) from exc
    answer = result["answers"][CATEGORY_CHOICE_KEY]
    return {
        "category": answer["choice"],
        "confidence": answer["confidence"],
        "cache_hit": result["cache_hit"],
        "probabilities": answer["probabilities"],
        "usage": result["usage"],
    }
