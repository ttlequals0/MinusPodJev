"""
Probability curve -> contiguous spans, with hysteresis thresholds.

Ported from the bench reference: a run opens on a segment at or above
``enter`` and extends across neighbours at or above ``stay``, so a mid-break
segment that reads weaker than its neighbours does not split the span.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# Signal is strongly bimodal, so a run should open on near-certainty and then
# extend generously across the weaker shoulders of the same break.
ENTER_THRESHOLD = 0.95
STAY_THRESHOLD = 0.40

# Adjacency is in time, not list position: a silence gap wider than this ends
# a run. Only speech bridges; an empty gap is silence.
MAX_RUN_GAP_SECONDS = 30.0
BRIDGE_SECONDS = 30.0


def _time(seg: dict[str, Any], key: str, fallback: float) -> float:
    value = seg.get(key)
    return float(value) if isinstance(value, (int, float)) else fallback


def _prob(probabilities: dict[str, float], sid: int) -> float:
    return probabilities.get(f"s{sid}", 0.0)


def spans_from_probabilities(
    segments: Sequence[dict[str, Any]],
    probabilities: dict[str, float],
    *,
    enter: float | None = None,
    stay: float | None = None,
    max_gap: float = MAX_RUN_GAP_SECONDS,
    bridge: float = BRIDGE_SECONDS,
) -> list[dict[str, Any]]:
    """Contiguous runs of ad-ish segments, bounded by real segment edges."""
    enter = ENTER_THRESHOLD if enter is None else enter
    stay = STAY_THRESHOLD if stay is None else stay
    if stay > enter:
        raise ValueError("stay threshold must not exceed enter threshold")

    ordered = sorted(segments, key=lambda s: int(s["sid"]))
    missing = [s["sid"] for s in ordered if f"s{int(s['sid'])}" not in probabilities]
    if missing:
        raise ValueError(f"{len(missing)} segment(s) returned no answer")
    probs = [_prob(probabilities, int(s["sid"])) for s in ordered]
    above_stay = [p >= stay for p in probs]
    opens = [p >= enter for p in probs]

    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(ordered):
        if not above_stay[i]:
            i += 1
            continue
        j = i
        while (
            j + 1 < len(ordered)
            and above_stay[j + 1]
            and _time(ordered[j + 1], "start", float(j + 1))
            - _time(ordered[j], "end", float(j + 1))
            <= max_gap
        ):
            j += 1
        if any(opens[k] for k in range(i, j + 1)):
            runs.append((i, j))
        i = j + 1

    spans: list[dict[str, Any]] = []
    for lo, hi in _bridged(runs, ordered, bridge):
        members = ordered[lo : hi + 1]
        spans.append(
            {
                "start_id": int(members[0]["sid"]),
                "end_id": int(members[-1]["sid"]),
                "confidence": max(_prob(probabilities, int(m["sid"])) for m in members),
            }
        )
    return spans


def _bridged(
    runs: list[tuple[int, int]], ordered: Sequence[dict[str, Any]], bridge: float
) -> list[tuple[int, int]]:
    """Join runs separated by a short stretch of low-scoring speech.

    The gap must contain at least one segment: an empty gap is silence, and
    silence between two breaks is what separates them. Without timing fields
    the spanned gap is 0.0 seconds, so only directly adjacent runs merge.
    """
    if not runs:
        return []
    out = [runs[0]]
    for lo, hi in runs[1:]:
        prev_lo, prev_hi = out[-1]
        between = ordered[prev_hi + 1 : lo]
        spanned = sum(
            _time(s, "end", 0.0) - _time(s, "start", 0.0) for s in between
        )
        wall = _time(ordered[lo], "start", 0.0) - _time(ordered[prev_hi], "end", 0.0)
        if between and spanned <= bridge and wall <= bridge:
            out[-1] = (prev_lo, hi)
        else:
            out.append((lo, hi))
    return out
