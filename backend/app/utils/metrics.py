"""Ephemeral, process-scoped runtime metrics."""

from __future__ import annotations

import os
import threading
import time
from typing import Any

_REVIEW_OUTCOMES = (
    "confirmed",
    "adjusted",
    "rejected",
    "inconclusive",
    "upstream_error",
    "invalid_request",
    "internal_error",
)
_REVIEW_REASON_CODES = (
    "ambiguous_spans",
    "insufficient_evidence",
    "no_valid_pairs",
    "too_many_boundary_options",
    "transcript_gap",
    "choice_inconclusive",
    "neither_complete",
    "ad_content_unconfirmed",
    "programme_content_detected",
    "missing_boundary_coverage",
    "insufficient_boundary_text",
    "edge_content_unconfirmed",
    "adjacent_message_continues",
    "unrelated_editorial",
    "malformed_context",
    "invalid_choice",
    "upstream_failure",
    "upstream_invalid_response",
    "invalid_request",
    "internal_error",
    "unknown",
)
_REFINEMENT_SKIP_REASONS = (
    "disabled",
    "missing_word_timings",
    "insufficient_evidence",
    "ambiguous_spans",
    "no_overlapping_span",
    "no_valid_pairs",
    "transcript_gap",
)


class RuntimeMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._started_at = time.monotonic()
            self._proxy = self._empty_timing()
            self._jev = self._empty_timing()
            self._review = self._empty_timing()
            self._review_outcomes = dict.fromkeys(_REVIEW_OUTCOMES, 0)
            self._review_reasons = dict.fromkeys(_REVIEW_REASON_CODES, 0)
            self._review_refinement = dict.fromkeys(
                ("attempted", "completed", "changed", "unchanged", "inconclusive", "upstream_error"),
                0,
            )
            self._review_refinement_skipped = dict.fromkeys(_REFINEMENT_SKIP_REASONS, 0)
            self._cache_hits = 0
            self._cache_misses = 0
            self._estimated_input_usd = 0.0
            self._unknown_usage = 0

    @staticmethod
    def _empty_timing() -> dict[str, float | int | None]:
        return {"count": 0, "success": 0, "failure": 0, "sum": 0.0, "min": None, "max": None}

    @staticmethod
    def _record(timing: dict[str, float | int | None], duration_ms: float, success: bool) -> None:
        timing["count"] = int(timing["count"] or 0) + 1
        timing["success" if success else "failure"] = int(timing["success" if success else "failure"] or 0) + 1
        timing["sum"] = float(timing["sum"] or 0.0) + duration_ms
        current_min = timing["min"]
        current_max = timing["max"]
        timing["min"] = duration_ms if current_min is None else min(float(current_min), duration_ms)
        timing["max"] = duration_ms if current_max is None else max(float(current_max), duration_ms)

    def record_proxy_request(self, duration_ms: float, success: bool) -> None:
        with self._lock:
            self._record(self._proxy, duration_ms, success)

    def record_jev_attempt(
        self, duration_ms: float, success: bool, estimated_input_usd: float | None = None
    ) -> None:
        with self._lock:
            self._record(self._jev, duration_ms, success)
            if success and estimated_input_usd is None:
                self._unknown_usage += 1
            elif success and estimated_input_usd is not None:
                self._estimated_input_usd += estimated_input_usd

    def record_cache(self, hit: bool) -> None:
        with self._lock:
            if hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1

    def record_review(
        self, outcome: str, duration_ms: float, reason_code: str | None = None
    ) -> None:
        with self._lock:
            safe_outcome = outcome if outcome in _REVIEW_OUTCOMES else "internal_error"
            safe_reason = reason_code if reason_code in _REVIEW_REASON_CODES else "unknown"
            self._record(self._review, duration_ms, safe_outcome not in {"upstream_error", "internal_error"})
            self._review_outcomes[safe_outcome] += 1
            self._review_reasons[safe_reason] += 1

    def record_review_refinement(
        self,
        event: str,
        *,
        skip_reason: str | None = None,
        changed: bool | None = None,
    ) -> None:
        with self._lock:
            if event == "attempted":
                self._review_refinement["attempted"] += 1
            elif event == "completed":
                self._review_refinement["completed"] += 1
                self._review_refinement["changed" if changed else "unchanged"] += 1
            elif event in {"inconclusive", "upstream_error"}:
                self._review_refinement[event] += 1
            elif event == "skipped" and skip_reason in _REFINEMENT_SKIP_REASONS:
                self._review_refinement_skipped[skip_reason] += 1

    @staticmethod
    def _latency(timing: dict[str, float | int | None]) -> dict[str, float | int]:
        count = int(timing["count"] or 0)
        total = float(timing["sum"] or 0.0)
        return {
            "count": count,
            "sum": total,
            "min": float(timing["min"] or 0.0),
            "max": float(timing["max"] or 0.0),
            "average": total / count if count else 0.0,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            raw_workers = os.environ.get("WORKERS")
            try:
                configured_workers = int(raw_workers) if raw_workers else None
            except ValueError:
                configured_workers = None
            return {
                "scope": {
                    "kind": "process",
                    "pid": os.getpid(),
                    "configured_workers": configured_workers,
                },
                "reset_on_restart": True,
                "uptime_seconds": time.monotonic() - self._started_at,
                "proxy_requests": {
                    "count": int(self._proxy["count"] or 0),
                    "success": int(self._proxy["success"] or 0),
                    "failure": int(self._proxy["failure"] or 0),
                    "latency_ms": self._latency(self._proxy),
                },
                "jev_http": {
                    "attempts": int(self._jev["count"] or 0),
                    "success": int(self._jev["success"] or 0),
                    "failure": int(self._jev["failure"] or 0),
                    "unknown_usage": self._unknown_usage,
                    "latency_ms": self._latency(self._jev),
                },
                "review": {
                    "count": int(self._review["count"] or 0),
                    "outcomes": dict(self._review_outcomes),
                    "reasons": dict(self._review_reasons),
                    "latency_ms": self._latency(self._review),
                    "refinement": {
                        "attempted": int(self._review_refinement["attempted"]),
                        "completed": int(self._review_refinement["completed"]),
                        "changed": int(self._review_refinement["changed"]),
                        "unchanged": int(self._review_refinement["unchanged"]),
                        "inconclusive": int(self._review_refinement["inconclusive"]),
                        "upstream_error": int(self._review_refinement["upstream_error"]),
                        "skipped": {
                            reason: int(self._review_refinement_skipped[reason])
                            for reason in _REFINEMENT_SKIP_REASONS
                        },
                    },
                },
                "cache": {"hits": self._cache_hits, "misses": self._cache_misses},
                "cost": {"estimated_input_usd": self._estimated_input_usd},
            }


metrics = RuntimeMetrics()
