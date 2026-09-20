"""Ephemeral, process-scoped runtime metrics."""

from __future__ import annotations

import os
import threading
import time
from typing import Any


class RuntimeMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._started_at = time.monotonic()
            self._proxy = self._empty_timing()
            self._jev = self._empty_timing()
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
                "cache": {"hits": self._cache_hits, "misses": self._cache_misses},
                "cost": {"estimated_input_usd": self._estimated_input_usd},
            }


metrics = RuntimeMetrics()
