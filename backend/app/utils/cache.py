"""
On-disk JSON response cache for Jev payloads.

Threshold sweeps re-read these entries instead of re-billing the upstream:
the only spend is the first pass. Entries are keyed by a hash of the payload,
so editing a question invalidates its entries rather than silently scoring
stale answers.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


def hash_payload(payload: dict[str, Any]) -> str:
    """Cache key covering everything that would change the answer."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


class JsonCache:
    """Disk cache of per-window Jev answers."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._data: dict[str, dict[str, Any]] = {}
        if self.path.is_file():
            try:
                self._data = json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                self._data = {}
        self.hits = 0
        self.misses = 0

    def get_or_fetch(
        self,
        payload: dict[str, Any],
        fetch: Callable[[], dict[str, Any]],
    ) -> tuple[dict[str, Any], bool]:
        """Return the cached entry for this payload, fetching it on miss."""
        key = hash_payload(payload)
        entry = self._data.get(key)
        if entry is not None:
            self.hits += 1
            return entry, True
        entry = fetch()
        self._data[key] = entry
        self.misses += 1
        self.save()
        return entry, False

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=0, sort_keys=True))
