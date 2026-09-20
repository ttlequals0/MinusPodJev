"""Bounded, process-safe disk cache for Jev responses.

``JEV_CACHE_PATH`` historically named a JSON cache. Existing JSON entries are
imported once into a sibling SQLite database; the source file is left untouched.
SQLite avoids lost updates across Uvicorn workers and atomic commits prevent a
partial write from invalidating the whole cache.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def hash_payload(payload: dict[str, Any]) -> str:
    """Cache key covering everything that would change the answer."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


class JsonCache:
    """Disk cache of per-window Jev answers, backed by SQLite."""

    def __init__(self, path: Path | str, *, max_entries: int = 10_000):
        self.path = Path(path)
        self.db_path = Path(f"{self.path}.sqlite3")
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS entries "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at INTEGER NOT NULL)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS entries_created_at ON entries(created_at)")
            conn.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            migrated = conn.execute(
                "SELECT 1 FROM metadata WHERE key = 'legacy_json_imported'"
            ).fetchone()
            if migrated is None:
                legacy: dict[str, Any] = {}
                if self.path.is_file():
                    try:
                        parsed = json.loads(self.path.read_text())
                        if isinstance(parsed, dict):
                            legacy = parsed
                    except (json.JSONDecodeError, OSError):
                        pass
                conn.executemany(
                    "INSERT OR IGNORE INTO entries(key, value, created_at) VALUES (?, ?, ?)",
                    ((key, json.dumps(value, sort_keys=True), int(time.time())) for key, value in legacy.items()),
                )
                conn.execute("INSERT INTO metadata(key, value) VALUES ('legacy_json_imported', '1')")
                conn.execute(
                    "DELETE FROM entries WHERE key IN "
                    "(SELECT key FROM entries ORDER BY created_at DESC, rowid DESC LIMIT -1 OFFSET ?)",
                    (self.max_entries,),
                )
            conn.commit()
        finally:
            conn.close()

    def get_or_fetch(
        self,
        payload: dict[str, Any],
        fetch: Callable[[], dict[str, Any]],
        *,
        valid: Callable[[dict[str, Any]], bool] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Return a valid cached entry or fetch and atomically persist a new one."""
        key = hash_payload(payload)
        with self._connection() as conn:
            row = conn.execute("SELECT value FROM entries WHERE key = ?", (key,)).fetchone()
            if row is not None:
                try:
                    entry = json.loads(row[0])
                except json.JSONDecodeError:
                    entry = None
                if isinstance(entry, dict) and (valid is None or valid(entry)):
                    self.hits += 1
                    return entry, True
                conn.execute("DELETE FROM entries WHERE key = ?", (key,))

        entry = fetch()
        if valid is not None and not valid(entry):
            raise ValueError("upstream response did not contain valid answers for every question")
        encoded = json.dumps(entry, sort_keys=True)
        with self._connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO entries(key, value, created_at) VALUES (?, ?, strftime('%s', 'now'))",
                (key, encoded),
            )
            row = conn.execute("SELECT value FROM entries WHERE key = ?", (key,)).fetchone()
            if row is None:
                raise RuntimeError("cache entry disappeared after insert")
            stored = json.loads(row[0])
            conn.execute(
                "DELETE FROM entries WHERE key IN "
                "(SELECT key FROM entries ORDER BY created_at DESC, rowid DESC LIMIT -1 OFFSET ?)",
                (self.max_entries,),
            )
        self.misses += 1
        return stored, False
