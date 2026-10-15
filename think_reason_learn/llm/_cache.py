"""A disk cache of model replies, so no answer is paid for twice.

Keys are hashes of everything that changes a reply: the endpoint, the model's
settings, the request and a draw number for intentional repeats. Values are
JSON. One SQLite file in WAL mode is safe across threads and processes, and a
rerun after a crash or a budget stop replays what was already paid for.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

type JSON = str | int | float | bool | None | list[JSON] | dict[str, JSON]

_VERSION: Final = 1
_BATCH: Final = 500


def cache_key(**parts: JSON) -> str:
    """A stable hash of ``parts``."""
    canonical = json.dumps({"v": _VERSION, **parts}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


class Cache:
    """Replies stored by key in one SQLite file.

    Args:
        path: The database file. Its folder is created if needed.
    """

    def __init__(self, path: Path) -> None:
        """Open, or create, the database at ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(
            path, timeout=30, isolation_level=None, check_same_thread=False
        )
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS replies "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL, created REAL NOT NULL)"
        )
        self._lock = threading.Lock()

    def get_many(self, keys: Iterable[str]) -> dict[str, JSON]:
        """The stored replies among ``keys``."""
        wanted = list(dict.fromkeys(keys))
        found: dict[str, JSON] = {}
        for start in range(0, len(wanted), _BATCH):
            batch = wanted[start : start + _BATCH]
            marks = ",".join("?" * len(batch))
            with self._lock:
                rows = self._db.execute(
                    f"SELECT key, value FROM replies WHERE key IN ({marks})", batch
                ).fetchall()
            found.update((key, json.loads(value)) for key, value in rows)
        return found

    def put_many(self, replies: Mapping[str, JSON]) -> None:
        """Store ``replies``; a failure to write is logged, not raised."""
        now = time.time()
        rows = [
            (key, json.dumps(value, ensure_ascii=False), now)
            for key, value in replies.items()
        ]
        try:
            with self._lock:
                self._db.executemany(
                    "INSERT OR REPLACE INTO replies (key, value, created) "
                    "VALUES (?, ?, ?)",
                    rows,
                )
        except sqlite3.Error:
            logger.warning("Could not write the reply cache.", exc_info=True)


_caches: Final[dict[Path, Cache]] = {}
_caches_lock: Final = threading.Lock()


def open_cache(path: Path) -> Cache:
    """The cache at ``path``, opened once per process."""
    with _caches_lock:
        if path not in _caches:
            _caches[path] = Cache(path)
        return _caches[path]


os.register_at_fork(after_in_child=_caches.clear)
