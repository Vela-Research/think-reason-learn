"""A disk cache of model replies, so no answer is paid for twice.

Keys are hashes of everything that changes a reply: the endpoint, the model's
settings, the request and a draw number for intentional repeats. Values are
JSON. One SQLite file in WAL mode is safe across threads and processes, and a
rerun after a crash or a budget stop replays what was already paid for.

Each answer is one row, so a large run writes millions: keys are 16 bytes and the
table is stored by key (``WITHOUT ROWID``), which halves the file and the time
each write blocks the event loop.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import sqlite3
import threading
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final

from ._loop import on_fork

logger = logging.getLogger(__name__)

type JSON = str | int | float | bool | None | list[JSON] | dict[str, JSON]

_VERSION: Final = 2
_BATCH: Final = 500


def cache_key(**parts: JSON) -> str:
    """A stable hash of ``parts``: 128 bits, as hex."""
    canonical = json.dumps({"v": _VERSION, **parts}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


class Cache:
    """Replies stored by key in one SQLite file.

    Args:
        path: The database file. Its folder is created if needed.
    """

    def __init__(self, path: Path) -> None:
        """Open, or create, the database at ``path``, readable only by its owner."""
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.touch(mode=0o600)  # SQLite gives its -wal and -shm files this mode
        self._db = sqlite3.connect(
            path, timeout=30, isolation_level=None, check_same_thread=False
        )
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA wal_autocheckpoint=4000")  # 16 MB between syncs
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS replies (key BLOB PRIMARY KEY, "
            "value TEXT NOT NULL, created REAL NOT NULL) WITHOUT ROWID"
        )
        self._lock = threading.Lock()

    def get_many(self, keys: Iterable[str]) -> dict[str, JSON]:
        """The stored replies among ``keys``; an unreadable one counts as missing."""
        found: dict[str, JSON] = {}
        for keyed in itertools.batched(dict.fromkeys(keys), _BATCH, strict=False):
            batch = [bytes.fromhex(key) for key in keyed]
            marks = ",".join("?" * len(batch))
            try:
                with self._lock:
                    rows = self._db.execute(
                        f"SELECT key, value FROM replies WHERE key IN ({marks})", batch
                    ).fetchall()
            except sqlite3.Error:
                logger.warning("Could not read the reply cache.", exc_info=True)
                return found
            found.update(
                (key, value) for key, value in map(_decoded, rows) if key is not None
            )
        return found

    def put_many(self, replies: Mapping[str, JSON]) -> None:
        """Store ``replies``; a failure to write is logged, not raised."""
        now = time.time()
        rows = [
            (bytes.fromhex(key), json.dumps(value, ensure_ascii=False), now)
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


def open_cache(path: Path) -> Cache | None:
    """The cache at ``path``, opened once per process; ``None`` if it cannot open.

    Opening it says where it is, once. A read-only folder or a corrupt file
    turns caching off for the run, with a warning, instead of stopping it.
    """
    with _caches_lock:
        if path not in _caches:
            try:
                _caches[path] = Cache(path)
            except (OSError, sqlite3.Error) as error:
                logger.warning(f"Running without the reply cache at {path}: {error}")
                return None
            logger.warning(
                f"Model replies are cached at {path}, so a rerun reuses them "
                "instead of calling again. set_config(cache=False) turns it off."
            )
        return _caches[path]


on_fork(_caches.clear)


def _decoded(row: tuple[bytes, str]) -> tuple[str | None, JSON]:
    key, value = row
    try:
        return key.hex(), json.loads(value)
    except ValueError:
        return None, None
