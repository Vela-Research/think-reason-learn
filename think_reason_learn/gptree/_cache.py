"""Persistent answer cache for GPTree.

Stores (question, sample, temperature, model) → answer mappings in SQLite
so that re-runs, resumed training, and development iteration avoid
redundant LLM calls.
"""

from __future__ import annotations

import hashlib
import sqlite3
import datetime
from pathlib import Path


class AnswerCache:
    """SQLite-backed cache for ``_answer_question_for_row`` responses.

    Args:
        db_path: Path to the SQLite database file. Parent directories
            are created automatically.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache "
            "(key TEXT PRIMARY KEY, answer TEXT, created_at TEXT)"
        )
        self._conn.commit()
        self.hits: int = 0
        self.misses: int = 0

    @staticmethod
    def _make_key(question: str, sample: str, temperature: float, model: str) -> str:
        payload = f"{question}\x00{sample}\x00{temperature}\x00{model}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(
        self, question: str, sample: str, temperature: float, model: str
    ) -> str | None:
        """Look up a cached answer. Returns ``None`` on miss."""
        key = self._make_key(question, sample, temperature, model)
        row = self._conn.execute(
            "SELECT answer FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row is not None:
            self.hits += 1
            return row[0]
        self.misses += 1
        return None

    def put(
        self,
        question: str,
        sample: str,
        temperature: float,
        model: str,
        answer: str,
    ) -> None:
        """Store an answer in the cache."""
        key = self._make_key(question, sample, temperature, model)
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, answer, created_at) VALUES (?, ?, ?)",
            (key, answer, datetime.datetime.now().isoformat()),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()
        return row[0] if row else 0
