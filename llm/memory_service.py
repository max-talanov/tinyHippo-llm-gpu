"""Episodic-memory service: SQLite content store plus a pluggable index backend.

The agent talks to this class through three operations:

    remember(text)   -> memory id      (text to SQLite, index to the backend)
    sleep(n_blocks)                    (offline consolidation in the backend)
    recall(cue, k)   -> ranked records (backend ranks ids, SQLite returns text)

Every call is timed and logged to the ``calls`` table, so latency is measured
the same way for every backend (stub, CPU NEST, NEST GPU).
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass

from .backends.base import MemoryBackend

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    session     TEXT,
    created_at  REAL NOT NULL,
    sleeps_at_creation INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS calls (
    op        TEXT NOT NULL,
    backend   TEXT NOT NULL,
    started   REAL NOT NULL,
    seconds   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS state (
    key   TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
"""


@dataclass(frozen=True)
class Recalled:
    memory_id: str
    text: str
    score: float


class MemoryService:
    def __init__(self, backend: MemoryBackend, db_path: str = ":memory:",
                 min_score: float = 0.0) -> None:
        """min_score: hits below this confidence are dropped (the no-recall
        threshold; each backend needs its own, set from the benchmark)."""
        self.backend = backend
        self.min_score = min_score
        self.db = sqlite3.connect(db_path)
        self.db.executescript(_SCHEMA)
        self.db.execute("INSERT OR IGNORE INTO state VALUES ('sleep_blocks', 0)")
        self.db.commit()
        # Rebuild the backend index from the store, so a service reopened on an
        # existing db file recalls what it remembered before. The model backend
        # will restore from its own checkpoint instead (LLM plan L3).
        for mid, text in self.db.execute("SELECT id, text FROM memories ORDER BY rowid"):
            self.backend.remember(mid, text)

    def _log(self, op: str, t0: float) -> None:
        self.db.execute("INSERT INTO calls VALUES (?, ?, ?, ?)",
                        (op, self.backend.name, t0, time.perf_counter() - t0))

    @property
    def sleep_blocks(self) -> int:
        return self.db.execute("SELECT value FROM state WHERE key='sleep_blocks'").fetchone()[0]

    def remember(self, text: str, session: str | None = None,
                 memory_id: str | None = None) -> str:
        mid = memory_id or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        self.backend.remember(mid, text)
        self.db.execute("INSERT INTO memories VALUES (?, ?, ?, ?, ?)",
                        (mid, text, session, time.time(), self.sleep_blocks))
        self._log("remember", t0)
        self.db.commit()
        return mid

    def sleep(self, n_blocks: int = 1) -> None:
        if n_blocks < 1:
            raise ValueError("n_blocks must be >= 1")
        t0 = time.perf_counter()
        self.backend.sleep(n_blocks)
        self.db.execute("UPDATE state SET value = value + ? WHERE key='sleep_blocks'", (n_blocks,))
        self._log("sleep", t0)
        self.db.commit()

    def recall(self, cue: str, k: int = 3) -> list[Recalled]:
        t0 = time.perf_counter()
        hits = [h for h in self.backend.recall(cue, k) if h.score >= self.min_score]
        self._log("recall", t0)
        self.db.commit()
        out = []
        for h in hits:
            row = self.db.execute("SELECT text FROM memories WHERE id=?", (h.memory_id,)).fetchone()
            if row is not None:
                out.append(Recalled(h.memory_id, row[0], h.score))
        return out

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def latencies(self, op: str) -> list[float]:
        return [r[0] for r in self.db.execute("SELECT seconds FROM calls WHERE op=?", (op,))]

    def close(self) -> None:
        self.db.close()
