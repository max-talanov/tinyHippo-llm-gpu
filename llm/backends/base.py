"""Backend interface for the episodic-memory service.

A backend holds the *index* of memories, not their text: it is told
"memory <id> has content <text>" and later asked "which ids does this cue
point to?". The text itself lives in the service's SQLite store.

The interface is shaped by the hippocampal backend that will implement it
later (joint plan §7), so the stubs follow the same contract:

- ``remember`` may only activate pre-allocated resources (NEST GPU cannot
  create neurons after the first Simulate); a backend with a fixed bank
  raises ``CapacityError`` when it is full.
- ``sleep`` is where consolidation happens. Stubs can make it a no-op; the
  model's recall is expected to change after it.
- ``recall`` is slow on the model (seconds to a minute per call at 1%), and
  returns ranked ids with a confidence score in [0, 1] so the service can
  apply a no-recall threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class CapacityError(RuntimeError):
    """The backend has no free slot for another memory."""


@dataclass(frozen=True)
class Hit:
    memory_id: str
    score: float  # confidence in [0, 1]; 1 = certain


@runtime_checkable
class MemoryBackend(Protocol):
    name: str

    def remember(self, memory_id: str, text: str) -> None:
        """Store the index for one memory."""

    def sleep(self, n_blocks: int) -> None:
        """Run n_blocks of offline replay/consolidation."""

    def recall(self, cue: str, k: int) -> list[Hit]:
        """Return up to k hits for a partial cue, best first."""

    def forget_all(self) -> None:
        """Drop every stored memory (a fresh network on the model)."""
