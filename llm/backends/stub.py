"""Stub backends: the baselines the hippocampal backend is compared against.

``KeywordBackend`` scores a cue by token overlap with each memory (an
"exact match" baseline). ``TfidfBackend`` is a small vector store: TF-IDF
bag-of-words vectors and cosine similarity. Neither sleeps, forgets or
interferes, which is the point: they show what plain retrieval does, so the
model's sleep-dependent behaviour has something to stand out against.

Both are pure numpy/stdlib so the benchmark runs offline and deterministically.
A sentence-embedding vector store belongs to L4 (codec), not here.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .base import Hit

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are at be by did do does for from has have i in is it its me my "
    "of on or our that the their them they this to was we were what when where "
    "which who why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class KeywordBackend:
    """Token-overlap baseline: score = |cue ∩ memory| / |cue|."""

    name = "keyword"

    def __init__(self) -> None:
        self._tokens: dict[str, set[str]] = {}

    def remember(self, memory_id: str, text: str) -> None:
        self._tokens[memory_id] = set(tokenize(text))

    def sleep(self, n_blocks: int) -> None:
        pass

    def recall(self, cue: str, k: int) -> list[Hit]:
        q = set(tokenize(cue))
        if not q:
            return []
        hits = [Hit(mid, len(q & toks) / len(q)) for mid, toks in self._tokens.items()]
        hits = [h for h in hits if h.score > 0]
        hits.sort(key=lambda h: (-h.score, h.memory_id))
        return hits[:k]

    def forget_all(self) -> None:
        self._tokens.clear()


class TfidfBackend:
    """Vector-store baseline: TF-IDF vectors, cosine similarity.

    IDF is recomputed from the current memory set at each recall, so a word
    shared by many memories (a common name, say) counts for little.
    """

    name = "tfidf"

    def __init__(self) -> None:
        self._tf: dict[str, Counter[str]] = {}

    def remember(self, memory_id: str, text: str) -> None:
        self._tf[memory_id] = Counter(tokenize(text))

    def sleep(self, n_blocks: int) -> None:
        pass

    def _idf(self) -> dict[str, float]:
        n = len(self._tf)
        df: Counter[str] = Counter()
        for tf in self._tf.values():
            df.update(tf.keys())
        return {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}

    @staticmethod
    def _vec(tf: Counter[str], idf: dict[str, float]) -> dict[str, float]:
        v = {t: c * idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values()))
        return {t: x / norm for t, x in v.items()} if norm else {}

    def recall(self, cue: str, k: int) -> list[Hit]:
        if not self._tf:
            return []
        idf = self._idf()
        q = self._vec(Counter(tokenize(cue)), idf)
        if not q:
            return []
        hits = []
        for mid, tf in self._tf.items():
            v = self._vec(tf, idf)
            s = sum(x * v.get(t, 0.0) for t, x in q.items())
            if s > 0:
                hits.append(Hit(mid, min(1.0, s)))
        hits.sort(key=lambda h: (-h.score, h.memory_id))
        return hits[:k]

    def forget_all(self) -> None:
        self._tf.clear()


BACKENDS = {cls.name: cls for cls in (KeywordBackend, TfidfBackend)}
