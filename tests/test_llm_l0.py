"""L0 tests: memory service, stub backends, benchmark, agent tool wiring (offline)."""

from __future__ import annotations

import pytest

from llm.agent import build_tools
from llm.backends import BACKENDS, KeywordBackend, MemoryBackend, TfidfBackend
from llm.bench import dialogue, run_bench
from llm.memory_service import MemoryService


@pytest.mark.parametrize("cls", [KeywordBackend, TfidfBackend])
def test_backend_protocol_and_ranking(cls):
    b = cls()
    assert isinstance(b, MemoryBackend)
    b.remember("a", "Alice's dentist appointment is on Tuesday at 2pm.")
    b.remember("b", "Bruno drives a red Volvo.")
    hits = b.recall("When is Alice's dentist appointment?", k=2)
    assert hits and hits[0].memory_id == "a"
    assert all(0 < h.score <= 1 for h in hits)
    b.forget_all()
    assert b.recall("Alice dentist", k=2) == []


def test_service_store_recall_threshold_and_reopen(tmp_path):
    db = str(tmp_path / "m.sqlite")
    s = MemoryService(TfidfBackend(), db_path=db)
    mid = s.remember("Chen is allergic to peanuts.")
    s.remember("Dana grew up in Porto.")
    s.sleep(2)
    assert s.sleep_blocks == 2
    top = s.recall("What is Chen allergic to?", k=1)
    assert top[0].memory_id == mid and top[0].text == "Chen is allergic to peanuts."
    assert len(s.latencies("recall")) == 1
    s.min_score = 1.01
    assert s.recall("What is Chen allergic to?") == []
    s.close()
    # a reopened service rebuilds the index from the store
    s2 = MemoryService(TfidfBackend(), db_path=db)
    assert s2.count() == 2 and s2.sleep_blocks == 2
    assert s2.recall("Where did Dana grow up?", k=1)[0].text == "Dana grew up in Porto."


def test_sleep_rejects_zero():
    with pytest.raises(ValueError):
        MemoryService(KeywordBackend()).sleep(0)


def test_dialogue_deterministic_and_well_formed():
    a, b = dialogue.generate(seed=7), dialogue.generate(seed=7)
    assert a == b and a != dialogue.generate(seed=8)
    stored = {e["fact"] for e in a if e["op"] == "remember"}
    queries = [e for e in a if e["op"] == "query"]
    assert {e["kind"] for e in queries} == {"direct", "paraphrase", "near_dup", "negative"}
    assert all(q["target"] in stored for q in queries if q["target"] is not None)
    # negatives must name a (person, attribute) pair that was never stored
    pairs = {(e["person"], e["attribute"]) for e in a if e["op"] == "remember"}
    texts = {e["text"] for e in a if e["op"] == "remember"}
    assert len(pairs) == len(stored) == len(texts)
    assert sum(e["op"] == "sleep" for e in a) == 4


def test_bench_scores_are_consistent():
    events = dialogue.generate(seed=0)
    records = run_bench.run(events, MemoryService(TfidfBackend()), k=3)
    s = run_bench.score(records, threshold=0.0, k=3)
    pos = s["all_positive"]
    assert 0 <= pos["recall@1"] <= pos["recall@3"] <= 1
    assert pos["recall@1"] + pos["wrong_recall"] <= 1 + 1e-9
    # at threshold 0 any lexical overlap is a hit, so negatives mostly false-recall;
    # a threshold above every score suppresses all recall
    high = run_bench.score(records, threshold=1.01, k=3)
    assert high["negative"]["false_recall"] == 0 and high["all_positive"]["recall@3"] == 0


def test_bench_cli(tmp_path):
    out = tmp_path / "r.json"
    for name in BACKENDS:
        r = run_bench.main(["--backend", name, "--seed", "1", "--sweep", "--out", str(out)])
        assert r["n_memories"] > 0 and "0.5" in r["sweep"]


def test_agent_tools_drive_service():
    s = MemoryService(TfidfBackend())
    remember, recall, sleep = build_tools(s)
    assert remember.name == "remember" and "fact" in remember.input_schema["properties"]
    out = remember.call({"fact": "Elif has a cat named Pixel."})
    assert out.startswith("stored as ")
    assert "Pixel" in recall.call({"cue": "What is the name of Elif's cat?"})
    assert "no matching" in recall.call({"cue": "zzz"})
    sleep.call({"n_blocks": 2})
    assert s.sleep_blocks == 2
