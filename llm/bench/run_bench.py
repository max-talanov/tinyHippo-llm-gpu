"""Run a benchmark scenario against a backend and score it.

This drives the MemoryService directly, with no LLM in the loop, so backends
are compared on retrieval alone and the run is deterministic and free. The
agent-in-the-loop comparison is L5.

    python -m llm.bench.run_bench --backend tfidf --seed 0 --out out/bench_tfidf.json

Metrics (positives = queries with a stored target, negatives = never stored):
    recall@1, recall@k   target is the top hit / in the top k
    false_recall         negatives that return any hit at or above threshold
    wrong_recall         positives whose top hit is confident but wrong
    latency              p50 / p95 seconds per recall call
Each is broken down by query kind, by lag (remember events since the target
was stored) and by sleep blocks since storage. Raw top scores are kept so the
threshold can be swept after the run (--sweep).
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict

from ..backends import BACKENDS
from ..memory_service import MemoryService
from . import dialogue


def run(events: list[dict], service: MemoryService, k: int = 3) -> list[dict]:
    """Replay events; return one record per query with the ranked hits."""
    stored_at: dict[str, tuple[int, int]] = {}  # fact -> (n_remembered, sleeps) at storage
    n_remembered = 0
    records = []
    for ev in events:
        if ev["op"] == "remember":
            service.remember(ev["text"], session=str(ev["session"]), memory_id=ev["fact"])
            n_remembered += 1
            stored_at[ev["fact"]] = (n_remembered, service.sleep_blocks)
        elif ev["op"] == "sleep":
            service.sleep(ev["n_blocks"])
        elif ev["op"] == "query":
            hits = service.recall(ev["cue"], k)
            rec = {"cue": ev["cue"], "target": ev["target"], "kind": ev["kind"],
                   "hits": [(h.memory_id, round(h.score, 4)) for h in hits]}
            if ev["target"] is not None:
                n0, s0 = stored_at[ev["target"]]
                rec["lag"] = n_remembered - n0
                rec["sleeps_since"] = service.sleep_blocks - s0
            records.append(rec)
    return records


def _lag_bucket(lag: int) -> str:
    return "0-7" if lag < 8 else "8-23" if lag < 24 else "24+"


def score(records: list[dict], threshold: float = 0.0, k: int = 3) -> dict:
    def confident(rec):
        return [h for h in rec["hits"][:k] if h[1] >= threshold and h[1] > 0]

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["target"] is None:
            groups["negative"].append(r)
            continue
        groups["all_positive"].append(r)
        groups[f"kind={r['kind']}"].append(r)
        groups[f"lag={_lag_bucket(r['lag'])}"].append(r)
        groups[f"sleeps={min(r['sleeps_since'], 3)}{'+' if r['sleeps_since'] >= 3 else ''}"].append(r)

    out: dict = {"threshold": threshold, "k": k}
    for name, recs in sorted(groups.items()):
        n = len(recs)
        if name == "negative":
            out[name] = {"n": n, "false_recall": sum(bool(confident(r)) for r in recs) / n}
            continue
        r1 = rk = wrong = 0
        for r in recs:
            hits = confident(r)
            ids = [h[0] for h in hits]
            r1 += bool(ids) and ids[0] == r["target"]
            rk += r["target"] in ids
            wrong += bool(ids) and ids[0] != r["target"]
        out[name] = {"n": n, "recall@1": r1 / n, f"recall@{k}": rk / n, "wrong_recall": wrong / n}
    return out


def latency(service: MemoryService) -> dict:
    lat = sorted(service.latencies("recall"))
    if not lat:
        return {}
    return {"n": len(lat), "p50_s": statistics.median(lat),
            "p95_s": lat[min(len(lat) - 1, int(0.95 * len(lat)))]}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="tfidf")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenario", help="JSON scenario file (default: generate from --seed)")
    ap.add_argument("--n-sessions", type=int, default=4)
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--sweep", action="store_true", help="also score at thresholds 0.0..0.9")
    ap.add_argument("--out", help="write records and scores as JSON")
    args = ap.parse_args(argv)

    events = (dialogue.load(args.scenario) if args.scenario
              else dialogue.generate(seed=args.seed, n_sessions=args.n_sessions))
    service = MemoryService(BACKENDS[args.backend]())
    records = run(events, service, k=args.k)
    result = {"backend": args.backend, "seed": args.seed,
              "n_memories": service.count(), "scores": score(records, args.threshold, args.k),
              "latency": latency(service)}
    if args.sweep:
        result["sweep"] = {f"{t:.1f}": score(records, t, args.k)
                           for t in [i / 10 for i in range(10)]}

    s = result["scores"]
    print(f"[{args.backend}] seed={args.seed} memories={result['n_memories']} threshold={args.threshold}")
    for name, m in s.items():
        if isinstance(m, dict):
            print(f"  {name:<16} " + "  ".join(f"{key}={val:.3f}" if isinstance(val, float) else f"{key}={val}"
                                               for key, val in m.items()))
    if result["latency"]:
        print(f"  recall latency p50={result['latency']['p50_s']*1e3:.2f} ms "
              f"p95={result['latency']['p95_s']*1e3:.2f} ms")
    if args.out:
        result["records"] = records
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
    return result


if __name__ == "__main__":
    main()
