# RESULTS: LLM integration and NEST GPU backend

Experiment log for `tinyHippo-llm-gpu`. Numbered from §1; add a section per
result and do not rewrite older ones. The science log is upstream's
[RESULTS.md](RESULTS.md) (read-only here).

---

## 1. CPU reference baseline at 1% (S0 gate reference, D4 latency input)

**Machine.** Apple M3, 8 cores, macOS; NEST 3.9.0 (Homebrew, Python 3.14.2),
numpy 2.4.1, h5py 3.15.1. `.venv` created with `--system-site-packages` so it
sees the Homebrew NEST build.

**Config.** JOB H17's CA3-mode configuration (gate G1/G2 path, cortical recall
on) at 1% with 4 SWR blocks instead of 28, seed 202, 4 threads. Script:
`out/baseline/run_baseline.sh` (not committed); flags:

```
--scale 1 --threads 4 --ec-lii --ec-lii-k 50 --n-patterns 3 --seed 202
--schaffer-k 500 --schaffer-group-frac 0.7 --ca1-ec-group-frac 0.7 --cortex-group-frac 0.7
--cortical-recall --cr-cue-frac 0.4 --cr-prime-rate 120 --cr-prime-weight 1.0
--dg --dg-scale 1 --het 0.30 --het-wcomp 2.3 --w-ec-dg 0.6 --pp-residual 0.9 --dg-delay-jitter 4.0
--pattern-source ca3 --stc --n-swr 4 --epoch-ms 1000 --prp-threshold 14.0 --ec-lv --mpfc --no-figures
```

This exercises most of the `nest.*` surface the shim must route: DG, EC LII,
EC LV, mPFC, STC hook, mPFC association and recurrent hooks, the lesion and the
G2 cue trials.

**Reproducibility: not bit-identical at 4 threads on this machine.** The HDF5
files always differ byte-wise (`created_utc` attribute), so runs are compared
with a content digest from the new `h5_digest.py` (all datasets and attributes
except `created_utc`):

| run | threads | env | content digest (sha256, first 16) | first spike divergence vs. pair |
|---|---|---|---|---|
| A | 4 | | `25dd0c353ca0e3a4` | A/B: 2211.0 ms (CA1 basket) |
| B | 4 | | `bcf168e77f3802f8` | |
| C | 4 | `PYTHONHASHSEED=0` | `de9f5b39e60e3242` | C/D: 2865.2 ms (CA1 basket) |
| D | 4 | `PYTHONHASHSEED=0` | `c4f914ea6a82175f` | B/C: 2519.0 ms (CA1 basket) |
| E | 1 | `PYTHONHASHSEED=0` | `1e218550c850e2a3` | E/F: **identical** |
| F | 1 | `PYTHONHASHSEED=0` | `1e218550c850e2a3` | |

Every 4-thread pair matches spike-for-spike through epochs 1-2 (STC events
1-4 print identically), then diverges at a varying time in epoch 3 (2211-2865
ms), always first in CA1 basket cells, after which the runs differ grossly
(B has ~1.7× A's CA3/CA1 spikes; one of the four even passes the G2 assembly
size check that the others skip). Fixing Python's hash seed does not help;
there is no unseeded RNG in `replay_scaled.py`/`tiny.py`, and the per-epoch
hooks (DG residual refresh, STC, mPFC assoc/recurrent) are deterministic. The
single-thread pair is bit-identical. So the nondeterminism is in multi-threaded
simulation on this NEST build (Homebrew NEST 3.9, macOS/libomp), not in the
model's Python. Why it only shows from epoch 3 (the first epoch replaying
pattern 2 with `--n-patterns 3`) is not understood.

Consequences:
- RESULTS.md §29 (bit-identical at fixed seed, config and threads) was shown on
  MN5. It does **not** carry over to macOS with threads > 1. Worth reporting
  upstream; it is a reproducibility caveat for any local multi-thread pilot.
- **S0 gate:** compare digests from `--threads 1` runs locally (digest
  `1e218550c850e2a3536ab2750fca63520d630d2485d134ec3f0cba00147d4696` is the
  reference for this config), or on MN5 at its usual thread count after
  first checking that two unmodified MN5 runs match.
- The divergence amplifies fast, so even "statistical" CPU-vs-GPU equivalence
  (D2) needs several seeds; a single 4-thread local run is one draw.

**Timing breakdown (run A, 4 threads).**

| phase | time |
|---|---|
| core CA3/CA1 build | 2.4 s |
| DG build | 2.7 s |
| EC LII + EC LV + mPFC build | 1.6 s |
| 4 epochs × 1000 ms simulate (with per-epoch hooks) | 143.3 s (~36 s per epoch) |
| lesion + prime + 500 ms cortical-recall simulate | ~16 s |
| total wall | 169.6 s |

Build is about 4% of the run; the simulation dominates. The network has
**339,610 NEST nodes** for roughly 8k neurons: the rest are per-neuron
generators (the `sinusoidal_poisson_generator` fan-out noted in RESULTS.md
"Open items"). Real-time factor ≈ 32-35 during simulation.

**What this means for D4.** A 500 ms cue window on an already-built 1%
network costs ~16 s on 4 CPU threads, inside the L3 target of < 60 s per recall
with no GPU. A GPU recall server therefore has to beat a persistent CPU server
on repeated recall (K trials × N memories), not on single-recall latency.
The node count, not the neuron count, is likely what limits step speed on both
backends, so the generator replacement planned for Phase 0 (joint plan §3)
may speed up CPU as well.

Run A ran on an otherwise idle machine; B (311 s) and C-F overlapped with other
jobs, so only A's timings are clean. The single-thread runs took ~1960 s each,
but two ran concurrently with the C/D pair, so that is not a clean 1-thread
figure either.

**Caveats.** At 1% with 4 blocks the G2 probe usually skips (assemblies too
small) and cortical completion is near baseline; this is a timing and
reproducibility reference, not a science result. One machine, one seed.

---

## 2. L0 benchmark: stub-backend baselines

`python -m llm.bench.run_bench --backend {keyword,tfidf} --seed {0,1} --sweep`.
Default scenario: 4 sessions, each storing 4 targets, 4 distractors and 2
near-duplicates (~40 memories), then a sleep; every stored target is queried
after each session (40 positive queries per run) plus 2 never-stored
negatives per session (8 per run). k = 3. No LLM in the loop.

recall@1 / recall@3 / false-recall at confidence thresholds t:

| backend | seed | t = 0.0 | t = 0.5 | t = 0.7 |
|---|---|---|---|---|
| keyword | 0 | 0.70 / 1.00 / 0.88 | 0.57 / 0.60 / 0.38 | 0.45 / 0.45 / 0.00 |
| keyword | 1 | 0.82 / 0.93 / 0.88 | 0.68 / 0.68 / 0.75 | 0.50 / 0.50 / 0.12 |
| tfidf | 0 | 0.80 / 1.00 / 0.88 | 0.78 / 0.85 / 0.38 | 0.38 / 0.38 / 0.00 |
| tfidf | 1 | 0.75 / 0.93 / 0.88 | 0.75 / 0.78 / 0.62 | 0.45 / 0.45 / 0.00 |

recall@1 by query kind at t = 0 (seed 0 / seed 1): keyword direct 0.91 / 0.91,
near-duplicate 0.83 / 1.00, paraphrase 0.27 / 0.40; tfidf direct 1.00 / 0.73,
near-duplicate 0.83 / 1.00, paraphrase 0.55 / 0.30.

Reading: the benchmark separates the conditions it was built to separate.
Lexical baselines handle direct cues, fail on paraphrases, and cannot both
recall and reject never-stored queries at any single threshold (false recall
reaches 0 only where recall@1 has dropped below 0.5). Neither changes with
sleep, by construction, so the `sleeps=` breakdown is flat for them; that
breakdown exists for the model backend. 8 negatives per run is few: use more
seeds or `negatives_per_session` before quoting false-recall numbers. Recall
latency for both stubs is under 1 ms.
