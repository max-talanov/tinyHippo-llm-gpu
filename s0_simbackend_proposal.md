# S0: a simulator backend shim for tinyHippo (proposal)

Status: proposal for review, 2026-10-24. Nothing here is implemented yet. It asks for a decision on D0
(§6) and answers to four questions (§8) before any change to `replay_scaled.py`.

Context: [plan_llm_gpu_joint.md](plan_llm_gpu_joint.md) §4-§6 (S0 is the shim stream; D0 is where it lands),
[nest_gpu_migration.md](nest_gpu_migration.md) §3 Phase 1 (its first sketch). Phase 0 has run on a local
GPU (PR #2, `RESULTS_llm_gpu.md` §1): three of the four D1 conditions pass outright, and replay ρ moves to
Phase 2. That makes S0 the next step on the GPU path.

## 1. What S0 is for, and its one hard requirement

S0 puts one layer between the model and the simulator, so `replay_scaled.py --backend nest` and, from
Phase 2, `--backend nestgpu` run the same model code. S0 itself ships only the CPU backend; the NEST GPU
backend is Phase 2 (stream S1).

The hard requirement: **on CPU NEST, S0 must not change a single spike.** Runs are bit-identical at fixed seed,
configuration and thread count (RESULTS §29) and chaotically sensitive to any change (§16). So the shim has to
issue the same NEST calls with the same arguments in the same order, and the gate (§5.4) checks that by
comparing HDF5 outputs.

## 2. What the model asks of the simulator

A census of every NEST use in `replay_scaled.py` (223 references) and `tiny.py` (85), done for this proposal.
The surface is smaller than the reference count suggests:

| Area | What the model uses |
|---|---|
| Nodes | `Create` for `izhikevich` (15 sites), `poisson_generator` (13), `sinusoidal_poisson_generator` (2, in `tiny.py`), `spike_recorder` (15), `multimeter` (1, never read). Per-cell parameter lists under `--het`. No spike generators, parrots, `CopyModel` or `SetDefaults` |
| Connections | `fixed_indegree` (through `fixed_connect`, 36 callers), `pairwise_bernoulli` (CA3 sequence chain), `all_to_all` (grouped/clustered connects, one `Connect` per post neuron), `one_to_one` (drives). Static synapses; scalar weights and delays, or `nest.random` / `nest.math` Parameters. No STDP synapses |
| State | `GetStatus(x, key)` (`events`, synapse `source/target/weight/delay`); `SetStatus(x, key, values)` (`V_m`, `I_e`, `rate`, synapse `weight`) |
| Queries | `GetConnections` with `source=` or `target=` only (5 sites); one helper rebuilds a `SynapseCollection` from the NEST-internal `._datum` |
| NodeCollection | slicing, `+`, `.tolist()`, `len()`, `isinstance(x, nest.NodeCollection)`. No `.get/.set`, no iteration |
| Kernel | `ResetKernel` (1 site, twice per run under `--pattern-completion`); `SetKernelStatus` (resolution, threads, seed, `print_time`, `overwrite_files`); `GetKernelStatus` incl. `biological_time`; `Simulate` (5 sites: epoch loop, cortical recall, G2 probe, pattern completion, homeostasis verification) |
| Recording | Whole-run cumulative `events`, never reset; every reader windows in numpy. `gid_first = pop.tolist()[0]` exported for `decode_identity.py` |
| MPI | mpi4py rank/size, spike gather to rank 0 for HDF5 export only |
| Imports | Module-level `import nest` in both files, **22 local `import nest` inside functions**, one `from nest import GetStatus`. `reconstruct_connectivity.py` monkeypatches `nest.Simulate` and execs `replay_scaled.py` |
| Fallbacks | Exceptions used as feature detection: NEST 2 seed keys before `rng_seed`, sinusoidal → plain Poisson, multimeter `U_m`, `node_models` → `Models()` |

Hooks and when they touch the simulator:

| Hook | Build | Per epoch | Creates after the first `Simulate` |
|---|---|---|---|
| STC, mPFC assoc, mPFC recurrent | `get_conns_between` (source-side query) | read `events`, write `weight` | no |
| Schaffer STDP | `GetConnections(target=CA1 PYR)` + 4 reads | read `events`, write all weights | no |
| Homeostasis | after training: `GetConnections(target=CA3 SUP)` | write weights, then a verification `Simulate` | no |
| DG perforant plasticity, residual refresh | none | read `events` / write `rate`, `weight` | no |
| DG neurogenesis | none | **creates cells, generators, connections every epoch** | **yes** |
| Cortical recall, G2 probe | none | once: lesion, then **creates prime and cue generators**, `Simulate` | **yes** |
| Pattern completion | `ResetKernel` + full rebuild | one `Simulate` | no (fresh kernel) |

## 3. Options

| | A. Intent-level API | B. Drop-in proxy only | C. Proxy + three seams (recommended) |
|---|---|---|---|
| Idea | Rewrite all 308 call sites as backend-neutral calls (`create_izh`, `connect`, `record_spikes`, ...), as migration doc Phase 1 sketched | Replace `import nest` with a proxy; the GPU backend imitates everything the model uses | B, plus three model-level seams for the things NEST GPU cannot imitate |
| `replay_scaled.py` diff | Every NEST line (~300+) | ~25 import lines | ~25 import lines + three local refactors |
| CPU bit-identity risk | High: every line rewritten | None for the pass-through part | None for the pass-through; low for the seams (moved code, gated per commit) |
| Conflicts with upstream science merges | Every hook, every build function | Almost none | Concentrated in hook code |
| GPU fidelity | Full control | Fails: three behaviours cannot be imitated (below) | Full |

**Why B alone fails.** Phase 0 found three things that no imitation of PyNEST on NEST GPU can provide:
1. **Creating nodes or connections after the first `Simulate`.** NEST GPU refuses ("Nodes cannot be created
   after calibration"). The model does it in neurogenesis (every epoch), cortical recall and the G2 probe.
2. **`ResetKernel`.** NEST GPU has none; pattern completion rebuilds the network twice in one process.
3. **Plastic-synapse access.** `get_conns_between` relies on a NEST internal (`._datum`). And on NEST GPU,
   hook traffic is only fast through numpy buffers (Phase 0: the stock wrappers are 30× slower for 600k
   weights), and a source-target query has to be split to fit in GPU memory.

Everything else the census lists can be imitated by the GPU backend behind the PyNEST surface, without
touching the model (§5.3).

## 4. Recommendation

Option C. The proxy makes the CPU path identical by construction: on CPU the proxy forwards every call to the
real `nest` module. The three seams are the only model changes, and each one is code moved into a function
whose CPU implementation makes exactly today's calls. The GPU work then lives almost entirely in a backend
module that never touches `replay_scaled.py`.

## 5. Design

### 5.1 Layer 1: `simbackend.nest`

```
simbackend/
  __init__.py     select(name), backend(), the `nest` proxy, the seam functions (§5.2)
  cpu.py          CPU implementations of the seams (today's code, moved)
  nestgpu.py      Phase 2: PyNEST-subset backend on NEST GPU (§5.3); absent in S0
```

- `simbackend.nest` is a proxy whose attribute lookups go, at call time, to the selected backend module. The
  default is `"nest"`, so scripts that never call `select()` (calibration, plotting, `reconstruct_connectivity.py`)
  behave as today. Because lookups are dynamic, `reconstruct_connectivity.py`'s monkeypatch of `nest.Simulate`
  keeps working.
- Edits: `import nest` → `from simbackend import nest` at `replay_scaled.py` line 85, in the 22 functions that
  re-import it, and in `tiny.py`; `from nest import GetStatus as _gs` (≈6081) → `_gs = nest.GetStatus`.
- New flag `--backend {nest,nestgpu}`, default `nest`, applied with `simbackend.select()` before the first NEST
  call. In S0, `nestgpu` exits with "not implemented yet".
- `replay_scaled.py` makes no NEST call at import time, so loading NEST lazily on first use changes nothing.
- Other scripts (`bidirectional_replay*.py`, `nest_dg_ca3_fi_calibration.py`, `check_nest.py`) keep
  plain `import nest`; they are CPU-only tools.

### 5.2 Layer 2: three seams

Each seam is a small API in `simbackend`, implemented for CPU in `simbackend/cpu.py` by moving today's code
unchanged.

**Seam 1: plastic-synapse handles.**
`synapses_between(pre, post)` and `synapses_onto(target)` return a `SynapseSet` with `sources`, `targets`,
`delays` (numpy arrays, cached once) and `get_weights()`, `set_weights(array)`, `len()`.
- Callers: the STC, mPFC assoc and recurrent, Schaffer STDP, homeostasis, perforant and neurogenesis-cohort
  hooks, and the lesion.
- CPU: wraps the current `get_conns_between` / `GetConnections(target=)` plus the `GetStatus` / `SetStatus`
  calls, in the same order. `._datum` is then confined to one function in `simbackend/cpu.py`.
- GPU: chunked direct queries and numpy get/set. Phase 0 measured 0.6 s to query the STC synapses and 3 ms
  to write their weights at 12%.

**Seam 2: late creation through reservations.**
At build time, the model declares what it will create later: `reserve("neurogenesis", ...)`,
`reserve("recall_cues", ...)`, `reserve("g2_cues", ...)`. At the point of use it calls `acquire(...)`.
- CPU: `reserve` does nothing, and `acquire` runs today's `Create` / `Connect` calls at today's point in the
  run. GIDs and RNG draws are therefore unchanged.
- GPU: `reserve` creates silent cells, zero-rate generators and zero-weight synapses before `Calibrate`, and
  `acquire` switches them on.
- The GPU version draws the neurogenesis wiring at build time instead of at the epoch it is used, which is
  statistically the same but not identical. That is fine because GPU runs are compared as distributions (D2).
- Needs the maximum sizes at build time; see question 2 in §8.

**Seam 3: isolated runs.**
`run_isolated(fn, args)` wraps `run_pattern_completion`.
- CPU: calls it in-process, as now (`ResetKernel` included).
- GPU: runs it in a fresh subprocess and returns its results.

### 5.3 What the NEST GPU backend absorbs (Phase 2, not S0)

Listed so the seam boundary is clear. Each row is backed by a Phase 0 measurement (`RESULTS_llm_gpu.md` §1).

| PyNEST behaviour the model relies on | NEST GPU backend |
|---|---|
| `izhikevich` with per-cell `a, b, c, d`; `U_m` | `izhikevich_het` (project NEST GPU build); `u` |
| `pairwise_bernoulli` | `fixed_total_number` with the expected count (matches the in-degree spread) |
| `sinusoidal_poisson_generator`, generator `start`/`stop` | Rate-sliced drives updated inside the backend's `Simulate`: 1 ms chunks, rates sampled at the chunk midpoint, windows enforced by rate 0 |
| One generator per target neuron | One generator per shared-rate drive; NEST GPU gives each target its own train |
| Cumulative `spike_recorder` `events` | Recorders drained into a host store every chunk or epoch |
| `nest.random` / `nest.math` Parameters | Drawn with numpy and passed as arrays |
| `GetKernelStatus` (`biological_time`, threads, rank) | Tracked by the backend |
| NodeCollection subset (slice, `+`, `tolist`, `len`, `isinstance`) | Thin wrapper over NEST GPU's contiguous `NodeSeq` and id lists |
| Unread multimeter | Created as a no-op |
| Exceptions used as feature detection | Raised at the same points, so the model takes the same fallback branches |

GPU runs will differ statistically from CPU, not bit for bit: other RNG streams, `fixed_total_number` instead
of Bernoulli, the ripple modulation 5% weaker, and generators firing one step (0.1 ms) later. D2 compares
them as distributions over at least 2 seeds.

### 5.4 Gate

- **Reference:** each configuration below is run on the commit before S0 and after every S0 commit, in the
  same image (`nest/nest-simulator:3.9`), with the same seed and the same thread count.
- **Comparison:** a new `tools/h5_equal.py` checks that every dataset and attribute in the two HDF5 files is
  equal, ignoring only `created_utc` and wall-clock timings. Equal spike datasets mean every population's
  spike train is identical.
- **Configuration matrix at 1%:**
  1. Base: `--n-swr 1`.
  2. Gate path, CA3 mode: the JOB H15/H17 flags at 1% (`--dg --het 0.30 ... --stc --ec-lv --mpfc
     --cortex-group-frac 0.7 --cortical-recall`, 3 patterns, ≥4 epochs).
  3. Gate path, EC LII mode: the JOB H16 flags at 1%.
  4. Remaining plasticity: `--dg-neurogenesis --dg-perforant-stdp --schaffer-stdp --homeostasis`.
  5. `--pattern-completion`.
  6. `reconstruct_connectivity.py` on configuration 2 (the monkeypatch path).
- **Commit discipline:** each seam lands as its own commit with the matrix rerun, so a failure points at one
  commit.

### 5.5 What S0 does not change

- No model behaviour, existing flags or HDF5 datasets change.
- `run.sh` jobs keep working unchanged, since `--backend nest` is the default.
- The HDF5 file gains one root attribute, `backend`, which the gate ignores.

## 6. D0: where S0 lands

| | (a) Upstream first: a behaviour-neutral PR to `tinyHippo`, then merged here | (b) Here only, upstream merged in weekly |
|---|---|---|
| For | Future science changes go through the shim automatically; this repo's merges stay trivial; the gate runs once, on the reference repo | No change to the science repo |
| Against | Needs review and a gate run in `tinyHippo` | Every upstream change to hook code conflicts with seam 1 |

Recommendation: **(a)**. Under option C, the change to `replay_scaled.py` is small: imports, the `--backend`
flag and three seams. It is reviewable and bit-identical by gate, and seam 1 sits in the hook code that
science work changes most. If (a) is unwelcome, (b) works, with conflicts concentrated in that hook code.

## 7. Effort and order

Estimates, not measurements:
1. Proxy, imports, `--backend`, `tools/h5_equal.py`, reference runs: ~1 day.
2. Seam 1 (synapse handles): ~1 day.
3. Seam 3 (isolated runs): ~0.5 day.
4. Seam 2 (reservations), once its sizing is agreed: ~1 day.

Each step includes its gate run (about 1-1.5 h for the matrix at 1% on a 12-core workstation). After S0,
Phase 2 builds `simbackend/nestgpu.py` (§5.3) on the Phase 0 scripts and the July port's `nestgpu_compat.py`:
roughly 1-2 weeks, then D2.

## 8. Questions for review

1. **D0** (§6): upstream first, or here only?
2. **Seam 2 sizing.** Can the build know the maximum neurogenesis cohort per epoch, the number of epochs that
   add cohorts, and the recall/G2 cue and prime counts, from the flags alone? If neurogenesis sizes depend on
   activity, an upper bound is enough.
3. **Gate matrix** (§5.4): any configuration or flag combination missing, and is 8 threads acceptable for the
   reference runs?
4. **Scope:** agreed that `bidirectional_replay*.py` and the calibration scripts stay on plain `import nest`?

## 9. Observations from the census, for upstream (outside S0)

The census turned up three things in `replay_scaled.py` that are not shim problems but may matter for the
science. S0 changes none of them.

- **Some RNG seeds ignore `--seed`.** EC LII, EC LV, mPFC and mPFC INT draw their heterogeneity and initial
  `V_m` from fixed seeds (`default_rng(11)`, `(21)`-`(24)` and similar, ≈2845-3315). The cortical-recall cue
  uses `default_rng(7)` (≈5963) and the cue generator helper `default_rng(0)` (≈3741). Seed replication
  therefore does not vary these. Worth checking that this is intended.
- **Homeostasis verification windows ignore recall time.** `epoch_t0_verify = (n_epochs + idx) * SIM_MS`
  (≈6061) does not count the time simulated by cortical recall and G2, which run before the homeostasis sweep.
  With `--homeostasis` and `--cortical-recall` together, the verification windows would sit at the wrong times.
- **An unread multimeter.** The multimeter created at ≈1538 is never read. It is harmless, and S0 keeps it,
  since removing it would shift GIDs.
