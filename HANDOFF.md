# Handoff: starting work in tinyHippo-llm-gpu

Written 2026-10-24 for a fresh Claude Code session. Nothing in the old session's memory carries over
(Claude's per-project memory is keyed by directory), so everything needed is here.

## 1. What this repo is

`max-talanov/tinyHippo-llm-gpu` is a fork of `max-talanov/tinyHippo` (remote `upstream`), cut at commit
`06a3a62`. It exists for two things that must not be mixed into the science repo:

1. **A NEST GPU backend** for the tinyHippo model (spiking rat hippocampus, Izhikevich neurons).
2. **An LLM episodic-memory integration**: an LLM agent calls `remember` / `sleep` / `recall`; the model
   stores the *index* (a sparse code per memory ID), replays and consolidates it, and returns the right ID
   from a partial cue. Text lives in a normal store (SQLite).

`tinyHippo` keeps the science, the gate runs, `RESULTS.md`, and the **bio-plasticity** work on CPU NEST.

## 2. Read these, in order

1. [plan_llm_gpu_joint.md](plan_llm_gpu_joint.md): repo layout, sync policy, schedule, decisions D0-D4, service design. **Authoritative.**
2. [nest_gpu_migration.md](nest_gpu_migration.md): what the code needs from NEST, what NEST GPU offers, Phases 0-5.
3. [plan_branch_llm_integration.md](plan_branch_llm_integration.md): LLM phases L0-L6, design, baselines.
4. [CLAUDE.md](CLAUDE.md): inherited project goal and **methodological rules** (they apply to every result here).
5. `RESULTS.md` (upstream's log, read-only here), especially §22-§33 and "Open items".

## 3. State of play

- **Entry gate not met.** G1 (pattern ID decodable from mPFC/EC LV, p < 0.01, >=2 seeds), G2 (lesion + partial
  cue reactivates the right pattern), G3 (claims 1-2 still hold). The deciding 12% runs are JOB H15 (G1),
  H16 (EC LII path), H17 (G2 probe), queued on MareNostrum 5 by the user. Until they pass, do **not** create
  `llm-integration` or build on L1+. Last known: G1 only weakly positive on one seed (RESULTS §28).
- **No code written yet** in this repo beyond the plans. No `simbackend.py`, no `llm/`, no `gpu_spike/`.
- Decisions made: only this repo targets NEST GPU; `bio-plasticity` stays CPU in `tinyHippo`; GPU use is
  *conditional* on gates D1, D2, D4 (joint plan §6). A CPU persistent server is an acceptable outcome.
- Open decision for the user: **D0**, whether to upstream the shim to `tinyHippo`. Default: develop here,
  merge upstream weekly. D3 (how memories map to pattern groups) must be settled before L1.

## 4. NEST GPU facts (read from `nest/nest-gpu` source on 2026-10-09)

Confirmed in source: `izhikevich` exists (`src/izhikevich.cu`: same equations, `I_e` in pA, spike adds weight
to V, forward Euler, `a,b,c,d,V_th` are **group** params). Connection rules: one_to_one, all_to_all,
fixed_total_number, fixed_indegree, fixed_outdegree, assigned_nodes. Devices: `poisson_generator` (rate,
unique train per target), `spike_generator`, parrot, multimeter via `CreateRecord`, spikes via
`ActivateRecSpikeTimes` / `GetRecSpikeTimes`. One `stdp` synapse (pair STDP). Multi-GPU via MPI,
`RemoteCreate` / `RemoteConnect`. Linux + NVIDIA + CUDA only. GPL-2.0.

Known gaps vs the code: no `sinusoidal_poisson_generator` / `inhomogeneous_poisson_generator` (theta and ripple
in `tiny.py`), no kernel reset, per-neuron `b` heterogeneity unsupported on the stock model.

**Not verified (Phase 0 must settle):** per-neuron `b` workaround; inhibitory sign on the single port; delay
arrays; poisson per-target independence; whether `Create`/`Connect` works after the first `Simulate`; cost of
`GetConnections` / `GetConnectionFloatParam` / `SetConnectionStatus` at 12%-size (~19M synapses); NESTML GPU
target maturity; STDP type details; whether a GPU is allocatable on MN5 `uab100` / `acc_resc`.

## 5. First actions, in order

1. **Environment.** Local machine is macOS: no CUDA, so NEST GPU cannot run locally. CPU NEST 3.9 worked in
   the old repo's `.venv` (verify with `python -c "import nest; print(nest.__version__)"`; this repo has no
   `.venv` yet, create one from `requirements.txt`). Confirm a GPU node on MN5 (`-A uab100 -q acc_resc`,
   as `run.sh` already submits) and CUDA + OpenMPI modules.
2. **L0 (no model dependency, start now).** Branch `llm-l0`: create `llm/` with `memory_service.py` (SQLite
   content store + backend interface), `agent.py` (Claude tool-use agent, tools `remember` / `recall` /
   `sleep`), `backends/stub.py` (exact match or vector store), `bench/` (multi-session dialogue with
   distractors and near-duplicates; recall@k, false-recall, latency). Use the `claude-api` skill; default model
   `claude-opus-5-5`.
3. **Phase 0 GPU spike** (branch `nest-gpu`, scripts in `gpu_spike/`, do not touch `replay_scaled.py`):
   migration doc §3 Phase 0, including the theta/ripple generator replacement and the connection get/set
   benchmark. Write the numbers to `RESULTS_llm_gpu.md` §1. Decide **D1**.
4. **S0 shim** (`simbackend.py`, `--backend nest` default), regardless of D1; gate is a bit-identical HDF5 at
   fixed seed and threads (RESULTS §29). Keep RNG call order unchanged.
5. Weekly: `git fetch upstream && git merge upstream/main`.

## 6. Rules that bite

- Score replay on the SWR window `(win[0], win[1])`, never padded.
- Rheobase by DC current, not Poisson drive: `I_rheo = (5-b)²/0.16 - 140 - I_e`. Keep `b` heterogeneity small
  and below `b_crit`. Calibrated neuron params live in `nest_dg_ca3_fi_calibration.py` (`CANDIDATE_PARAMS`;
  `mc_high I_e = -15.1`).
- CA3 SUP and CA1 PYR run near saturation: use rate-based readouts (`decode_identity.py`), not Jaccard.
- Read cell positions from each population's `gid_first`, never the lowest spiking GID.
- DG is chaotic across blocks; CPU runs are bit-identical only at fixed seed, config, thread count. GPU runs
  will not be, so compare distributions over >=2 seeds. Single-seed or 1%-only results are pilots.
- Never `GetConnections(source=, target=)` (whole-kernel scan, hours at 12%); use `target=` or cached handles.
- Do not rewrite older `RESULTS.md` sections; here, add to `RESULTS_llm_gpu.md`.

## 7. Practical facts

- GitHub account `max-talanov`; both repos are public. Remotes: `origin` (this repo), `upstream` (`tinyHippo`).
- MN5 results the user uploads land in `res/<date>/`; ad hoc local output in `out/`. Neither is committed
  (and neither was copied into this clone).
- `run.sh` is the SLURM launcher; JOB descriptions are in its header. A GPU launcher `run_gpu.sh` does not exist yet.
- Commit trailer used so far: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## 8. Suggested opening prompt for the new session

> Read HANDOFF.md and plan_llm_gpu_joint.md. Start L0 on branch `llm-l0` (llm/ skeleton, stub backend,
> benchmark), and in parallel set up the Phase 0 NEST GPU spike plan for an MN5 GPU node. Ask me before
> creating any branch from the gate commit or touching `replay_scaled.py`.
