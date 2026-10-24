# Local GPU assessment: what the joint plan can do now

Status: notes, 2026-10-24. A review of [plan_llm_gpu_joint.md](plan_llm_gpu_joint.md),
[nest_gpu_migration.md](nest_gpu_migration.md) and [HANDOFF.md](HANDOFF.md) against the hardware that is
available and against earlier NEST GPU work. No model code is changed. Corrections to the plan documents are
proposed in §7, not applied.

## 1. Summary

- **A CUDA GPU is available, and NEST GPU 2.0 runs on it.** The development workstation has an RTX 4060 Ti
  (16 GB, compute capability 8.9). NEST GPU, built in Docker, passes its smoke test. Phase 0 and decision D1
  therefore do not wait for an MN5 GPU node; MN5 GPU access matters again at Phase 4 (multi-GPU, production 12%).
- **Earlier GPU work exists** and the plans do not mention it: a port of the replay model to NEST GPU,
  `tfatykhov/tinyHippo` branch `feat/gpu-port` (commit `4a157da`, 2026-07-27). It settles several Phase 0
  questions by measurement (§3). It is built on a much older model, so the reusable part is its compatibility
  layer, not its model file.
- **Per-neuron `a`, `b`, `c`, `d` are not supported by NEST GPU's `izhikevich`; measured, not only read from the
  source** (§5). The gate runs use `--het 0.30`, which spreads `a`, `c` and `d` per cell, so this is the main D1 risk.
  `I_e` and `V_m` are per neuron and work.
- **First timing point** (§6): at 1% on the base network, the GPU simulates 1.7x faster than CPU NEST (21.6 s vs
  36.3 s for 1 s of model time), with the same wall time. The case for the GPU rests on 12% and on repeated
  recall; it is not made at 1%.
- **Two plan numbers are off**: 12% is about 80.2M synapses (RESULTS §22), not ~19M; and the shim covers 308
  `nest.*` call sites, not ~190.

## 2. Hardware (checked 2026-10-24)

| Item | Finding |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti, 16 GB, compute capability 8.9 (Ada) |
| GPU in containers | Docker Desktop (WSL2 backend) passes the GPU through; `nvidia-smi` works inside `nvidia/cuda:12.4.1` images |
| NEST GPU | 2.0.0, built from `nest/nest-gpu` `90f87ab` with CUDA 12.4, `-Dwith-gpu-arch=89`, MPI on. The bundled smoke test (500 neurons, 200 ms) passes |
| NEST GPU upstream | `nest/nest-gpu` `main` is 116 commits ahead (`ba52eb9`, 2026-10-20). The only change to `izhikevich.{h,cu}` is the clang-format commit `6f3e609`, so the parameter layout in §5 holds at HEAD |
| CPU | Ryzen 9 5900X, 12 cores / 24 threads, 32 GB RAM |
| CPU NEST reference | NEST 3.9.0 (Docker image `nest/nest-simulator:3.9`) runs the current `replay_scaled.py` at 1% (§6) |
| MN5 | No MN5 host is configured on this workstation, so GPU allocation on `uab100` / `acc_resc` is still unverified |

## 3. The earlier GPU port

`tfatykhov/tinyHippo@feat/gpu-port` adds a self-contained `gpu/` folder: `replay_gpu.py` (the model),
`nestgpu_compat.py` (every NEST to NEST GPU deviation, each with the measurement behind it), a Dockerfile pinned to
NEST GPU `90f87ab`, a compose file and `run_gpu.sh` (same environment-variable contract as `run.sh`). Its README
records nine findings, each forced by NEST GPU:

| Finding (measured in the port) | Consequence for this plan |
|---|---|
| No `sinusoidal_poisson_generator`; a `poisson_generator` whose `rate` is rewritten from Python every 1 ms reproduces theta and ripple drive | Answers the joint plan §3 gap 1, option (a). Must be re-validated on the current model |
| `spike_generator` emits nothing when the run is advanced in chunks (264/264 neurons fire under one `Simulate`, 0/264 chunked) | Joint plan §3 option (b), precomputed spike trains, cannot be combined with rate slicing |
| `Simulate()` shorter than 10 time steps silently kills Poisson drive | 1 ms is the chunk floor at dt 0.1 ms |
| Generator `start`/`stop` keeps firing after `stop` | Windowed drives are gated by setting `rate = 0` |
| No `pairwise_bernoulli`; `fixed_total_number` reproduces Bernoulli's in-degree spread, `fixed_indegree` does not | Affects the CA3 sequence chain and the DEEP to SUP projection |
| Per-synapse weight write through an array segfaults | Writes are grouped by weight quantised to 1e-3 mV: 50k synapses in 0.126 s |
| `GetConnections` before `Calibrate()` is an illegal memory access | Hooks are built after `Calibrate()` |
| `GetRecSpikeTimes` drains the buffer; the per-neuron buffer is fixed size | Recorders drain every epoch into a host store |
| `a`, `b`, `c`, `d`, `V_th`, `t_ref` are group parameters | See §5 |

**It is built on an old model.** The port follows `replay_scaled.py` at `8543f32` (2,558 lines). This repo is 145
commits later and the file is 6,219 lines (+3,838 / -177). The port has no DG module, no `--*-group-frac` flags,
no EC LII to CA3 path, no `--het`, no pattern sources and no G2 probe, so it cannot run any of the JOB H15-H17
configurations. Its validation (README table) is one seed, 1%, 1 s, which is a pilot under this project's rules.

**Recommendation:** lift `nestgpu_compat.py` and its findings into the `simbackend` NEST GPU backend (S1); do not
revive `replay_gpu.py`, which would be a second copy of the model to keep in sync.

## 4. Phase 0 checklist (migration doc §3)

| Phase 0 item | Status | Remaining work |
|---|---|---|
| 1. Build NEST GPU | Done | none |
| 2. f-I curves within 5% of CPU NEST | Not done; the port compared population rates only | DC rheobase and f-I for the DG/CA3 parameter sets on both backends |
| 3. Per-neuron heterogeneity | Gap confirmed (§5) | Pick and test a fix |
| 4. Inhibitory sign, delay arrays | Sign works in the port (all interneuron populations); delay arrays untested | Test per-synapse delays (`DG_DELAY_JITTER=4.0` is in every gate config) |
| 5. Connect / get / set benchmark at 12% | Not done | Target is ~80M synapses, not ~19M (RESULTS §22). Scaled linearly, the port's quantised write gives ~1.5 s for the 600k STC synapses; unmeasured at scale |
| 6. `Create`/`Connect` after `Simulate` | Not tested | Needed for the neurogenesis hook and for `remember()` design (joint plan §7) |
| 7. Spike recording | Solved in the port | none |
| Theta / ripple replacement | Solved on the old model | Re-check replay ρ band on the current model, at least 2 seeds |

All four D1 criteria (f-I within ~5%, per-neuron heterogeneity, get/set under ~1 min at 12%, replay ρ in band)
can be measured on the local GPU.

Memory is unlikely to limit 12%: NEST GPU reports 1B synapses on an 11 GB card, which puts 80M synapses near
1 GB. This is an estimate; item 5 measures it.

## 5. Per-neuron Izhikevich parameters

In NEST GPU's `src/izhikevich.h`, the per-neuron parameters are `I_e` and `den_delay`; `V_th`, `a`, `b`, `c`, `d`
and `t_ref` are per group (one value per `Create`). A probe on the local GPU (4 neurons, `SetStatus` with
`{"array": [...]}` then `GetStatus`) gives:

```
I_e  [per-neuron param] array set -> read back [0.0, 1.0, 2.0, 3.0]
a    [GROUP param] array set -> REJECTED (cannot be initialized by arrays or distributions)
b    [GROUP param] array set -> REJECTED
c    [GROUP param] array set -> REJECTED
d    [GROUP param] array set -> REJECTED
V_m  [per-neuron var] array set -> read back [-70.0, -69.0, -68.0, -67.0]
```

Why it matters: every gate configuration (JOB H15-H17) runs with `HET=0.30`, i.e. `--het 0.30`, which also sets
the neuron-parameter CV (`--het-neuron-cv` defaults to `--het`). `het_params` then draws per cell:

| Parameter | Spread at cv 0.30 | Supported on NEST GPU |
|---|---|---|
| `a`, `d` | 30% relative | no |
| `c` | 1.5 mV absolute (clipped to -80..-55 mV) | no |
| `b` | 3% relative (`B_CV_FACTOR = 0.10`), capped at `b_crit - 0.02`, skipped at the cap | no, but small by design |
| `I_e` | 0.9 pA absolute | yes |

So per-neuron `b` is nearly a non-issue (the code already carries excitability spread on `I_e`), and the real gap
is `a`, `c` and `d`. Options:

1. **A copy of `izhikevich` in a `user_m1` slot with `a`, `b`, `c`, `d` as per-neuron parameters** (recommended).
   Faithful to `het_params`, leaves the stock model untouched, and costs four floats per neuron. Needs a small
   CUDA change in a NEST GPU fork and a rebuild.
2. **K sub-groups per population**, each with its own `a, b, c, d`. No CUDA change, but it quantises the
   distributions, multiplies `Create` calls, and every `Connect` must accept the concatenated range (check that
   `gid_first` stays the first cell of the population).
3. Running the GPU with `--het-neuron-cv 0` changes the model, so it is only useful as a diagnostic, not for
   equivalence (D2).

## 6. First timing point

1% scale, base network (CA3 + CA1, 8,450 neurons, no DG / EC / STC), one SWR epoch of 1000 ms, seed 42.

| | CPU NEST 3.9, 12 threads | NEST GPU 2.0 (the port) |
|---|---:|---:|
| Build | 2.5 s | 0.2 s (+ `Calibrate`) |
| `Simulate` for 1000 ms | 36.3 s | 21.6 s |
| Wall time | 43 s | 34 s in-script, 43 s with container start |
| Forward replay ρ (SWR-1) | +0.770 PASS | +0.976 PASS |
| Reverse replay ρ (SWR-2) | -0.309 WEAK | -0.127 WEAK |

Caveats: the two runs use different model versions (current `replay_scaled.py` vs the port's base), one seed
each. The GPU run advances in 1000 one-millisecond `Simulate` calls with Python rate updates in between, so it
measures that overhead as much as the GPU. The CPU kernel holds 760,442 nodes for 8,450 neurons; almost all of
them are per-target generators, which NEST steps every time step.

Rerunning the port today gives the same ρ values, to three decimals, as its July README table (+0.976 / -0.127).

For D4 (≥3x on repeated recall at 1%), this base network does not reach it. Settle D4 at 12% and on the recall
loop itself, where build cost is paid once.

## 7. Corrections to the plan documents (proposed, not applied here)

| Document | Text | Correction |
|---|---|---|
| HANDOFF.md §5.1 | "Local machine is macOS: no CUDA, so NEST GPU cannot run locally." | A CUDA GPU is available; Phase 0 runs locally |
| HANDOFF.md §4, migration doc §3 item 5 | "12%-size (~19M synapses)" | ~80.2M synapses (RESULTS §22) |
| Joint plan §4, §5; migration doc §3 Phase 1 | "~190 `nest.*` call sites" | 223 in `replay_scaled.py` + 85 in `tiny.py` = 308 |
| Joint plan §3 gap 1, option (b) | precomputed `spike_generator` trains | Not usable together with rate slicing (§3) |
| Joint plan §5 week 1 | "Phase 0 spike on an MN5 GPU node" | Local GPU; the MN5 GPU check moves to Phase 4 |
| Joint plan §9, last row | fallback "a Linux/NVIDIA workstation" | The workstation is the primary path for Phase 0-3 at 1%, and probably 12% on one GPU |
| Migration doc §1 | per-neuron `b` **[verify]** | Verified unsupported; the gap is `a`, `c`, `d` (§5) |
| HANDOFF.md, migration doc | no mention of earlier work | Reference `tfatykhov/tinyHippo@feat/gpu-port` and its findings (§3) |

## 8. What can proceed now, and what is blocked

Can proceed now:
- **Phase 0** (S1, branch `nest-gpu`, scripts in `gpu_spike/`): the items left in §4. Record wall time per
  phase (build, `Calibrate`, `Simulate` per epoch, each hook) so D4 has data.
- **S0 shim** against CPU NEST 3.9, with the bit-identical HDF5 gate at fixed seed and threads.
- **Phase 2 at 1%** once S0 lands: the NEST GPU backend from `nestgpu_compat.py`, DG through replay, scored with
  `decode_identity.py` against CPU runs over at least 2 seeds.
- **L0** (`llm/` skeleton, stub backend, benchmark): no model dependency. An API key is needed only for agent runs.

Blocked:
- **L1 onward**: waits for JOB H15-H17 to pass upstream and the gate commit to be merged here.
- **D3** (memory to group mapping): a model owner's decision. **D0** (upstream the shim): the user's.
- **D4**: needs 12% and recall-loop timings, which need Phase 2-3.
- **Phase 4** (multi-GPU, scaling ladder): needs MN5 GPU access, unverified.

## 9. Recommended order

1. Phase 0 on the local GPU, starting with the per-neuron parameter fix (§5, option 1) and the 80M-synapse
   connect / get / set benchmark: these two decide D1.
2. S0 shim in parallel; budget for 308 call sites.
3. D1 at the end of Phase 0 from local numbers.
4. Phase 2 at 1% on top of S0, reusing `nestgpu_compat.py`.
5. L0 in parallel throughout.

One option opens up on the GPU: rate-sliced generators make a per-epoch ripple background cheap. That background
is still epoch-0 only on CPU because it needs ~16k `sinusoidal_poisson_generator`s per window (RESULTS, Open
items). Changing it changes the model, so it belongs upstream in `tinyHippo`, not here.
