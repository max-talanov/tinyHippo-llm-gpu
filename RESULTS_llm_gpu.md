# Results: tinyHippo-llm-gpu

Numbered log for this repo's GPU and LLM work (joint plan §0). Upstream's `RESULTS.md` stays the record for
the science; sections here are not mirrored there. Add a new section per result; do not rewrite older ones.

## 1. Phase 0 on the latest NEST GPU: the simulator passes, and two of its blockers are fixed in a project branch

Date 2026-10-24. Hardware: RTX 4060 Ti 16 GB (compute capability 8.9), Ryzen 9 5900X, Docker Desktop
(WSL2). CPU reference: NEST 3.9.0 (`nest/nest-simulator:3.9`). GPU: NEST GPU 2.0.0 built from
`nest/nest-gpu` `main` (`ba52eb9`, 2026-10-20) plus two project commits on `tfatykhov/nest-gpu@tinyhippo`
([gpu/README.md](gpu/README.md)). Scripts are in `gpu_spike/`; every number below comes from them.
These are simulator checks with single runs, not model results: they settle what the GPU backend can rely on,
not replay or consolidation on the GPU (that is Phase 2, scored over ≥2 seeds).

### The simulator, upgraded from the July port's `90f87ab`

`compat_recheck.py` reran each behaviour the July port (`tfatykhov/tinyHippo@feat/gpu-port`) worked around,
on both the old and the new build. Upstream's 116 newer commits change none of them:

| Behaviour | 90f87ab | ba52eb9 | Consequence for the backend |
|---|---|---|---|
| Poisson output vs `Simulate()` chunk (800 Hz, 1 s; parrot spikes, at most one per step) | 376 / 697 / 780 / 780 at 0.5 / 0.9 / 1.0 / 1000 ms | same | Chunks must be ≥ 10 steps (1 ms at dt 0.1) |
| `spike_generator` advanced in 1 ms chunks | 0 / 264 targets fire (264 / 264 in one call) | same | Precomputed trains cannot be mixed with rate slicing |
| Generator `start`/`stop` 60-80 ms at 2 kHz | 35 spikes inside, 758 after | same | Gate drives by writing `rate = 0` |
| `GetRecSpikeTimes` | drains: 60, then 0, then 55 | same | Drain into a host store every epoch |
| Per-synapse weight write from an array | segfault | segfault | **Fixed in `b0c1f1a`** (below) |
| `GetConnections` before `Calibrate()` | CUDA error | CUDA error | Build hooks after `Calibrate()` |
| `pairwise_bernoulli` | rejected | rejected | Use `fixed_total_number` (July port) |
| `sinusoidal_poisson_generator` | absent | absent | Rate slicing (below) |
| `Create`/`Connect` after the first `Simulate` | refused: "Nodes cannot be created after calibration" | same | Neurogenesis pool and `remember()` must be preallocated (joint plan §7) |
| One `poisson_generator` onto 50 targets | not tested in July | 50 distinct trains, 100.5 Hz at 100 Hz | One generator per shared-rate drive, not per neuron |

The last row matters for speed: CPU NEST holds 760,442 nodes for an 8,450-neuron 1% network, almost all of
them generators created one per target neuron. On the GPU, a drive whose targets share a rate (background,
theta, ripple, sharp wave) needs one generator; only per-cell rate profiles such as EC LII's place fields
still need one per cell.

### Per-neuron Izhikevich parameters: solved with a new model, `izhikevich_het`

Stock `izhikevich` keeps `V_th, a, b, c, d` per group and rejects per-neuron arrays for them; every gate run
(`--het 0.30`) draws `a, b, c, d` per cell. Commit `509657d` adds `izhikevich_het`, the same kernel with those
five as per-neuron parameters (`t_ref` stays per group). `izh_het_check.py`:

- `V_th, a, b, c, d` accept per-neuron arrays on `izhikevich_het` and are rejected on `izhikevich`.
- With identical parameters, 200/200 neurons give identical spike trains on both models (6,354 spikes each;
  DC drive plus a shared input burst).
- A per-cell `b` sets that cell's threshold: the first-spike DC threshold for b = 0.10 / 0.15 / 0.20 / 0.24 is
  9.90 / 6.70 / 3.45 / 0.90 pA on `izhikevich_het` and exactly the same on stock populations with that group
  `b`. (Below the analytic tonic rheobase, 10.06 / 7.02 / 4.00 / 1.61 pA, because the cells start away from
  rest and one onset spike counts; the point is that the two models agree.)

### f-I curves match CPU NEST

`fi_curves.py`: DC sweep 0-40 pA in 0.25 pA steps on top of each population's `I_e`, 1.2 s, rate over the
last 1 s, dt 0.1 ms, for every Izhikevich parameter set the model uses.

| Set (populations) | Rheobase CPU / GPU (pA) | Rate at +20 pA CPU / GPU (Hz) | Largest difference |
|---|---|---|---|
| `rs_d8` (CA3 SUP, CA1 PYR, DG GC, MC LOW, mPFC) | 3.5 / 3.5 | 43 / 43 | 0 Hz |
| `ec_lii` (EC LII) | 3.5 / 3.5 | 54 / 54 | 1 Hz |
| `deep_lv` (CA3 DEEP, EC LV) | 0.5 / 0.5 | 90 / 90 | 1 Hz |
| `basket` (all baskets, mPFC INT) | 3.75 / 3.75 | 304 / 304 | 1 Hz |
| `olm` (CA1 OLM) | 0.25 / 0.25 | 147 / 147 | 1 Hz |
| `mc_high` (MC HIGH) | 18.75 / 18.75 | 10 / 10 | 0 Hz |

The largest difference anywhere on the six curves is one spike in the 1 s window; mean relative difference
≤ 0.1%. `izhikevich_het` gives the same curves. The ~5% recalibration the migration plan feared is not needed:
both simulators integrate with the same forward-Euler step.

### Synaptic sign, delays and latency

`delay_sign_check.py`: weights of both signs and per-synapse delays given as arrays arrive with the right sign
and size; delays are stored rounded to the 0.1 ms grid (5.33 → 5.3, 7.77 → 7.8) on both simulators.
Neuron-to-neuron latency, spike to spike, is delay + 0.1 ms on both. Two offsets differ but do not accumulate:
a `spike_generator` emits one step later on the GPU (its target fires at 11.2 ms vs 11.1 ms), and the GPU
multimeter samples `V_m` one step later.

### Theta and ripple drive without `sinusoidal_poisson_generator`

`theta_ripple_check.py`: the model's two oscillatory drives (replay_scaled.py defaults) feed 500 parrot
neurons; mean rate and the first Fourier component are measured over the drive window. CPU NEST uses
`sinusoidal_poisson_generator`; the GPU uses one `poisson_generator` whose rate is rewritten before every 1 ms
`Simulate()` chunk, sampled at the start of the chunk (as the July port did) or at its midpoint.

| Drive | Variant | Rate (Hz) | Modulation (Hz) | Phase (°) |
|---|---|---:|---:|---:|
| theta 8 Hz | target | 1100 | 1000 | 0 |
| | CPU `sinusoidal_poisson_generator` | 1099.1 | 1000.4 | 359.9 |
| | GPU, sampled at chunk start | 1098.6 | 1000.1 | 358.7 |
| | GPU, sampled at chunk midpoint | 1098.8 | 999.9 | 0.2 |
| ripple 180 Hz, 300-420 ms | target | 1100 | 850 | 0 |
| | CPU `sinusoidal_poisson_generator` | 1115.2 | 883.0 | 359.2 |
| | GPU, sampled at chunk start | 1116.5 | 833.1 | 331.8 |
| | GPU, sampled at chunk midpoint | 1116.1 | 835.2 | 2.4 |

Theta is reproduced exactly. The ripple keeps its rate and its phase if the rate is sampled at the chunk
midpoint; sampling at the chunk start, as the July port did, delays the ripple by half a chunk (28° at
180 Hz). The ripple modulation is 5% below CPU NEST's (835 vs 883 Hz), which is the attenuation expected
from holding the rate for 1 ms (0.947 at 180 Hz); both backends sit about 30 Hz above their expected value,
a bias of the estimator over a 120 ms window. The backend must sample at the midpoint. Whether this keeps replay ρ in its band is a Phase 2 question, since it needs the full model.

Measurement note: a NEST GPU `parrot_neuron` emits one spike per step however many arrive, unless
`hold_spike_mul` is set, so at these rates its spike times undercount by 5-8%. The GPU numbers above count
the per-step input multiplicity (the parrot's `V`) instead.

### Connections at 12% size: build, query, read and write weights

`conn_bench.py` builds the 12% production architecture as counted in upstream RESULTS §22
(`plot_architecture.py`: 15 populations, 38 projections) with placeholder neuron parameters and weights:
271,046 neurons and 80,182,025 synapses, plus one Poisson generator per population. It measures the
simulator's connection machinery, not the model.

| Operation | Time |
|---|---:|
| Create + Connect + Calibrate, 80.2M synapses | 0.2 s (GPU memory 1.70 GB) |
| Query CA1 PYR → EC LII (STC hook), 600,250 synapses | 0.55 s |
| Query EC LV → mPFC (association hook), 28,800 synapses | 0.02 s |
| Query CA3 SUP → CA1 PYR (Schaffer), 27,597,500 synapses | 1.99 s |
| Read 600k weights / all 27.6M Schaffer weights | 0.003 s / 0.115 s |
| Write one value to 1M synapses | 0.002 s |
| Write 600k per-synapse weights from an array (after `b0c1f1a`), read back exact | 0.003 s |
| Same write by the July port's quantised workaround (101 calls) | 0.033 s |
| Simulate 100 ms (~10 Hz everywhere), one call / 1 ms chunks with rate updates | 1.18 s / 1.30 s |

For comparison, upstream RESULTS §24: on MN5 CPU the three hook lookups of JOB H10 took 5.3 h before the
source-side fix and minutes after it; a 12% run simulates 14 s of model time in about 5.4 h on 50 cores. The
GPU's 1.2 s per 100 ms on this placeholder network suggests two orders of magnitude, but the real model's
speed is a Phase 2 measurement: its activity and drive structure differ.

Two findings for the backend:

- **NEST GPU's `GetConnections(source, target)` allocates three device arrays of n_source × n_target
  entries (24 bytes per pair), whatever the synapse count.** CA3 SUP × CA1 PYR at 12% is 1.75e9 pairs
  (~42 GB) and runs out of memory; the STC query (6.6e8 pairs, ~16 GB) only just fits on 16 GB. The backend
  splits the source range into chunks of ≤ 1e8 pairs (`get_conns_direct`), which costs little because each
  chunk is one pass over the connection array. A filter-based `getConnections` in the NEST GPU branch would
  remove the problem.
- **The stock Python wrappers convert ids and values element by element** (0.094 s vs 0.003 s for 600k
  weights). The backend calls the C entry points with numpy buffers.

Weight write-back no longer needs quantisation: commit `b0c1f1a` fixes the null `Distribution` pointer that
made `SetConnectionFloatParamDistr` segfault, so per-synapse arrays write exactly.

### D1

Joint plan §6 sets four conditions for continuing on the GPU. At the simulator level all four hold:

| D1 condition | Result |
|---|---|
| f-I curves within ~5% after recalibration | Within one spike per second, identical rheobase, no recalibration needed |
| Per-neuron `b` heterogeneity solved | `izhikevich_het` (per-neuron `V_th, a, b, c, d`), identical to `izhikevich` at equal parameters |
| Connection get/set for 1M synapses under ~1 min at 12% | Milliseconds; queries ≤ 2 s once chunked |
| Theta and ripple reproduced well enough to keep replay ρ in band | Theta exact; ripple rate and phase exact with midpoint sampling, modulation −5% vs CPU. Replay ρ itself is untested on the current model |

Recommendation: D1 passes; start S0 (shim) and Phase 2. The open risk moves to Phase 2: replay ρ, DG
sparsity and the home-group code on the full current model, compared with CPU NEST over ≥2 seeds. The
backend requirements found here: `izhikevich_het` wherever `--het` is on, rate sampling at chunk midpoints with
1 ms chunks, one Poisson generator per shared-rate drive, chunked connection queries, direct C calls with numpy
buffers, and a preallocated network (no `Create` after the first `Simulate`).

This supersedes the per-neuron and weight-write parts of `local_gpu_assessment.md` (PR #1, §3 and §5
there), which were written before these measurements.
