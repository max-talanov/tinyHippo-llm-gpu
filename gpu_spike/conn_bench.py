"""Connection build / query / weight get-set benchmark at 12% size on NEST GPU.

Phase 0 item 5 (nest_gpu_migration.md §3) and a D1 criterion: plasticity hooks
must be able to read and write their synapses in well under a minute at 12%.

The network copies the 12% production architecture counted in RESULTS.md §22
(plot_architecture.py: 15 populations, 38 projections, ~80M synapses). Neuron
parameters and weights are placeholders, not the model's: this measures the
simulator's connection machinery, not the model's dynamics.

Two access paths are timed for each operation:
  * `wrapper`: the stock nestgpu.py functions, which convert ids and values
    through Python lists element by element;
  * `direct`: the same C entry points called with numpy buffers, which is what
    a simbackend implementation would do.

    python3 gpu_spike/conn_bench.py [--out out/phase0/conn_bench.json]
"""
import argparse
import ctypes
import json
import os
import sys
import time

import numpy as np
import nestgpu as ngpu

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from plot_architecture import POPS, PROJ, SELF  # noqa: E402  (12% counts, RESULTS §22)

lib = ngpu._nestgpu
INHIBITORY = {"DG BSK", "INT SUP", "INT DEEP", "CA1 BSK", "CA1 OLM", "mPFC INT"}
W_EXC, W_INH, DELAY = 0.05, -0.2, 1.5


def _cstr(s):
    return ctypes.create_string_buffer(s.encode())


def get_conns_direct(src, tgt, max_pairs=100_000_000):
    """Connection ids between two NodeSeqs, as an int64 numpy array.

    NEST GPU's getConnections allocates three device arrays of
    n_source * n_target entries (24 bytes per source-target pair), whatever
    the number of synapses. CA3 SUP x CA1 PYR at 12% is 1.75e9 pairs (~42 GB),
    so the source range is split into chunks of at most max_pairs pairs.
    """
    step = max(1, int(max_pairs // max(tgt.n, 1)))
    parts = []
    for i0 in range(src.i0, src.i0 + src.n, step):
        n_src = min(step, src.i0 + src.n - i0)
        n = ctypes.c_int64(0)
        ptr = lib.NESTGPU_GetSeqSeqConnections(i0, n_src, tgt.i0, tgt.n, -1,
                                               ctypes.byref(n))
        if ngpu.GetErrorCode() != 0:
            raise RuntimeError(ngpu.GetErrorMessage())
        if n.value:
            parts.append(np.ctypeslib.as_array(ptr, shape=(n.value,)).copy())
    return np.concatenate(parts) if parts else np.zeros(0, np.int64)


def get_weights_direct(ids):
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    out = np.empty(len(ids), dtype=np.float32)
    lib.NESTGPU_GetConnectionFloatParam(ids.ctypes.data_as(ngpu.c_int64_p), len(ids),
                                        out.ctypes.data_as(ngpu.c_float_p), _cstr("weight"))
    return out


def set_weight_scalar_direct(ids, value):
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    lib.NESTGPU_SetConnectionFloatParam(ids.ctypes.data_as(ngpu.c_int64_p), len(ids),
                                        ctypes.c_float(value), _cstr("weight"))


def set_weights_array_direct(ids, values):
    """Per-synapse write through the distribution 'array' path (needs the
    SetConnectionFloatParamDistr fix on the tinyhippo branch)."""
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    vals = np.ascontiguousarray(values, dtype=np.float32)
    ngpu.SetDistributionIntParam("distr_idx", ngpu.distribution_dict["array"])
    ngpu.SetDistributionIntParam("vect_size", 1)
    ngpu.SetDistributionFloatPtParam("array_pt", vals.ctypes.data_as(ctypes.c_void_p))
    lib.NESTGPU_SetConnectionFloatParamDistr(ids.ctypes.data_as(ngpu.c_int64_p), len(ids),
                                             _cstr("weight"))
    if ngpu.GetErrorCode() != 0:
        raise RuntimeError(ngpu.GetErrorMessage())


def timed(fn, *a):
    t0 = time.perf_counter()
    r = fn(*a)
    return r, time.perf_counter() - t0


def gpu_mem_used_gb():
    return (ngpu.getCUDAMemTotal() - ngpu.getCUDAMemFree()) / 1e9


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="out/phase0/conn_bench.json")
    ap.add_argument("--sim-ms", type=float, default=100.0)
    args = ap.parse_args()
    res = {"nestgpu_commit": os.environ.get("NESTGPU_COMMIT", "?")}

    ngpu.SetTimeResolution(0.1)
    ngpu.SetRandomSeed(42)
    ngpu.SetKernelStatus("verbosity_level", 0)
    ngpu.SetKernelStatus("print_time", False)
    mem0 = gpu_mem_used_gb()

    # ---- build -------------------------------------------------------------
    t0 = time.perf_counter()
    pops = {}
    for name, (n, *_rest) in POPS.items():
        pop = ngpu.Create("izhikevich", int(n))
        if name in INHIBITORY:
            ngpu.SetStatus(pop, {"a": 0.1, "d": 2.0})
        ngpu.SetStatus(pop, "V_m", -65.0)
        ngpu.SetStatus(pop, "u", -13.0)
        pops[name] = pop
    res["create_s"] = time.perf_counter() - t0
    res["n_neurons"] = int(sum(len(p) for p in pops.values()))

    projections = [(s, t, n) for s, t, n, *_ in PROJ] + [(p, p, n) for p, n, *_ in SELF]
    t0 = time.perf_counter()
    n_syn = 0
    for src, tgt, n in projections:
        n_post = len(pops[tgt])
        k = n / n_post
        w = W_INH if src in INHIBITORY else W_EXC
        if k >= 1.0 and abs(round(k) * n_post - n) / n < 0.01:
            rule = {"rule": "fixed_indegree", "indegree": int(round(k))}
            n_syn += int(round(k)) * n_post
        else:
            rule = {"rule": "fixed_total_number", "total_num": int(n)}
            n_syn += int(n)
        ngpu.Connect(pops[src], pops[tgt], rule, {"weight": w, "delay": DELAY})
    res["connect_s"] = time.perf_counter() - t0
    res["n_synapses"] = n_syn

    # Background: one generator per population (each target gets its own train).
    gens = {}
    for name, pop in pops.items():
        g = ngpu.Create("poisson_generator", 1)
        ngpu.SetStatus(g, "rate", 4000.0)
        ngpu.Connect(g, pop, {"rule": "all_to_all"}, {"weight": 1.0, "delay": 1.0})
        gens[name] = g

    for pop in pops.values():
        ngpu.ActivateRecSpikeTimes(pop, 200)
    _, res["calibrate_s"] = timed(ngpu.Calibrate)
    res["gpu_mem_used_gb"] = gpu_mem_used_gb() - mem0
    res["host_mem_peak_gb"] = ngpu.getCUDAMemHostPeak() / 1e9
    print(f"built {res['n_neurons']:,} neurons, {n_syn:,} synapses: "
          f"connect {res['connect_s']:.1f}s, calibrate {res['calibrate_s']:.1f}s, "
          f"GPU memory {res['gpu_mem_used_gb']:.2f} GB")

    # ---- connection queries ------------------------------------------------
    queries = {
        "CA1 PYR->EC LII (STC hook)": ("CA1 PYR", "EC LII"),
        "EC LV->mPFC (assoc hook)": ("EC LV", "mPFC"),
        "CA3 SUP->CA1 PYR (Schaffer)": ("CA3 SUP", "CA1 PYR"),
    }
    res["queries"] = {}
    ids_by_query = {}
    for label, (s, t) in queries.items():
        pairs = len(pops[s]) * len(pops[t])
        ids, t_direct = timed(get_conns_direct, pops[s], pops[t])
        row = {"n": int(len(ids)), "pairs": pairs, "direct_chunked_s": t_direct}
        if pairs <= 100_000_000:   # the unchunked wrapper only where it fits
            conns, row["wrapper_s"] = timed(ngpu.GetConnections, pops[s], pops[t])
            assert len(conns) == len(ids)
        ids_by_query[label] = ids
        res["queries"][label] = row
        print(f"GetConnections {label}: {len(ids):,} synapses, {pairs:.2e} pairs, "
              f"{t_direct:.2f}s direct+chunked"
              + (f", {row['wrapper_s']:.2f}s wrapper" if "wrapper_s" in row else ""))

    # ---- weight get / set ----------------------------------------------------
    stc = ids_by_query["CA1 PYR->EC LII (STC hook)"]
    big = ids_by_query["CA3 SUP->CA1 PYR (Schaffer)"]
    one_m = big[:1_000_000]
    rng = np.random.default_rng(0)
    w_new = rng.uniform(0.0, 0.1, len(stc)).astype(np.float32)
    ops = {}
    _, ops["get 600k direct"] = timed(get_weights_direct, stc)
    _, ops["get 600k wrapper"] = timed(ngpu.GetConnectionFloatParam, stc.tolist(), "weight")
    _, ops["get 1M direct"] = timed(get_weights_direct, one_m)
    _, ops["get all Schaffer direct"] = timed(get_weights_direct, big)
    _, ops["set 1M scalar direct"] = timed(set_weight_scalar_direct, one_m, 0.05)
    try:
        _, ops["set 600k array direct"] = timed(set_weights_array_direct, stc, w_new)
        back = get_weights_direct(stc)
        res["array_write_max_err"] = float(np.max(np.abs(back - w_new)))
    except Exception as e:  # noqa: BLE001 -- report and continue
        res["array_write_error"] = str(e)
    keys = np.round(w_new / 1e-3).astype(np.int64)
    t0 = time.perf_counter()
    order = np.argsort(keys, kind="stable")
    splits = np.flatnonzero(np.diff(keys[order])) + 1
    n_calls = 0
    for grp in np.split(order, splits):
        set_weight_scalar_direct(stc[grp], float(keys[grp[0]] * 1e-3))
        n_calls += 1
    ops[f"set 600k quantised ({n_calls} calls) direct"] = time.perf_counter() - t0
    res["weight_ops_s"] = ops
    for k, v in ops.items():
        print(f"{k:42s} {v:8.3f} s")
    if "array_write_max_err" in res:
        print(f"array write read-back max error: {res['array_write_max_err']:.2e}")
    else:
        print(f"array write failed: {res.get('array_write_error')}")

    # ---- simulate ----------------------------------------------------------
    _, t_one = timed(ngpu.Simulate, args.sim_ms)
    t0 = time.perf_counter()
    theta = [g for name, g in gens.items() if name in ("CA3 SUP", "CA1 PYR", "CA1 OLM")]
    for i in range(int(args.sim_ms)):
        rate = 4000.0 * (1.0 + 0.5 * np.sin(2 * np.pi * 8.0 * i / 1000.0))
        for g in theta:
            ngpu.SetStatus(g, "rate", float(rate))
        ngpu.Simulate(1.0)
    t_chunked = time.perf_counter() - t0
    rates = {}
    for name, pop in pops.items():
        n_spk = sum(len(ts) for ts in ngpu.GetRecSpikeTimes(pop))
        rates[name] = n_spk / len(pop) / (2 * args.sim_ms / 1000.0)
    res["simulate"] = {"sim_ms": args.sim_ms, "one_call_s": t_one,
                       "chunked_1ms_s": t_chunked, "mean_rate_hz": rates}
    print(f"Simulate {args.sim_ms:.0f} ms: {t_one:.2f}s in one call, "
          f"{t_chunked:.2f}s in 1 ms chunks with theta rate updates")
    print("mean rates (Hz): " + ", ".join(f"{k} {v:.1f}" for k, v in rates.items()))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
