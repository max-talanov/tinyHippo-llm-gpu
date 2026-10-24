"""DC f-I curves of tinyHippo's Izhikevich parameter sets on CPU NEST and NEST GPU.

Phase 0 item 2 (nest_gpu_migration.md §3): the curves must agree within ~5%
before the GPU backend can be trusted, since the model's operating points were
calibrated on CPU NEST. Rheobase is measured with injected DC current, as
CLAUDE.md requires: I_rheo = (5-b)^2/0.16 - 140 - I_e.

    python3 gpu_spike/fi_curves.py --backend nest    --out out/fi_nest.json
    python3 gpu_spike/fi_curves.py --backend nestgpu --out out/fi_gpu.json [--model izhikevich_het]
    python3 gpu_spike/fi_curves.py --compare out/fi_nest.json out/fi_gpu.json
"""
import argparse
import json
import os

import numpy as np

# Calibrated parameter sets, as built in replay_scaled.py and
# nest_dg_ca3_fi_calibration.py. U_m is the initial recovery variable.
PARAM_SETS = {
    # CA3 SUP, CA1 PYR, DG GC, MC_LOW, mPFC
    "rs_d8":      dict(a=0.02, b=0.20, c=-65.0, d=8.0, V_m=-65.0, U_m=-13.0,  I_e=0.0),
    # EC LII
    "ec_lii":     dict(a=0.02, b=0.20, c=-65.0, d=6.0, V_m=-65.0, U_m=-13.0,  I_e=0.0),
    # CA3 DEEP, EC LV
    "deep_lv":    dict(a=0.02, b=0.20, c=-55.0, d=4.0, V_m=-64.7, U_m=-13.0,  I_e=3.0),
    # CA1/CA3/DG baskets, mPFC INT
    "basket":     dict(a=0.10, b=0.20, c=-65.0, d=2.0, V_m=-65.0, U_m=-13.0,  I_e=0.0),
    # CA1 OLM
    "olm":        dict(a=0.02, b=0.25, c=-65.0, d=2.0, V_m=-65.0, U_m=-16.25, I_e=0.0),
    # MC_HIGH
    "mc_high":    dict(a=0.02, b=0.20, c=-65.0, d=8.0, V_m=-65.0, U_m=-13.0,  I_e=-15.1),
}
I_DC = np.round(np.arange(0.0, 40.0001, 0.25), 3)   # pA added to each set's I_e
SIM_MS = 1200.0
WARMUP_MS = 200.0
DT = 0.1


def run_nest():
    import nest
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": DT, "local_num_threads": 1,
                          "rng_seed": 42})
    nest.set_verbosity("M_ERROR")
    pops, recs = {}, {}
    for name, p in PARAM_SETS.items():
        params = {k: v for k, v in p.items() if k != "I_e"}
        params["I_e"] = (p["I_e"] + I_DC).tolist()
        pops[name] = nest.Create("izhikevich", len(I_DC), params=params)
        recs[name] = nest.Create("spike_recorder")
        nest.Connect(pops[name], recs[name])
    nest.Simulate(SIM_MS)
    out = {}
    for name, pop in pops.items():
        ev = recs[name].get("events")
        first = pop.tolist()[0]
        idx = np.asarray(ev["senders"]) - first
        t = np.asarray(ev["times"])
        out[name] = [np.asarray(t[idx == i]) for i in range(len(I_DC))]
    return f"nest {nest.__version__}", "izhikevich", out


def run_nestgpu(model):
    import nestgpu as ngpu
    ngpu.SetTimeResolution(DT)
    ngpu.SetRandomSeed(42)
    ngpu.SetKernelStatus("verbosity_level", 0)
    ngpu.SetKernelStatus("print_time", False)
    pops = {}
    for name, p in PARAM_SETS.items():
        pop = ngpu.Create(model, len(I_DC))
        ngpu.SetStatus(pop, {k: float(p[k]) for k in ("a", "b", "c", "d")})
        ngpu.SetStatus(pop, "I_e", {"array": (p["I_e"] + I_DC).tolist()})
        ngpu.SetStatus(pop, "V_m", float(p["V_m"]))
        ngpu.SetStatus(pop, "u", float(p["U_m"]))
        ngpu.ActivateRecSpikeTimes(pop, 2000)
        pops[name] = pop
    ngpu.Simulate(SIM_MS)
    out = {name: [np.asarray(ts) for ts in ngpu.GetRecSpikeTimes(pop)]
           for name, pop in pops.items()}
    return f"nestgpu {os.environ.get('NESTGPU_COMMIT', '?')}", model, out


def summarise(trains):
    window_s = (SIM_MS - WARMUP_MS) / 1000.0
    rate = [float(np.sum(ts >= WARMUP_MS)) / window_s for ts in trains]
    fired = [i for i, ts in enumerate(trains) if len(ts) > 0]
    rheo = float(I_DC[fired[0]]) if fired else None
    return {"rate_hz": rate, "rheobase_dc_pA": rheo}


def compare(path_a, path_b):
    a, b = (json.load(open(p)) for p in (path_a, path_b))
    print(f"A = {a['backend']} / {a['model']}\nB = {b['backend']} / {b['model']}")
    print(f"{'set':9s} {'rheo A':>7s} {'rheo B':>7s} {'analytic':>8s} "
          f"{'max|dHz|':>8s} {'max rel':>8s} {'mean rel':>8s}  (rel over A >= 5 Hz)")
    for name, p in PARAM_SETS.items():
        ra = np.asarray(a["sets"][name]["rate_hz"])
        rb = np.asarray(b["sets"][name]["rate_hz"])
        analytic = (5.0 - p["b"]) ** 2 / 0.16 - 140.0 - p["I_e"]
        mask = ra >= 5.0
        rel = np.abs(rb[mask] - ra[mask]) / ra[mask] if mask.any() else np.array([np.nan])
        print(f"{name:9s} {a['sets'][name]['rheobase_dc_pA']!s:>7s} "
              f"{b['sets'][name]['rheobase_dc_pA']!s:>7s} {analytic:8.2f} "
              f"{np.max(np.abs(rb - ra)):8.2f} {np.max(rel):8.3f} {np.mean(rel):8.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=("nest", "nestgpu"))
    ap.add_argument("--model", default="izhikevich")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    backend, model, trains = (run_nest() if args.backend == "nest"
                              else run_nestgpu(args.model))
    res = {"backend": backend, "model": model, "dt_ms": DT, "sim_ms": SIM_MS,
           "warmup_ms": WARMUP_MS, "i_dc_pA": I_DC.tolist(),
           "sets": {name: summarise(tr) for name, tr in trains.items()}}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f)
    print(f"wrote {args.out}: {backend} / {model}")


if __name__ == "__main__":
    main()
