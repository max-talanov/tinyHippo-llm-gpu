"""Theta and ripple drive: CPU NEST sinusoidal_poisson_generator vs NEST GPU rate slicing.

NEST GPU has no sinusoidal_poisson_generator (joint plan §3, gap 1). The
replacement is a poisson_generator whose rate is rewritten between 1 ms
Simulate() chunks (1 ms is the shortest chunk that keeps Poisson output exact;
see compat_recheck.py). This script measures how well that reproduces the
model's two oscillatory drives (replay_scaled.py defaults):

  theta : 8 Hz,   mean 1100 Hz, amplitude 1000 Hz, all run
  ripple: 180 Hz, mean 1100 Hz, amplitude  850 Hz, 300-420 ms window

Each drive feeds 500 parrot neurons; the recorded trains give the mean rate and
the first Fourier component (amplitude and phase) over the drive window. On the
GPU the rate is sampled either at the start of each chunk (what the July port
did) or at its midpoint.

    python3 gpu_spike/theta_ripple_check.py --backend nest    --out out/phase0/drive_nest.json
    python3 gpu_spike/theta_ripple_check.py --backend nestgpu --out out/phase0/drive_gpu.json
    python3 gpu_spike/theta_ripple_check.py --compare out/phase0/drive_nest.json out/phase0/drive_gpu.json
"""
import argparse
import json
import math
import os

import numpy as np

N = 500
DRIVES = {
    "theta":  dict(freq=8.0,   mean=1100.0, amp=1000.0, start=0.0,   stop=2000.0, sim=2000.0),
    "ripple": dict(freq=180.0, mean=1100.0, amp=850.0,  start=300.0, stop=420.0,  sim=600.0),
}
CHUNK_MS = 1.0
DELAY = 1.0


def fourier(times_ms, d):
    """Mean rate and first harmonic (amplitude, phase in degrees) per train."""
    t = np.asarray(times_ms) - DELAY
    t = t[(t >= d["start"]) & (t < d["stop"])]
    span_s = (d["stop"] - d["start"]) / 1000.0
    w = 2.0 * math.pi * d["freq"] / 1000.0
    c = np.sum(np.exp(-1j * w * t)) / (N * span_s)
    return {"rate_hz": len(t) / (N * span_s), "amp_hz": float(2 * abs(c)),
            "phase_deg": float(np.degrees(np.angle(c) + math.pi / 2) % 360.0)}


def run_nest(name, d):
    import nest
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.1, "rng_seed": 42})
    nest.set_verbosity("M_ERROR")
    gen = nest.Create("sinusoidal_poisson_generator", N, params={
        "rate": d["mean"], "amplitude": d["amp"], "frequency": d["freq"], "phase": 0.0,
        "start": d["start"], "stop": d["stop"]})
    parrots = nest.Create("parrot_neuron", N)
    nest.Connect(gen, parrots, "one_to_one", syn_spec={"delay": DELAY})
    sr = nest.Create("spike_recorder")
    nest.Connect(parrots, sr)
    nest.Simulate(d["sim"])
    return np.asarray(sr.get("events")["times"])


def run_nestgpu(name, d, sample):
    import nestgpu as ngpu
    ngpu.SetTimeResolution(0.1)
    ngpu.SetRandomSeed(42)
    ngpu.SetKernelStatus("verbosity_level", 0)
    ngpu.SetKernelStatus("print_time", False)
    gen = ngpu.Create("poisson_generator", 1)
    ngpu.SetStatus(gen, "rate", 0.0)
    parrots = ngpu.Create("parrot_neuron", N)
    ngpu.Connect(gen, parrots, {"rule": "all_to_all"}, {"weight": 1.0, "delay": DELAY})
    # A NEST GPU parrot emits one spike per step however many arrive (unless
    # hold_spike_mul is set), so spike times undercount at these rates. Its
    # V variable holds the input multiplicity of each step: record that.
    rec = ngpu.CreateRecord("", ["V"] * N, [parrots[i] for i in range(N)], [0] * N)
    w = 2.0 * math.pi * d["freq"] / 1000.0
    offset = 0.5 * CHUNK_MS if sample == "mid" else 0.0
    for i in range(int(round(d["sim"] / CHUNK_MS))):
        t = i * CHUNK_MS + offset
        on = d["start"] <= t < d["stop"]
        ngpu.SetStatus(gen, "rate", max(0.0, d["mean"] + d["amp"] * math.sin(w * t)) if on else 0.0)
        ngpu.Simulate(CHUNK_MS)
    data = np.asarray(ngpu.GetRecordData(rec))
    t, mult = data[:, 0], np.rint(data[:, 1:]).astype(int)
    return np.repeat(np.repeat(t[:, None], N, axis=1).ravel(), mult.ravel())


def compare(path_a, path_b):
    a, b = (json.load(open(p)) for p in (path_a, path_b))
    print(f"{'drive':7s} {'variant':22s} {'rate Hz':>8s} {'amp Hz':>8s} {'phase':>7s}")
    for name, d in DRIVES.items():
        print(f"{name:7s} {'target':22s} {d['mean']:8.1f} {d['amp']:8.1f} {0.0:7.1f}")
        for res in (a, b):
            for variant, r in res[name].items():
                print(f"{'':7s} {res['backend'] + ' ' + variant:22s} "
                      f"{r['rate_hz']:8.1f} {r['amp_hz']:8.1f} {r['phase_deg']:7.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=("nest", "nestgpu"))
    ap.add_argument("--drive", choices=sorted(DRIVES))
    ap.add_argument("--sample", choices=("start", "mid"))
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    if args.drive:   # one drive, one variant, in this process
        d = DRIVES[args.drive]
        t = run_nest(args.drive, d) if args.backend == "nest" else run_nestgpu(args.drive, d, args.sample)
        print("RESULT " + json.dumps(fourier(t, d)))
        return
    # NEST GPU has no kernel reset, so every (drive, variant) runs in its own process.
    import subprocess
    import sys
    res = {"backend": args.backend}
    variants = ["sinusoidal"] if args.backend == "nest" else ["start", "mid"]
    for name in DRIVES:
        res[name] = {}
        for v in variants:
            cmd = [sys.executable, __file__, "--backend", args.backend, "--drive", name]
            if args.backend == "nestgpu":
                cmd += ["--sample", v]
            out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
            line = [l for l in out.splitlines() if l.startswith("RESULT ")][-1]
            res[name][("rate-sliced, " + v) if args.backend == "nestgpu" else v] = json.loads(line[7:])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
