"""Re-check NEST GPU behaviours that the July 2026 port worked around.

The port (tfatykhov/tinyHippo@feat/gpu-port, gpu/nestgpu_compat.py) measured
them on nest/nest-gpu 90f87ab. This repo builds a later NEST GPU, so each one is
measured again. Every case runs in its own process: NEST GPU has no
ResetKernel, and some cases are expected to crash.

    python3 gpu_spike/compat_recheck.py            # run every case
    python3 gpu_spike/compat_recheck.py --case X   # run one case in-process
"""
import argparse
import json
import os
import subprocess
import sys

RESULT_TAG = "RESULT "


def _init(seed=42):
    import nestgpu as ngpu
    ngpu.SetTimeResolution(0.1)
    ngpu.SetRandomSeed(seed)
    ngpu.SetKernelStatus("verbosity_level", 0)
    ngpu.SetKernelStatus("print_time", False)
    return ngpu


def _report(**kw):
    print(RESULT_TAG + json.dumps(kw), flush=True)


def case_chunk_floor(chunk_ms):
    """Poisson output over 1000 ms must not depend on the Simulate() chunk."""
    ngpu = _init()
    gen = ngpu.Create("poisson_generator", 1)
    ngpu.SetStatus(gen, "rate", 800.0)
    parrot = ngpu.Create("parrot_neuron", 1)
    ngpu.Connect(gen, parrot, {"rule": "one_to_one"}, {"weight": 1.0, "delay": 1.0})
    ngpu.ActivateRecSpikeTimes(parrot, 5000)
    n = int(round(1000.0 / chunk_ms))
    for _ in range(n):
        ngpu.Simulate(chunk_ms)
    _report(chunk_ms=chunk_ms, steps=int(round(chunk_ms / 0.1)),
            spikes=len(ngpu.GetRecSpikeTimes(parrot)[0]))


def case_poisson_per_target():
    """One poisson_generator onto 50 parrots: independent trains, or one shared train?"""
    ngpu = _init()
    gen = ngpu.Create("poisson_generator", 1)
    ngpu.SetStatus(gen, "rate", 100.0)
    parrots = ngpu.Create("parrot_neuron", 50)
    ngpu.Connect(gen, parrots, {"rule": "all_to_all"}, {"weight": 1.0, "delay": 1.0})
    ngpu.ActivateRecSpikeTimes(parrots, 1000)
    ngpu.Simulate(1000.0)
    trains = [tuple(round(t, 1) for t in ts) for ts in ngpu.GetRecSpikeTimes(parrots)]
    counts = [len(t) for t in trains]
    _report(distinct_trains=len(set(trains)), of=50,
            mean_rate_hz=sum(counts) / 50.0, min_count=min(counts), max_count=max(counts))


def case_spike_generator(chunked):
    """A spike_generator burst at 300 ms onto 264 parrots, one call vs 1 ms chunks."""
    ngpu = _init()
    sg = ngpu.Create("spike_generator", 1)
    ngpu.SetStatus(sg, {"spike_times": [300.0]})
    parrots = ngpu.Create("parrot_neuron", 264)
    ngpu.Connect(sg, parrots, {"rule": "all_to_all"}, {"weight": 1.0, "delay": 1.0})
    ngpu.ActivateRecSpikeTimes(parrots, 10)
    if chunked:
        for _ in range(500):
            ngpu.Simulate(1.0)
    else:
        ngpu.Simulate(500.0)
    fired = sum(1 for ts in ngpu.GetRecSpikeTimes(parrots) if len(ts) > 0)
    _report(chunked=bool(chunked), fired=fired, of=264)


def case_start_stop():
    """A generator windowed to 60-80 ms must be silent after the window."""
    ngpu = _init()
    gen = ngpu.Create("poisson_generator", 1)
    ngpu.SetStatus(gen, {"rate": 2000.0, "start": 60.0, "stop": 80.0})
    parrot = ngpu.Create("parrot_neuron", 1)
    ngpu.Connect(gen, parrot, {"rule": "one_to_one"}, {"weight": 1.0, "delay": 1.0})
    ngpu.ActivateRecSpikeTimes(parrot, 5000)
    ngpu.Simulate(500.0)
    ts = ngpu.GetRecSpikeTimes(parrot)[0]
    _report(before=sum(t < 60.0 for t in ts),
            inside=sum(60.0 <= t <= 81.0 for t in ts),
            after=sum(t > 81.0 for t in ts), expected_inside=40)


def case_rec_drain():
    """GetRecSpikeTimes: does a second read return the same spikes?"""
    ngpu = _init()
    pop = ngpu.Create("izhikevich", 5)
    ngpu.SetStatus(pop, "I_e", 10.0)
    ngpu.ActivateRecSpikeTimes(pop, 1000)
    ngpu.Simulate(500.0)
    r1 = sum(len(ts) for ts in ngpu.GetRecSpikeTimes(pop))
    r2 = sum(len(ts) for ts in ngpu.GetRecSpikeTimes(pop))
    ngpu.Simulate(500.0)
    r3 = sum(len(ts) for ts in ngpu.GetRecSpikeTimes(pop))
    _report(read1=r1, read2_immediate=r2, read3_after_500ms=r3,
            drains=(r2 == 0))


def case_weight_array():
    """Per-synapse weight write from an array, then read back."""
    ngpu = _init()
    a = ngpu.Create("izhikevich", 10)
    b = ngpu.Create("izhikevich", 10)
    ngpu.Connect(a, b, {"rule": "all_to_all"}, {"weight": 1.0, "delay": 1.0})
    ngpu.Calibrate()
    conns = ngpu.GetConnections(a, b)
    want = [0.01 * i for i in range(len(conns))]
    ngpu.SetConnectionStatus(conns, "weight", {"array": want})
    got = [float(w) for w in ngpu.GetStatus(conns, "weight")]
    err = max(abs(x - y) for x, y in zip(want, got))
    _report(n=len(conns), max_abs_err=err, ok=err < 1e-5)


def case_getconn_before_calibrate():
    """GetConnections before Calibrate()."""
    ngpu = _init()
    a = ngpu.Create("izhikevich", 10)
    b = ngpu.Create("izhikevich", 10)
    ngpu.Connect(a, b, {"rule": "all_to_all"}, {"weight": 1.0, "delay": 1.0})
    conns = ngpu.GetConnections(a, b)
    _report(n=len(conns), expected=100)


def case_unknown_rule():
    """An unsupported rule name ("pairwise_bernoulli"): rejected or remapped?"""
    ngpu = _init()
    a = ngpu.Create("izhikevich", 20)
    b = ngpu.Create("izhikevich", 20)
    try:
        ngpu.Connect(a, b, {"rule": "pairwise_bernoulli"}, {"weight": 1.0, "delay": 1.0})
    except Exception as e:  # noqa: BLE001 -- the binding raises bare ValueError
        _report(rejected=True, error=str(e))
        return
    ngpu.Calibrate()
    _report(rejected=False, n_conn=len(ngpu.GetConnections(a, b)))


def case_sinusoidal():
    """Is sinusoidal_poisson_generator available?"""
    ngpu = _init()
    try:
        ngpu.Create("sinusoidal_poisson_generator", 1)
    except Exception as e:  # noqa: BLE001
        _report(available=False, error=str(e))
        return
    _report(available=True)


def case_create_after_simulate():
    """Create and Connect after the first Simulate(): legal, and do new cells run?"""
    ngpu = _init()
    pre = ngpu.Create("izhikevich", 10)
    ngpu.SetStatus(pre, "I_e", 10.0)
    ngpu.Simulate(100.0)
    try:
        post = ngpu.Create("izhikevich", 10)
        ngpu.Connect(pre, post, {"rule": "all_to_all"}, {"weight": 30.0, "delay": 1.0})
        ngpu.ActivateRecSpikeTimes(post, 1000)
        ngpu.Simulate(100.0)
    except Exception as e:  # noqa: BLE001
        _report(allowed=False, error=str(e))
        return
    fired = sum(1 for ts in ngpu.GetRecSpikeTimes(post) if len(ts) > 0)
    _report(allowed=True, new_cells_fired=fired, of=10)


CASES = {
    "chunk_floor_0.5": (case_chunk_floor, 0.5),
    "chunk_floor_0.9": (case_chunk_floor, 0.9),
    "chunk_floor_1.0": (case_chunk_floor, 1.0),
    "chunk_floor_1000": (case_chunk_floor, 1000.0),
    "poisson_per_target": (case_poisson_per_target,),
    "spike_generator_single": (case_spike_generator, False),
    "spike_generator_chunked": (case_spike_generator, True),
    "start_stop": (case_start_stop,),
    "rec_drain": (case_rec_drain,),
    "weight_array": (case_weight_array,),
    "getconn_before_calibrate": (case_getconn_before_calibrate,),
    "unknown_rule": (case_unknown_rule,),
    "sinusoidal": (case_sinusoidal,),
    "create_after_simulate": (case_create_after_simulate,),
}


def run_all():
    commit = os.environ.get("NESTGPU_COMMIT", "unknown")
    print(f"NEST GPU commit {commit}")
    for name in CASES:
        proc = subprocess.run([sys.executable, __file__, "--case", name],
                              capture_output=True, text=True, timeout=600)
        lines = [l[len(RESULT_TAG):] for l in proc.stdout.splitlines()
                 if l.startswith(RESULT_TAG)]
        if lines:
            print(f"{name:26s} {lines[-1]}")
        else:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or [""]
            print(f"{name:26s} CRASH rc={proc.returncode} {tail[0][:160]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", choices=sorted(CASES))
    args = ap.parse_args()
    if args.case is None:
        run_all()
        return
    fn, *fargs = CASES[args.case]
    fn(*fargs)


if __name__ == "__main__":
    main()
