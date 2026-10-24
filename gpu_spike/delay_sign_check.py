"""Inhibitory sign and per-synapse delay arrays: NEST GPU against CPU NEST.

Phase 0 item 4 (nest_gpu_migration.md §3). One input spike at 10 ms reaches
eight quiet izhikevich neurons through one synapse each, with weights of
alternating sign and a different delay per synapse (given as arrays). For each
target, find the step where V_m jumps. izhikevich adds the weight directly to
V_m, so the jump size and sign should equal the weight, and its time should be
10 ms + delay (on the 0.1 ms grid) plus a fixed recording offset. The offset is
reported per backend, so the two can be compared.

    python3 gpu_spike/delay_sign_check.py --backend nestgpu
    python3 gpu_spike/delay_sign_check.py --backend nest
"""
import argparse
import ctypes

import numpy as np

T_SPIKE = 10.0
WEIGHTS = [2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0]
DELAYS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.33, 7.77]
N = len(WEIGHTS)
SIM_MS = 30.0


def run_nestgpu():
    import nestgpu as ngpu
    ngpu.SetTimeResolution(0.1)
    ngpu.SetRandomSeed(42)
    ngpu.SetKernelStatus("verbosity_level", 0)
    ngpu.SetKernelStatus("print_time", False)
    sg = ngpu.Create("spike_generator", 1)
    ngpu.SetStatus(sg, {"spike_times": [T_SPIKE]})
    targets = ngpu.Create("izhikevich", N)
    ngpu.SetStatus(targets, "V_m", -70.0)
    ngpu.SetStatus(targets, "u", -14.0)
    w_arr = (ctypes.c_float * N)(*WEIGHTS)   # kept alive across Connect
    d_arr = (ctypes.c_float * N)(*DELAYS)
    ngpu.Connect(sg, targets, {"rule": "all_to_all"},
                 {"weight": {"array": w_arr}, "delay": {"array": d_arr}})
    src = ngpu.Create("izhikevich", 1)
    ngpu.Connect(sg, src, {"rule": "all_to_all"}, {"weight": 100.0, "delay": 1.0})
    targets2 = ngpu.Create("izhikevich", N)
    ngpu.SetStatus(targets2, "V_m", -70.0)
    ngpu.SetStatus(targets2, "u", -14.0)
    ngpu.Connect(src, targets2, {"rule": "all_to_all"},
                 {"weight": {"array": w_arr}, "delay": {"array": d_arr}})
    targets3 = ngpu.Create("izhikevich", N)
    ngpu.Connect(src, targets3, {"rule": "all_to_all"},
                 {"weight": 100.0, "delay": {"array": d_arr}})
    ngpu.ActivateRecSpikeTimes(src, 10)
    ngpu.ActivateRecSpikeTimes(targets3, 10)
    all_t = [targets[i] for i in range(N)] + [targets2[i] for i in range(N)]
    rec = ngpu.CreateRecord("", ["V_m"] * (2 * N), all_t, [0] * (2 * N))
    ngpu.Simulate(SIM_MS)
    data = np.asarray(ngpu.GetRecordData(rec))
    t_src = float(ngpu.GetRecSpikeTimes(src)[0][0])
    t3 = [float(ts[0]) for ts in ngpu.GetRecSpikeTimes(targets3)]
    conns = ngpu.GetConnections(sg, targets)
    return (data[:, 0], data[:, 1:N + 1], data[:, N + 1:], t_src, t3,
            [float(x) for x in ngpu.GetStatus(conns, "weight")],
            [float(x) for x in ngpu.GetStatus(conns, "delay")])


def run_nest():
    import nest
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.1, "rng_seed": 42})
    nest.set_verbosity("M_ERROR")
    sg = nest.Create("spike_generator", params={"spike_times": [T_SPIKE]})
    targets = nest.Create("izhikevich", N, params={"V_m": -70.0, "U_m": -14.0})
    for i in range(N):
        nest.Connect(sg, targets[i], syn_spec={"weight": WEIGHTS[i], "delay": DELAYS[i]})
    src = nest.Create("izhikevich")
    nest.Connect(sg, src, syn_spec={"weight": 100.0, "delay": 1.0})
    targets2 = nest.Create("izhikevich", N, params={"V_m": -70.0, "U_m": -14.0})
    for i in range(N):
        nest.Connect(src, targets2[i], syn_spec={"weight": WEIGHTS[i], "delay": DELAYS[i]})
    targets3 = nest.Create("izhikevich", N)
    for i in range(N):
        nest.Connect(src, targets3[i], syn_spec={"weight": 100.0, "delay": DELAYS[i]})
    sr = nest.Create("spike_recorder")
    nest.Connect(src, sr)
    sr3 = nest.Create("spike_recorder")
    nest.Connect(targets3, sr3)
    mm = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": 0.1})
    nest.Connect(mm, targets + targets2)
    nest.Simulate(SIM_MS)
    ev = mm.get("events")
    times = np.unique(ev["times"])

    def trace(pop):
        first = pop.tolist()[0]
        v = np.zeros((len(times), N))
        for i in range(N):
            v[:, i] = np.asarray(ev["V_m"])[ev["senders"] == first + i]
        return v

    t_src = float(sr.get("events")["times"][0])
    ev3 = sr3.get("events")
    first3 = targets3.tolist()[0]
    t3 = [float(min(ev3["times"][ev3["senders"] == first3 + i])) for i in range(N)]
    conns = nest.GetConnections(sg, targets)
    return (times, trace(targets), trace(targets2), t_src, t3,
            [float(w) for w in conns.get("weight")],
            [float(d) for d in conns.get("delay")])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=("nest", "nestgpu"), default="nestgpu")
    args = ap.parse_args()
    t, v, v2, t_src, t3, stored_w, stored_d = (run_nest() if args.backend == "nest"
                                           else run_nestgpu())

    print(f"backend {args.backend}")
    print("hop 1: spike_generator -> izhikevich")
    print("target  weight  delay  stored(w, d)    jump_at_ms  spike+delay  jump_mV")
    offsets, ok = [], True
    for i in range(N):
        dv = np.diff(v[:, i])
        k = int(np.argmax(np.abs(dv)))
        t_jump = float(t[k + 1])
        nominal = T_SPIKE + round(DELAYS[i], 1)
        offsets.append(round(t_jump - nominal, 3))
        good = (np.sign(dv[k]) == np.sign(WEIGHTS[i])
                and abs(abs(dv[k]) - abs(WEIGHTS[i])) < 0.3
                and abs(stored_d[i] - round(DELAYS[i], 1)) < 1e-4)
        ok &= good
        print(f"{i:6d}  {WEIGHTS[i]:6.1f}  {DELAYS[i]:5.2f}  ({stored_w[i]:5.1f}, {stored_d[i]:5.2f})"
              f"   {t_jump:10.1f}  {nominal:11.1f}  {dv[k]:7.2f}")
    constant = len(set(offsets)) == 1
    print(f"sign and size {'ok' if ok else 'MISMATCH'}; "
          f"jump offset after spike+delay: {sorted(set(offsets))} ms "
          f"({'constant' if constant else 'NOT constant'})")

    # hop 2: neuron -> neuron, timed from the recorded source spike
    offsets2 = []
    for i in range(N):
        dv = np.diff(v2[:, i])
        k = int(np.argmax(np.abs(dv)))
        offsets2.append(round(float(t[k + 1]) - (t_src + round(DELAYS[i], 1)), 3))
    print(f"hop 2: izhikevich -> izhikevich; source spike recorded at {t_src:.1f} ms; "
          f"jump offset after spike+delay: {sorted(set(offsets2))} ms")

    # hop 3: neuron -> neuron with a suprathreshold weight, timed spike to spike
    lat = sorted(set(round(t3[i] - t_src - round(DELAYS[i], 1), 3) for i in range(N)))
    print(f"hop 3: spike-to-spike latency minus delay: {lat} ms")


if __name__ == "__main__":
    main()
