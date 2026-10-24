"""Check izhikevich_het, the per-neuron-parameter Izhikevich model.

izhikevich_het (nest-gpu-tinyhippo branch `tinyhippo`) moves V_th, a, b, c, d
from group to per-neuron parameters. Three checks:

1. Which parameters accept per-neuron arrays, on izhikevich and izhikevich_het.
2. Equivalence: with the same parameters on every neuron, izhikevich_het gives
   the same spike times as izhikevich (DC drive plus a shared input burst).
3. Per-neuron b acts per neuron: the DC rheobase of each cell follows its own b,
   I_rheo = (5 - b)^2 / 0.16 - 140 - I_e, and matches a stock izhikevich
   population whose group b has the same value.

    python3 gpu_spike/izh_het_check.py
"""
import numpy as np
import nestgpu as ngpu

ngpu.SetTimeResolution(0.1)
ngpu.SetRandomSeed(42)
ngpu.SetKernelStatus("verbosity_level", 0)
ngpu.SetKernelStatus("print_time", False)

RS = dict(a=0.02, b=0.2, c=-65.0, d=8.0)        # CA3 SUP / CA1 PYR / GC
B_VALUES = [0.10, 0.15, 0.20, 0.24]              # all below b_crit = 0.267 at I_e=0
I_GRID = np.round(np.arange(0.0, 12.0, 0.05), 3)  # pA
SIM_MS = 2000.0


def kind(pop, name):
    if ngpu.IsNeuronScalParam(pop.i0, name):
        return "per-neuron"
    if ngpu.IsNeuronGroupParam(pop.i0, name):
        return "group"
    return "?"


def set_array(pop, name, values):
    ngpu.SetStatus(pop, name, {"array": [float(v) for v in values]})


# ---- 1. which parameters take arrays --------------------------------------
probe_stock = ngpu.Create("izhikevich", 4)
probe_het = ngpu.Create("izhikevich_het", 4)
print("1. per-neuron arrays")
for name in ("V_th", "a", "b", "c", "d", "I_e", "t_ref"):
    row = []
    for label, pop in (("izhikevich", probe_stock), ("izhikevich_het", probe_het)):
        try:
            set_array(pop, name, [1.0, 2.0, 3.0, 4.0] if name == "t_ref"
                      else np.linspace(0.1, 0.2, 4))
            ok = "accepted"
        except Exception:  # noqa: BLE001 -- binding raises bare ValueError
            ok = "rejected"
        row.append(f"{label}: {kind(pop, name):10s} {ok}")
    print(f"   {name:5s} " + " | ".join(row))

# ---- 2. equivalence with identical parameters -----------------------------
N_EQ = 200
i_e = np.linspace(0.0, 15.0, N_EQ)
eq = {}
for model in ("izhikevich", "izhikevich_het"):
    pop = ngpu.Create(model, N_EQ)
    ngpu.SetStatus(pop, {k: float(v) for k, v in RS.items()})
    set_array(pop, "I_e", i_e)
    ngpu.SetStatus(pop, "V_m", -65.0)
    ngpu.SetStatus(pop, "u", -13.0)
    eq[model] = pop
burst = ngpu.Create("spike_generator", 1)
ngpu.SetStatus(burst, {"spike_times": [200.0, 205.0, 600.0]})
for pop in eq.values():
    ngpu.Connect(burst, pop, {"rule": "all_to_all"}, {"weight": 8.0, "delay": 1.0})

# ---- 3. per-neuron b vs rheobase ------------------------------------------
n_i = len(I_GRID)
het_b = ngpu.Create("izhikevich_het", len(B_VALUES) * n_i)
ngpu.SetStatus(het_b, {k: float(v) for k, v in RS.items() if k != "b"})
b_per_cell = np.repeat(B_VALUES, n_i)
i_per_cell = np.tile(I_GRID, len(B_VALUES))
set_array(het_b, "b", b_per_cell)
set_array(het_b, "I_e", i_per_cell)
set_array(het_b, "V_m", np.full(len(b_per_cell), -65.0))
set_array(het_b, "u", b_per_cell * -65.0)
stock_b = {}
for b in B_VALUES:
    pop = ngpu.Create("izhikevich", n_i)
    ngpu.SetStatus(pop, {**{k: float(v) for k, v in RS.items()}, "b": float(b)})
    set_array(pop, "I_e", I_GRID)
    ngpu.SetStatus(pop, "V_m", -65.0)
    ngpu.SetStatus(pop, "u", float(b) * -65.0)
    stock_b[b] = pop

recorded = [eq["izhikevich"], eq["izhikevich_het"], het_b, *stock_b.values()]
for pop in recorded:
    ngpu.ActivateRecSpikeTimes(pop, 200)
ngpu.Simulate(SIM_MS)
spikes = {pop.i0: ngpu.GetRecSpikeTimes(pop) for pop in recorded}

st, ht = spikes[eq["izhikevich"].i0], spikes[eq["izhikevich_het"].i0]
n_same = sum(1 for a, b in zip(st, ht) if list(a) == list(b))
n_spk = sum(len(a) for a in st)
print(f"\n2. equivalence: {n_same}/{N_EQ} neurons have identical spike trains "
      f"({n_spk} spikes in izhikevich, {sum(len(b) for b in ht)} in izhikevich_het)")


def rheobase(trains):
    fired = [i for i, ts in enumerate(trains) if len(ts) > 0]
    return float(I_GRID[fired[0]]) if fired else float("nan")


print("\n3. DC rheobase per b (pA; grid step 0.05)")
print("   b      analytic  izhikevich_het  izhikevich(group b)")
het_trains = spikes[het_b.i0]
for k, b in enumerate(B_VALUES):
    analytic = (5.0 - b) ** 2 / 0.16 - 140.0
    r_het = rheobase(het_trains[k * n_i:(k + 1) * n_i])
    r_stock = rheobase(spikes[stock_b[b].i0])
    print(f"   {b:.2f}   {analytic:8.3f}  {r_het:14.2f}  {r_stock:19.2f}")
