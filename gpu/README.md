# NEST GPU image

The GPU side of this repo runs in a Docker image built from a project copy of NEST GPU.

## Simulator source

A separate checkout, not vendored here (NEST GPU is GPL-2.0):

| | |
|---|---|
| Repository | `tfatykhov/nest-gpu`, branch `tinyhippo` |
| Base | `nest/nest-gpu` `main` at `ba52eb9` (2026-10-20) |
| Local path assumed below | `../nest-gpu-tinyhippo` (sibling of this repo) |

Project commits on `tinyhippo`, on top of upstream:

| Commit | Change | Why |
|---|---|---|
| `509657d` | New model `izhikevich_het`: `V_th`, `a`, `b`, `c`, `d` per neuron | Stock `izhikevich` keeps them per group, but `--het` draws them per cell (RESULTS_llm_gpu.md §1) |
| `b0c1f1a` | Fix a null-pointer segfault in `SetConnectionFloatParamDistr` | Per-synapse weight writes from an array crashed; plasticity hooks need them |

```sh
git clone https://github.com/tfatykhov/nest-gpu.git ../nest-gpu-tinyhippo
git -C ../nest-gpu-tinyhippo switch tinyhippo
git -C ../nest-gpu-tinyhippo remote add upstream https://github.com/nest/nest-gpu.git
```

On Windows, clone with `core.autocrlf=false`: the source is copied into a Linux build.

## Build

From the repository root:

```sh
docker build gpu \
    --build-context nestgpu=../nest-gpu-tinyhippo \
    --build-arg NESTGPU_COMMIT=$(git -C ../nest-gpu-tinyhippo rev-parse --short HEAD) \
    -t tinyhippo-nestgpu:latest
```

`GPU_ARCH` defaults to 89 (Ada, RTX 40-series); pass `--build-arg GPU_ARCH=90` for H100,
`80` for A100. The image records the simulator commit in `$NESTGPU_COMMIT`.

## Run

```sh
docker run --rm --gpus all -v "$PWD:/work" tinyhippo-nestgpu:latest python3 gpu_spike/<script>.py
```

From Git Bash on Windows, prefix the command with `MSYS_NO_PATHCONV=1`, or the `/work/...`
paths are rewritten. CPU NEST reference runs use `nest/nest-simulator:3.9` the same way, without
`--gpus`.

The Phase 0 scripts in `gpu_spike/` and their results are described in
[RESULTS_llm_gpu.md](../RESULTS_llm_gpu.md) §1.
