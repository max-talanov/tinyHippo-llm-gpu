# NEST GPU runtime for tinyHippo-llm-gpu.
#
# The simulator source is a separate checkout (branch `tinyhippo` of
# tfatykhov/nest-gpu, tracking nest/nest-gpu main), passed in as the named
# build context `nestgpu`. Build from the repository root:
#
#   docker build gpu \
#       --build-context nestgpu=../nest-gpu-tinyhippo \
#       --build-arg NESTGPU_COMMIT=$(git -C ../nest-gpu-tinyhippo rev-parse --short HEAD) \
#       -t tinyhippo-nestgpu:latest
#
# Run a script from this repository on the GPU:
#
#   docker run --rm --gpus all -v "$PWD:/work" tinyhippo-nestgpu:latest python3 gpu_spike/<script>.py
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Compute capability of the target GPU. 89 = Ada (RTX 40-series).
#   A100 80 | RTX 3080/3090 86 | RTX 4060 Ti/4090 89 | H100 90
ARG GPU_ARCH=89

# MPI stays ON (the upstream default); OpenMPI supports singleton MPI_INIT, so
# `python3 script.py` runs as one rank without mpirun.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        libomp-dev \
        libopenmpi-dev \
        openmpi-bin \
        python3-all-dev \
        python3-numpy \
        python3-scipy \
        python3-matplotlib \
        python3-h5py \
    && rm -rf /var/lib/apt/lists/*

COPY --from=nestgpu . /nest-gpu-src

RUN mkdir -p /nest-gpu-build && cd /nest-gpu-build \
    && cmake \
        -DCMAKE_INSTALL_PREFIX:PATH=/opt/nestgpu \
        -Dwith-gpu-arch="${GPU_ARCH}" \
        -Dwith-mpi=ON \
        /nest-gpu-src \
    && make -j"$(nproc)" \
    && make install \
    && rm -rf /nest-gpu-build

# Declared here, not at the top: a build arg invalidates the cache of every
# RUN after its declaration, and this one changes with each source commit.
ARG NESTGPU_COMMIT=unknown
ENV NESTGPU_LIB=/opt/nestgpu/lib/nestgpu/libnestgpukernel.so
ENV PYTHONPATH=/opt/nestgpu/lib/python3.10/site-packages
ENV NESTGPU_COMMIT=${NESTGPU_COMMIT}
ENV MPLBACKEND=Agg
ENV PYTHONUNBUFFERED=1
LABEL org.opencontainers.image.revision="${NESTGPU_COMMIT}"

WORKDIR /work
CMD ["/bin/bash"]
