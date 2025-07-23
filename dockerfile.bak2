### ────── BASE WITH MICROMAMBA + CUDA ──────
FROM mambaorg/micromamba:jammy-cuda-12.1.0 AS base

# 🛠 Optimize micromamba behavior
RUN micromamba config set extract_threads 1
ENV MAMBA_NO_LOW_SPEED_LIMIT=0

USER root
# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git git-lfs build-essential pkg-config \
    libtinfo5 libxml2-dev libzstd-dev zstd vulkan-tools \
    libvulkan1 libvulkan-dev cmake ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Rust (for tokenizers)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Python & CUDA toolchain with channels defined
RUN micromamba install -y -n base \
    -c conda-forge -c nvidia \
    python=3.11 cuda-toolkit=12.1 \
    cmake=3.24.* git git-lfs rust cuda-nvcc \
    libgcc-ng libstdcxx-ng zstd && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /workspace

### ────── DEVELOPMENT STAGE ──────
FROM base AS development

RUN micromamba install -y -n base \
    -c conda-forge \
    pytest pytest-cov pytest-xdist black isort pylint mypy \
    jupyter jupyterlab tensorboard ipykernel pre-commit && \
    pip install --no-cache-dir \
      fastapi uvicorn[standard] openai jsonschema requests \
      transformers huggingface-hub accelerate && \
    micromamba clean --all --yes

RUN git lfs install

RUN useradd --create-home --shell /bin/bash mlcuser && \
    chown -R mlcuser:mlcuser /workspace

COPY --chown=mlcuser:mlcuser docker/dev-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/dev-entrypoint.sh

USER mlcuser
WORKDIR /workspace
CMD ["/usr/local/bin/dev-entrypoint.sh"]

### ────── BUILDER STAGE ──────
FROM base AS builder
WORKDIR /workspace
COPY . .

RUN git lfs install && \
    mkdir -p build && cd build && \
      python ../cmake/gen_cmake_config.py && \
      cmake .. -DCMAKE_BUILD_TYPE=Release \
        -DUSE_CUDA=ON -DUSE_VULKAN=ON \
        -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89" && \
      make -j$(nproc)

RUN cd python && pip install -e . && \
    python -c "import mlc_llm; print('MLC‑LLM OK')"

### ────── PRODUCTION STAGE ──────
FROM nvidia/cuda:12.9.1-runtime-ubuntu22.04 AS production
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git-lfs libvulkan1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /workspace/build/libmlc_llm.so /usr/local/lib/
COPY --from=builder /workspace/build/libtvm_runtime.so /usr/local/lib/
COPY --from=builder /workspace/python /opt/mlc_llm_python

ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"
ENV PYTHONPATH="/opt/mlc_llm_python:${PYTHONPATH}"
RUN pip3 install --no-cache-dir /opt/mlc_llm_python

RUN useradd --create-home --shell /bin/bash mlcuser && \
    chown -R mlcuser:mlcuser /workspace
USER mlcuser
WORKDIR /app

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import mlc_llm" || exit 1

ENTRYPOINT ["python3", "-m", "mlc_llm"]
CMD ["--help"]
