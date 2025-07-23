# Ultra-minimal Dockerfile optimized for GitHub Actions disk space limits
# Strategy: Build only what's absolutely necessary, skip development image in CI

# ──────────────────────────────────────────────────────────
# BUILDER STAGE - Compile MLC-LLM (will be discarded)
# ──────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install minimal build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build \
    python3 python3-pip python3-dev \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Install Rust (minimal)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal
ENV PATH="/root/.cargo/bin:$PATH"

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install only essential build dependencies
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu121 \
    cmake ninja

# Copy source and build
WORKDIR /workspace
COPY . .

# Configure and build MLC-LLM
RUN mkdir -p build && cd build && \
    python ../cmake/gen_cmake_config.py && \
    cmake .. \
       -DCMAKE_BUILD_TYPE=Release \
       -DUSE_CUDA=ON \
       -DUSE_VULKAN=OFF \
       -DCMAKE_CUDA_ARCHITECTURES="75;80;86" \
       -GNinja && \
    ninja

# Install Python package
RUN cd python && pip install --no-deps -e .

# ──────────────────────────────────────────────────────────
# PRODUCTION STAGE - Ultra minimal runtime
# ──────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS production

# Install absolute minimum runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* \
    && apt-get clean

# Create venv and install minimal runtime deps
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu121

# Copy only essential artifacts
COPY --from=builder /workspace/build/libmlc_llm.so /usr/local/lib/
COPY --from=builder /workspace/build/libtvm_runtime.so /usr/local/lib/
COPY --from=builder /workspace/python/mlc_llm /opt/mlc_llm/mlc_llm
COPY --from=builder /workspace/python/setup.py /opt/mlc_llm/

# Set up Python package
ENV LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH" \
    PYTHONPATH="/opt/mlc_llm:$PYTHONPATH"

RUN cd /opt/mlc_llm && pip install --no-deps .

# Create user
RUN useradd --create-home --shell /bin/bash --uid 1000 mlcuser
USER mlcuser
WORKDIR /app

ENTRYPOINT ["python3", "-m", "mlc_llm"]
CMD ["--help"]

# ──────────────────────────────────────────────────────────
# DEVELOPMENT STAGE - Only for local development
# ──────────────────────────────────────────────────────────
FROM builder AS development

# Install dev dependencies
RUN pip install --no-cache-dir \
    pytest black isort \
    fastapi uvicorn \
    transformers huggingface-hub

RUN useradd --create-home --shell /bin/bash --uid 1000 mlcuser && \
    chown -R mlcuser:mlcuser /workspace

USER mlcuser
WORKDIR /workspace

CMD ["/bin/bash"]

# ──────────────────────────────────────────────────────────
# CI STAGE - Minimal for testing only
# ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS ci

# Install test dependencies only
RUN pip install --no-cache-dir \
    pytest black isort pylint

# Copy source for testing
COPY python /workspace/python
COPY tests /workspace/tests
WORKDIR /workspace

CMD ["python", "-m", "pytest", "tests/", "-v"]