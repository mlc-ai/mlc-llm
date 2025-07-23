# Multi-stage optimized Dockerfile for MLC-LLM
# Target: <10GB final images, GitHub Actions compatible

# ──────────────────────────────────────────────────────────
# BASE STAGE - Minimal CUDA runtime with Python
# ──────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS base

# Minimize layers and reduce image size
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies in single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials (minimal set)
    build-essential cmake ninja-build pkg-config \
    # Python and Git
    python3 python3-pip python3-dev python3-venv \
    git git-lfs curl wget ca-certificates \
    # Required libraries (minimal)
    libtinfo5 libxml2-dev libzstd-dev zstd \
    libvulkan1 libvulkan-dev \
    # Cleanup in same layer
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* \
    && apt-get clean

# Install Rust efficiently
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path \
    && echo 'export PATH="/root/.cargo/bin:$PATH"' >> /root/.bashrc
ENV PATH="/root/.cargo/bin:$PATH"

# Create Python virtual environment (lighter than conda)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install essential Python packages in one layer
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir \
    transformers huggingface-hub tokenizers accelerate \
    requests numpy

WORKDIR /workspace

# ──────────────────────────────────────────────────────────
# DEVELOPMENT STAGE - Adds dev tools on top of base
# ──────────────────────────────────────────────────────────
FROM base AS development

# Install development dependencies efficiently
RUN pip install --no-cache-dir \
    # Testing
    pytest pytest-cov pytest-xdist pytest-asyncio \
    # Code quality  
    black isort pylint mypy pre-commit \
    # Development
    jupyter jupyterlab ipykernel \
    # Web framework
    fastapi uvicorn[standard] \
    # Additional utilities
    openai jsonschema pyyaml click tqdm rich

# Configure Git LFS
RUN git lfs install --system

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 mlcuser \
    && chown -R mlcuser:mlcuser /workspace

# Copy entrypoint
COPY docker/dev-entrypoint.sh /usr/local/bin/dev-entrypoint.sh
RUN chmod +x /usr/local/bin/dev-entrypoint.sh \
    && chown mlcuser:mlcuser /usr/local/bin/dev-entrypoint.sh

USER mlcuser
ENV HOME=/home/mlcuser
CMD ["/usr/local/bin/dev-entrypoint.sh"]

# ──────────────────────────────────────────────────────────
# BUILDER STAGE - Compiles MLC-LLM (temporary, not exported)
# ──────────────────────────────────────────────────────────
FROM base AS builder

# Copy source code
COPY . /workspace/
WORKDIR /workspace

# Configure Git LFS and build
RUN git lfs install \
    && mkdir -p build \
    && cd build \
    && python ../cmake/gen_cmake_config.py \
    && cmake .. \
       -DCMAKE_BUILD_TYPE=Release \
       -DUSE_CUDA=ON \
       -DUSE_VULKAN=ON \
       -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89" \
       -DCMAKE_INSTALL_PREFIX=/opt/mlc_llm \
    && make -j$(nproc) \
    && make install

# Install Python package
RUN cd python \
    && pip install --no-deps -e . \
    && python -c "import mlc_llm; print('✅ MLC-LLM build successful')"

# ──────────────────────────────────────────────────────────
# PRODUCTION STAGE - Minimal runtime (heavily optimized)
# ──────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS production

# Install only essential runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libvulkan1 \
    git-lfs \
    # Cleanup in same layer
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* \
    && apt-get clean

# Create virtual environment for production
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install minimal runtime Python dependencies
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu121 \
    transformers huggingface-hub tokenizers \
    fastapi uvicorn[standard] \
    requests numpy click

# Copy built artifacts from builder (minimal set)
COPY --from=builder /opt/mlc_llm/lib/libmlc_llm.so /usr/local/lib/
COPY --from=builder /opt/mlc_llm/lib/libtvm_runtime.so /usr/local/lib/
COPY --from=builder /workspace/python /opt/mlc_llm_python

# Set library and Python paths
ENV LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH" \
    PYTHONPATH="/opt/mlc_llm_python:$PYTHONPATH"

# Install the Python package (no dependencies, already installed)
RUN pip install --no-deps /opt/mlc_llm_python

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 mlcuser
USER mlcuser
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import mlc_llm" || exit 1

ENTRYPOINT ["python3", "-m", "mlc_llm"]
CMD ["--help"]

# ──────────────────────────────────────────────────────────
# MINIMAL STAGE - Ultra-lightweight for specific use cases
# ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS minimal

# Install only what's absolutely necessary
RUN apt-get update && apt-get install -y --no-install-recommends \
    git-lfs libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the Python package and essential libraries
COPY --from=builder /workspace/python /opt/mlc_llm_python
COPY --from=builder /opt/mlc_llm/lib/libmlc_llm.so /usr/local/lib/
COPY --from=builder /opt/mlc_llm/lib/libtvm_runtime.so /usr/local/lib/

ENV LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH" \
    PYTHONPATH="/opt/mlc_llm_python:$PYTHONPATH"

# Install minimal Python dependencies
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    transformers huggingface-hub tokenizers \
    && pip install --no-deps /opt/mlc_llm_python

RUN useradd --create-home --shell /bin/bash --uid 1000 mlcuser
USER mlcuser
WORKDIR /app

ENTRYPOINT ["python3", "-m", "mlc_llm"]
CMD ["--help"]