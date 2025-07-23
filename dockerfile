# Simplified Dockerfile optimized for GitHub Actions
# Addresses registry naming and disk space issues

# ──────────────────────────────────────────────────────────
# BUILDER STAGE - Compile MLC-LLM (temporary)
# ──────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

# Minimize environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies in single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build python3 python3-pip python3-dev python3-venv \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* \
    && apt-get clean

# Install Rust (minimal profile)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal
ENV PATH="/root/.cargo/bin:$PATH"

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install minimal build dependencies
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

WORKDIR /workspace
COPY . .

# Build MLC-LLM
RUN mkdir -p build && cd build && \
    python ../cmake/gen_cmake_config.py && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DUSE_VULKAN=OFF -GNinja && \
    ninja -j2  # Limit parallel jobs to save memory

# Install Python package
RUN cd python && pip install --no-deps -e .

# Verify build
RUN python -c "import mlc_llm; print('✅ MLC-LLM build successful')"

# ──────────────────────────────────────────────────────────
# PRODUCTION STAGE - Minimal runtime
# ──────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS production

# Install minimal runtime dependencies INCLUDING python3-venv
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* \
    && apt-get clean

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/opt/mlc_llm"

# Install minimal runtime Python packages
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Copy built artifacts
COPY --from=builder /workspace/build/libmlc_llm.so /usr/local/lib/
COPY --from=builder /workspace/build/libtvm_runtime.so /usr/local/lib/
COPY --from=builder /workspace/python/mlc_llm /opt/mlc_llm/mlc_llm
COPY --from=builder /workspace/python/setup.py /opt/mlc_llm/

# Set environment (PYTHONPATH now defined above)
ENV LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"

# Install package
RUN cd /opt/mlc_llm && pip install --no-deps .

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
# CI STAGE - Ultra minimal for testing
# ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS ci

# Install test dependencies only
RUN pip install --no-cache-dir pytest black isort

# Copy source for testing (no build needed)
COPY python /workspace/python
COPY tests /workspace/tests
WORKDIR /workspace

CMD ["python", "-m", "pytest", "tests/", "-v", "--maxfail=5"]

# ──────────────────────────────────────────────────────────
# DEVELOPMENT STAGE - For local use only (not built in CI)
# ──────────────────────────────────────────────────────────
FROM builder AS development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest pytest-cov black isort pylint mypy \
    jupyter jupyterlab fastapi uvicorn \
    transformers huggingface-hub

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 mlcuser && \
    chown -R mlcuser:mlcuser /workspace

USER mlcuser
WORKDIR /workspace

# Copy entrypoint script
COPY --chown=mlcuser:mlcuser docker/dev-entrypoint.sh /usr/local/bin/
USER root
RUN chmod +x /usr/local/bin/dev-entrypoint.sh
USER mlcuser

CMD ["/usr/local/bin/dev-entrypoint.sh"]