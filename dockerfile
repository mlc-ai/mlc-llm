# Multi-stage Docker build for MLC-LLM
# Supports both development and build environments with GPU acceleration

# Base image with CUDA support
ARG CUDA_VERSION=12.2
ARG UBUNTU_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/conda/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    git-lfs \
    build-essential \
    libvulkan-loader \
    libvulkan-dev \
    vulkan-tools \
    zstd \
    pkg-config \
    libtinfo5 \
    libxml2-dev \
    libzstd-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (required for tokenizers)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Create conda environment
RUN conda create -n mlc-llm python=3.11 -y && \
    echo "conda activate mlc-llm" >> ~/.bashrc

# Activate conda environment for subsequent RUN commands
SHELL ["/opt/conda/bin/conda", "run", "-n", "mlc-llm", "/bin/bash", "-c"]

# Install CMake and build dependencies via conda
RUN conda install -c conda-forge \
    "cmake>=3.24" \
    rust \
    git \
    libgcc-ng \
    zstd \
    -y

# Set working directory
WORKDIR /workspace

# Development stage - includes all dev tools and source code
FROM base AS development

# Install development dependencies
RUN conda install -c conda-forge \
    pytest \
    black \
    isort \
    pylint \
    mypy \
    jupyter \
    tensorboard \
    -y

# Install pip packages for development
RUN pip install --no-cache-dir \
    pre-commit \
    pytest-cov \
    pytest-xdist \
    ipykernel

# Copy source code (for development)
COPY . /workspace/

# Set up git-lfs
RUN git lfs install

# Configure development environment
ENV MLC_LLM_SOURCE_DIR=/workspace
ENV PYTHONPATH="/workspace/python:$PYTHONPATH"

# Create non-root user for development
RUN useradd --create-home --shell /bin/bash --uid 1000 mlcuser && \
    chown -R mlcuser:mlcuser /workspace
USER mlcuser

# Development entrypoint
CMD ["/bin/bash"]

# Build stage - optimized for building MLC-LLM
FROM base AS builder

# Copy source code
COPY . /workspace/

# Set up git-lfs and clone submodules
RUN git lfs install && \
    git submodule update --init --recursive

# Create build directory and configure
RUN mkdir -p build && cd build && \
    python ../cmake/gen_cmake_config.py && \
    cmake .. && \
    cmake --build . --parallel $(nproc)

# Install MLC-LLM Python package
RUN cd python && pip install -e .

# Validate installation
RUN python -c "import mlc_llm; print(mlc_llm)" && \
    mlc_llm chat -h

# Production stage - minimal runtime image
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS production

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libvulkan1 \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Copy built libraries and Python package from builder
COPY --from=builder /workspace/build/libmlc_llm.so /usr/local/lib/
COPY --from=builder /workspace/build/libtvm_runtime.so /usr/local/lib/
COPY --from=builder /opt/conda/envs/mlc-llm/lib/python3.11/site-packages /usr/local/lib/python3.11/dist-packages/

# Set library path
ENV LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 mlcuser
USER mlcuser

WORKDIR /app

# Production entrypoint
ENTRYPOINT ["python3", "-m", "mlc_llm"]
CMD ["--help"]

# Multi-platform build target selector
FROM ${TARGETPLATFORM:-production} AS final