#!/bin/bash
# docker/dev-entrypoint.sh - Development container entrypoint script

set -e

echo "🚀 Starting MLC-LLM Development Environment"
echo "Container OS: $(uname -a)"
echo "Python: $(python --version)"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'false')"
echo "Working Directory: $(pwd)"
echo "User: $(whoami)"

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate mlc-llm

# Ensure proper permissions for mounted volumes
if [ -d "/workspace" ]; then
    echo "Setting up workspace permissions..."
    # Only try to change ownership if we have permission
    if [ -w "/workspace" ]; then
        find /workspace -type d -exec chmod 755 {} + 2>/dev/null || true
        find /workspace -type f -exec chmod 644 {} + 2>/dev/null || true
    fi
fi

# Initialize git configuration if not set
if [ ! -f ~/.gitconfig ]; then
    echo "Setting up git configuration..."
    git config --global user.name "MLC Developer"
    git config --global user.email "dev@mlc.ai"
    git config --global init.defaultBranch main
    git config --global safe.directory /workspace
fi

# Set up git LFS if not already done
if [ -d "/workspace/.git" ]; then
    cd /workspace
    git lfs install --local 2>/dev/null || echo "Git LFS already configured"
fi

# Install pre-commit hooks if .pre-commit-config.yaml exists
if [ -f "/workspace/.pre-commit-config.yaml" ]; then
    cd /workspace
    echo "Installing pre-commit hooks..."
    pre-commit install 2>/dev/null || echo "Pre-commit hooks already installed"
fi

# Start Jupyter Lab in background if requested
if [ "${START_JUPYTER:-false}" = "true" ]; then
    echo "Starting Jupyter Lab..."
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
        --NotebookApp.token='' --NotebookApp.password='' \
        --notebook-dir=/workspace &
    echo "Jupyter Lab will be available at http://localhost:8888"
fi

# Start TensorBoard in background if logs directory exists
if [ -d "/workspace/logs" ]; then
    echo "Starting TensorBoard..."
    tensorboard --logdir=/workspace/logs --host=0.0.0.0 --port=6006 &
    echo "TensorBoard will be available at http://localhost:6006"
fi

# Display helpful information
echo ""
echo "🔧 Development Environment Ready!"
echo ""
echo "Available commands:"
echo "  python -m mlc_llm --help    # MLC-LLM CLI"
echo "  python -m pytest tests/    # Run tests"
echo "  jupyter lab                # Start Jupyter (if not auto-started)"
echo "  make help                  # Show Makefile commands"
echo ""
echo "Useful directories:"
echo "  /workspace                 # Your source code (mounted)"
echo "  /workspace/models          # Model storage"
echo "  /workspace/logs            # Logs and TensorBoard"
echo ""
echo "To build MLC-LLM:"
echo "  mkdir -p build && cd build"
echo "  python ../cmake/gen_cmake_config.py"
echo "  cmake .. && make -j\$(nproc)"
echo ""
echo "Network access:"
echo "  Jupyter Lab:  http://localhost:8888"
echo "  TensorBoard:  http://localhost:6006"
echo "  FastAPI:      http://localhost:8000"
echo ""

# Change to workspace directory
cd /workspace

# Execute the command passed to the container, or start an interactive shell
if [ $# -eq 0 ]; then
    echo "Starting interactive shell..."
    echo "Type 'exit' to leave the container"
    exec /bin/bash
else
    echo "Executing command: $*"
    exec "$@"
fi