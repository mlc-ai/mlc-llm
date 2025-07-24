set -e

echo "🚀 Setting up MLC-LLM development environment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check for NVIDIA Docker support
if ! docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo "⚠️  NVIDIA Docker support not detected. GPU features may not work."
    echo "   Continuing with CPU-only setup..."
    GPU_ARGS=""
else
    echo "✅ NVIDIA Docker support detected"
    GPU_ARGS="--gpus all"
fi

# Build development image if it doesn't exist
DEV_IMAGE="mlc-llm:dev"
if [[ "$(docker images -q $DEV_IMAGE 2> /dev/null)" == "" ]]; then
    echo "📦 Building development image (this may take 10-20 minutes first time)..."
    docker build --target development -t $DEV_IMAGE .
else
    echo "✅ Development image already exists: $DEV_IMAGE"
fi

# Function to run development container
run_dev() {
    echo "🔧 Starting MLC-LLM development container..."
    docker run -it --rm \
        $GPU_ARGS \
        --name mlc-dev-$(date +%s) \
        -v "$(pwd):/workspace" \
        -w /workspace \
        --shm-size=2g \
        -e PYTHONPATH="/workspace/python:/workspace/3rdparty/tvm/python" \
        -e LD_LIBRARY_PATH="/workspace/build:/usr/local/lib" \
        $DEV_IMAGE \
        bash
}

# Function to run Jupyter notebook
run_jupyter() {
    echo "📓 Starting Jupyter notebook server..."
    docker run -it --rm \
        $GPU_ARGS \
        --name mlc-jupyter-$(date +%s) \
        -v "$(pwd):/workspace" \
        -w /workspace \
        -p 8888:8888 \
        --shm-size=2g \
        -e PYTHONPATH="/workspace/python:/workspace/3rdparty/tvm/python" \
        -e LD_LIBRARY_PATH="/workspace/build:/usr/local/lib" \
        $DEV_IMAGE \
        jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
}

# Function to run tests
run_tests() {
    echo "🧪 Running tests in development container..."
    docker run --rm \
        $GPU_ARGS \
        -v "$(pwd):/workspace" \
        -w /workspace \
        -e PYTHONPATH="/workspace/python:/workspace/3rdparty/tvm/python" \
        $DEV_IMAGE \
        python -m pytest tests/ -v
}

# Parse command line arguments
case "${1:-shell}" in
    "shell"|"bash"|"dev")
        run_dev
        ;;
    "jupyter"|"notebook")
        run_jupyter
        ;;
    "test"|"tests")
        run_tests
        ;;
    "build")
        echo "🔨 Rebuilding development image..."
        docker build --target development -t $DEV_IMAGE .
        echo "✅ Development image rebuilt"
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  shell     Start interactive development shell (default)"
        echo "  jupyter   Start Jupyter notebook server"
        echo "  test      Run test suite"
        echo "  build     Rebuild development image"
        echo "  help      Show this help"
        echo ""
        echo "Examples:"
        echo "  $0              # Start development shell"
        echo "  $0 jupyter      # Start Jupyter on http://localhost:8888"
        echo "  $0 test         # Run tests"
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac