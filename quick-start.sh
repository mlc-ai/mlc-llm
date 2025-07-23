#!/bin/bash
# quick-start.sh - Platform-agnostic setup script for MLC-LLM
# Works on macOS, Linux, and Windows (via Git Bash or WSL)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Main setup function
main() {
    echo "🚀 MLC-LLM Docker-First Setup"
    echo "=============================="
    echo ""
    
    local os=$(detect_os)
    log_info "Detected OS: $os"
    echo ""

    # Check prerequisites
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command_exists docker; then
        log_error "Docker not found!"
        echo "Please install Docker Desktop:"
        case $os in
            "macos")
                echo "  https://docs.docker.com/desktop/mac/install/"
                ;;
            "windows")
                echo "  https://docs.docker.com/desktop/windows/install/"
                ;;
            "linux")
                echo "  https://docs.docker.com/engine/install/"
                ;;
        esac
        exit 1
    fi
    log_success "Docker found: $(docker --version)"
    
    # Check Docker Compose
    if ! command_exists "docker compose" && ! docker compose version >/dev/null 2>&1; then
        log_error "Docker Compose not found!"
        echo "Please update Docker or install Docker Compose separately"
        exit 1
    fi
    log_success "Docker Compose found"
    
    # Check Git
    if ! command_exists git; then
        log_error "Git not found! Please install Git first."
        exit 1
    fi
    log_success "Git found: $(git --version)"
    
    # Check if we're in MLC-LLM repository
    if [ ! -f "CMakeLists.txt" ] || [ ! -d "python" ]; then
        log_error "This doesn't appear to be the MLC-LLM repository root"
        echo "Please run this script from the MLC-LLM repository directory"
        exit 1
    fi
    log_success "MLC-LLM repository detected"
    
    # Check Git LFS
    if ! command_exists "git lfs"; then
        log_warning "Git LFS not found. Installing Git LFS is recommended for large files."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Please install Git LFS: https://git-lfs.github.io/"
            exit 1
        fi
    else
        log_success "Git LFS found"
        git lfs install --local >/dev/null 2>&1 || true
    fi
    
    # Check submodules
    if git submodule status | grep -q "^-"; then
        log_warning "Git submodules not initialized"
        read -p "Initialize submodules now? (Y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            log_error "Submodules are required for building MLC-LLM"
            exit 1
        fi
        log_info "Initializing submodules (this may take a while)..."
        git submodule update --init --recursive
        log_success "Submodules initialized"
    else
        log_success "Git submodules ready"
    fi
    
    echo ""
    log_info "All prerequisites satisfied!"
    echo ""
    
    # Docker setup
    log_info "Setting up Docker environment..."
    
    # Test Docker daemon
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon not running!"
        case $os in
            "macos"|"windows")
                echo "Please start Docker Desktop"
                ;;
            "linux")
                echo "Please start Docker service: sudo systemctl start docker"
                ;;
        esac
        exit 1
    fi
    log_success "Docker daemon is running"
    
    # Check for existing containers
    if docker ps -a --format "table {{.Names}}" | grep -q "mlc-dev"; then
        log_warning "Existing MLC development container found"
        read -p "Remove existing container and rebuild? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Removing existing containers..."
            docker compose down >/dev/null 2>&1 || true
            docker rm -f mlc-dev mlc-prod >/dev/null 2>&1 || true
            log_success "Existing containers removed"
        fi
    fi
    
    # Build development image
    log_info "Building MLC-LLM development image..."
    log_warning "This will take 10-20 minutes on first run (downloading and compiling)"
    echo ""
    
    if ! docker build -f dockerfile --target development -t mlc-llm:dev1 .; then
        log_error "Failed to build development image"
        echo "Check the error messages above and try again"
        exit 1
    fi
    
    log_success "Development image built successfully!"
    echo ""
    
    # Test the image
    log_info "Testing the development image..."
    if docker run --rm mlc-llm:dev python -c "import sys; print(f'Python {sys.version}')"; then
        log_success "Development image test passed"
    else
        log_error "Development image test failed"
        exit 1
    fi
    
    # Start development environment
    log_info "Starting development environment..."
    if docker compose up -d; then
        log_success "Development environment started"
    else
        log_error "Failed to start development environment"
        exit 1
    fi
    
    # Wait for container to be ready
    log_info "Waiting for container to be ready..."
    for i in {1..30}; do
        if docker exec mlc-dev python -c "import mlc_llm" >/dev/null 2>&1; then
            break
        fi
        if [ $i -eq 30 ]; then
            log_error "Container failed to start properly"
            exit 1
        fi
        sleep 2
        echo -n "."
    done
    echo ""
    log_success "Container is ready!"
    
    # Success message
    echo ""
    echo "🎉 MLC-LLM Development Environment Setup Complete!"
    echo "=================================================="
    echo ""
    echo "What's running:"
    echo "  • Development container: mlc-dev"
    echo "  • Jupyter Lab: http://localhost:8888"
    echo "  • TensorBoard: http://localhost:6006"
    echo ""
    echo "Quick commands:"
    echo "  make shell         # Open development shell"
    echo "  make jupyter       # Start Jupyter Lab"
    echo "  make test-fast     # Run fast tests"
    echo "  make help          # Show all commands"
    echo ""
    echo "Next steps:"
    echo "  1. Open development shell: make shell"
    echo "  2. Build MLC-LLM: make build-mlc"
    echo "  3. Run tests: make test-fast"
    echo ""
    echo "Documentation:"
    echo "  • README-DOCKER.md for detailed usage"
    echo "  • make help for all available commands"
    echo ""
    
    # Optional: Open shell
    read -p "Open development shell now? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        log_info "Opening development shell..."
        echo "Type 'exit' to leave the container"
        docker exec -it mlc-dev /bin/bash
    fi
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${YELLOW}Setup interrupted by user${NC}"; exit 1' INT

# Run main function
main "$@"