#!/bin/bash
# docker/build-mlc.sh - Space-optimized MLC-LLM build script
# Inspired by dusty-nv's jetson-containers approach

set -e

echo "🚀 Starting space-optimized MLC-LLM build"

# Environment setup for space efficiency
export DEBIAN_FRONTEND=noninteractive
export MAKEFLAGS="-j1"  # Single thread to save memory
export PYTHONDONTWRITEBYTECODE=1
export PIP_NO_CACHE_DIR=1

# Build configuration
cd /workspace

# Clean any existing build artifacts
rm -rf build/ || true
mkdir -p build && cd build

echo "📋 Generating CMake configuration..."
# Automated config generation (no interactive input)
printf '\ny\nn\ny\nn\nn\nn\nn\nn\n' | python3 ../cmake/gen_cmake_config.py

echo "🔧 Configuring CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=ON \
    -DUSE_VULKAN=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86"

echo "🔨 Building MLC-LLM (single-threaded for memory efficiency)..."
make -j1

echo "📦 Installing Python package..."
cd /workspace/python
pip3 install -e . --no-deps

echo "✅ Verifying installation..."
# Add TVM to Python path and test imports
python3 -c "
import sys
sys.path.insert(0, '/workspace/3rdparty/tvm/python')
import tvm
print('✅ TVM import successful')
import mlc_llm  
print('✅ MLC-LLM import successful')
print('✅ Build verification completed')
"

echo "🧹 Cleaning up build artifacts to save space..."
cd /workspace/build

# Remove unnecessary build artifacts but keep essential libraries
find . -name "*.o" -delete 2>/dev/null || true
find . -name "*.a" -delete 2>/dev/null || true
find . -type d -name "CMakeFiles" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "_deps" -exec rm -rf {} + 2>/dev/null || true

# Keep only essential .so files
find . -name "*.so" -type f | head -10  # Show what we're keeping

echo "📊 Final build directory size:"
du -sh /workspace/build/ || true

echo "✅ Space-optimized MLC-LLM build completed successfully!"
echo "🎯 Ready for production deployment"