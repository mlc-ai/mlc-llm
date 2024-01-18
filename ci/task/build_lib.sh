#!/bin/bash
set -eo pipefail
set -x
: ${NUM_THREADS:=$(nproc)}
: ${WORKSPACE_CWD:=$(pwd)}
: ${GPU:="cpu"}
export CCACHE_COMPILERCHECK=content
export CCACHE_NOHASHDIR=1
export CCACHE_DIR=/ccache

source /multibuild/manylinux_utils.sh
source /opt/rh/gcc-toolset-11/enable # GCC-11 is the hightest GCC version compatible with NVCC < 12

mkdir -p $WORKSPACE_CWD/build/ && cd $WORKSPACE_CWD/build/
echo set\(USE_VULKAN ON\) >>config.cmake
if [[ ${GPU} == rocm* ]]; then
	echo set\(USE_ROCM ON\) >>config.cmake
	echo set\(USE_RCCL /opt/rocm/rccl/ \) >>config.cmake
elif [[ ${GPU} == cuda* ]]; then
	echo set\(CMAKE_CUDA_COMPILER_LAUNCHER ccache\) >>config.cmake
	echo set\(CMAKE_CUDA_ARCHITECTURES "80;90"\) >>config.cmake
	echo set\(CMAKE_CUDA_FLAGS \"\$\{CMAKE_CUDA_FLAGS\} -t $NUM_THREADS\"\) >>config.cmake
	echo set\(USE_CUDA ON\) >>config.cmake
	echo set\(USE_CUBLAS ON\) >>config.cmake
	echo set\(USE_NCCL ON\) >>config.cmake
	echo set\(USE_FLASHINFER ON\) >>config.cmake
	echo set\(USE_CUTLASS ON\) >>config.cmake
fi

cat config.cmake

cmake .. && make -j${NUM_THREADS}
