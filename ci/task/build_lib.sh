#!/bin/bash
export NUM_THREADS=$(nproc)
export PYTHONPATH="./python:$PYTHONPATH"

set -eo pipefail
set -x

GPU="cuda-12.1"
WORKSPACE_CWD=$(pwd)

source /multibuild/manylinux_utils.sh
source /opt/rh/gcc-toolset-11/enable # GCC-11 is the hightest GCC version compatible with NVCC < 12

mkdir -p $WORKSPACE_CWD/build/ && cd $WORKSPACE_CWD/build/
echo set\(USE_VULKAN ON\) >>config.cmake
if [[ ${GPU} == rocm* ]]; then
	echo set\(USE_ROCM ON\) >>config.cmake
	echo set\(USE_RCCL /opt/rocm/rccl/ \) >>config.cmake
elif [[ ${GPU} == cuda* ]]; then
	echo set\(USE_CUDA ON\) >>config.cmake
	# echo set\(USE_CUTLASS ON\) >>config.cmake
	echo set\(USE_NCCL ON\) >>config.cmake
fi
cmake .. && make -j${NUM_THREADS}
