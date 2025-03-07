#!/bin/bash
set -eo pipefail
set -x
: ${NUM_THREADS:=$(nproc)}
: ${WORKSPACE_CWD:=$(pwd)}
: ${GPU:="cpu"}
export CCACHE_COMPILERCHECK=content
export CCACHE_NOHASHDIR=1
export CCACHE_DIR=/ccache

# Temporary workaround to install ccache.
conda install -c conda-forge ccache

if [[ ${GPU} != metal ]]; then
    source /multibuild/manylinux_utils.sh
    source /opt/rh/gcc-toolset-11/enable # GCC-11 is the hightest GCC version compatible with NVCC < 12
fi

mkdir -p $WORKSPACE_CWD/build/ && cd $WORKSPACE_CWD/build/
if [[ ${GPU} == rocm* ]]; then
    echo set\(USE_VULKAN ON\) >>config.cmake
    echo set\(USE_ROCM ON\) >>config.cmake
    echo set\(USE_RCCL /opt/rocm/rccl/ \) >>config.cmake
elif [[ ${GPU} == cuda* ]]; then
    echo set\(USE_VULKAN ON\) >>config.cmake
    echo set\(CMAKE_CUDA_COMPILER_LAUNCHER ccache\) >>config.cmake
    echo set\(CMAKE_CUDA_ARCHITECTURES "80;90;100;120"\) >>config.cmake
    echo set\(CMAKE_CUDA_FLAGS \"\$\{CMAKE_CUDA_FLAGS\} -t $NUM_THREADS\"\) >>config.cmake
    echo set\(USE_CUDA ON\) >>config.cmake
    echo set\(USE_CUBLAS ON\) >>config.cmake
    echo set\(USE_NCCL ON\) >>config.cmake
    echo set\(USE_FLASHINFER ON\) >>config.cmake
    echo set\(FLASHINFER_ENABLE_FP8 OFF\) >>config.cmake
    echo set\(FLASHINFER_ENABLE_BF16 OFF\) >>config.cmake
    echo set\(FLASHINFER_GEN_GROUP_SIZES 1 4 6 8\) >>config.cmake
    echo set\(FLASHINFER_GEN_PAGE_SIZES 16\) >>config.cmake
    echo set\(FLASHINFER_GEN_HEAD_DIMS 128\) >>config.cmake
    echo set\(FLASHINFER_GEN_KV_LAYOUTS 0 1\) >>config.cmake
    echo set\(FLASHINFER_GEN_POS_ENCODING_MODES 0 1\) >>config.cmake
    echo set\(FLASHINFER_GEN_ALLOW_FP16_QK_REDUCTIONS "false"\) >>config.cmake
    echo set\(FLASHINFER_GEN_CASUALS "false" "true"\) >>config.cmake
    echo set\(USE_CUTLASS ON\) >>config.cmake
elif [[ ${GPU} == metal ]]; then
    export CCACHE_DIR=$HOME/ci/ccache
    echo set\(USE_METAL ON\) >>config.cmake
else
    echo set\(USE_VULKAN ON\) >>config.cmake
fi

cat config.cmake

cmake .. && make -j${NUM_THREADS}
