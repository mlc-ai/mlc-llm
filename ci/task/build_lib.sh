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
if [[ ${GPU} != metal ]]; then
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
fi
conda install -c conda-forge ccache

if [[ ${GPU} != metal ]]; then
    source /multibuild/manylinux_utils.sh
    source /opt/rh/gcc-toolset-11/enable # GCC-11 is the hightest GCC version compatible with NVCC < 12
fi

mkdir -p $WORKSPACE_CWD/build
if [[ ${GPU} == rocm* ]]; then
    echo set\(USE_VULKAN ON\) >>config.cmake
    echo set\(USE_ROCM ON\) >>config.cmake
    echo set\(USE_RCCL /opt/rocm/rccl/ \) >>config.cmake
elif [[ ${GPU} == cuda* ]]; then
    echo set\(USE_VULKAN ON\) >>config.cmake
    echo set\(CMAKE_CUDA_COMPILER_LAUNCHER ccache\) >>config.cmake
    echo set\(CMAKE_CUDA_ARCHITECTURES "80;90;100;110;120"\) >>config.cmake
    echo set\(CMAKE_CUDA_FLAGS \"\$\{CMAKE_CUDA_FLAGS\} -t $NUM_THREADS\"\) >>config.cmake
    echo set\(USE_CUDA ON\) >>config.cmake
    echo set\(USE_CUBLAS ON\) >>config.cmake
    echo set\(USE_NCCL ON\) >>config.cmake
elif [[ ${GPU} == metal ]]; then
    export CCACHE_DIR=$HOME/ci/ccache
    echo set\(USE_METAL ON\) >>config.cmake
else
    echo set\(USE_VULKAN ON\) >>config.cmake
fi

cat config.cmake

AUDITWHEEL_OPTS="--plat ${AUDITWHEEL_PLAT} -w repaired_wheels/"
AUDITWHEEL_OPTS="--exclude libtvm --exclude libtvm_runtime --exclude libtvm_ffi --exclude libvulkan ${AUDITWHEEL_OPTS}"
if [[ ${GPU} == rocm* ]]; then
    AUDITWHEEL_OPTS="--exclude libamdhip64 --exclude libhsa-runtime64 --exclude librocm_smi64 --exclude librccl ${AUDITWHEEL_OPTS}"
elif [[ ${GPU} == cuda* ]]; then
    AUDITWHEEL_OPTS="--exclude libcuda --exclude libcudart --exclude libnvrtc --exclude libcublas --exclude libcublasLt ${AUDITWHEEL_OPTS}"
fi

rm -rf ${WORKSPACE_CWD}/dist
cd ${WORKSPACE_CWD} && pip wheel --no-deps -w dist . -v

rm -rf ${WORKSPACE_CWD}/wheels/
if [[ ${GPU} != metal ]]; then
    mkdir -p ${WORKSPACE_CWD}/repaired_wheels
    rm -rf ${WORKSPACE_CWD}/repaired_wheels/*
    auditwheel repair ${AUDITWHEEL_OPTS} dist/*.whl
    mv ${WORKSPACE_CWD}/repaired_wheels/ ${WORKSPACE_CWD}/wheels/
else
    mkdir ${WORKSPACE_CWD}/wheels/
    mv dist/*.whl ${WORKSPACE_CWD}/wheels/
fi

chown -R $ENV_USER_ID:$ENV_GROUP_ID ${WORKSPACE_CWD}/wheels/
