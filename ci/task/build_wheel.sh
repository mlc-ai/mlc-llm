#!/bin/bash
set -eo pipefail
set -x
: ${NUM_THREADS:=$(nproc)}
: ${WORKSPACE_CWD:=$(pwd)}
: ${GPU:="cpu"}

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
