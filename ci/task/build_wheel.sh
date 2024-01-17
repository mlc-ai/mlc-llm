#!/bin/bash
set -eo pipefail
set -x
: ${NUM_THREADS:=$(nproc)}
: ${WORKSPACE_CWD:=$(pwd)}
: ${GPU:="cpu"}

AUDITWHEEL_OPTS="--plat ${AUDITWHEEL_PLAT} -w repaired_wheels/"
AUDITWHEEL_OPTS="--exclude libtvm --exclude libtvm_runtime --exclude libvulkan ${AUDITWHEEL_OPTS}"
if [[ ${GPU} == rocm* ]]; then
	AUDITWHEEL_OPTS="--exclude libamdhip64 --exclude libhsa-runtime64 --exclude librocm_smi64 --exclude librccl ${AUDITWHEEL_OPTS}"
elif [[ ${GPU} == cuda* ]]; then
	AUDITWHEEL_OPTS="--exclude libcuda --exclude libcudart --exclude libnvrtc --exclude libcublas --exclude libcublasLt ${AUDITWHEEL_OPTS}"
fi

cd ${WORKSPACE_CWD}/python && python setup.py bdist_wheel && auditwheel repair ${AUDITWHEEL_OPTS} dist/*.whl

rm -rf ${WORKSPACE_CWD}/wheels/
mv ${WORKSPACE_CWD}/python/repaired_wheels/ ${WORKSPACE_CWD}/wheels/
chown -R $ENV_USER_ID:$ENV_GROUP_ID ${WORKSPACE_CWD}/wheels/
