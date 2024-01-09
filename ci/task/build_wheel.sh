#!/bin/bash
export NUM_THREADS=$(nproc)
export PYTHONPATH="./python:$PYTHONPATH"

set -eo pipefail
set -x

GPU="cuda-12.1"
WORKSPACE_CWD=$(pwd)
PYTHON_BIN=/opt/conda/envs/py38/bin/python

AUDITWHEEL_OPTS="--plat ${AUDITWHEEL_PLAT} -w repaired_wheels/"
AUDITWHEEL_OPTS="--exclude libtvm --exclude libtvm_runtime --exclude libvulkan ${AUDITWHEEL_OPTS}"
if [[ ${GPU} == rocm* ]]; then
	AUDITWHEEL_OPTS="--exclude libamdhip64 --exclude libhsa-runtime64 --exclude librocm_smi64 --exclude librccl ${AUDITWHEEL_OPTS}"
elif [[ ${GPU} == cuda* ]]; then
	AUDITWHEEL_OPTS="--exclude libcuda --exclude libcudart --exclude libnvrtc ${AUDITWHEEL_OPTS}"
fi

cd ${WORKSPACE_CWD}/python && ${PYTHON_BIN} setup.py bdist_wheel && auditwheel repair ${AUDITWHEEL_OPTS} dist/*.whl

rm -rf ${WORKSPACE_CWD}/wheels/
mv ${WORKSPACE_CWD}/python/repaired_wheels/ ${WORKSPACE_CWD}/wheels/
chown -R $ENV_USER_ID:$ENV_GROUP_ID ${WORKSPACE_CWD}/wheels/
