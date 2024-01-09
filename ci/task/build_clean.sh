#!/bin/bash
export NUM_THREADS=$(nproc)
export PYTHONPATH="./python:$PYTHONPATH"

set -eo pipefail
set -x

GPU="cuda-12.1"
WORKSPACE_CWD=$(pwd)
PYTHON_BIN=/opt/conda/envs/py38/bin/python

rm -rf ${WORKSPACE_CWD}/build/ \
	${WORKSPACE_CWD}/python/dist/ \
	${WORKSPACE_CWD}/python/build/ \
	${WORKSPACE_CWD}/python/mlc_chat.egg-info
