#!/bin/bash
set -eo pipefail
set -x
: ${NUM_THREADS:=$(nproc)}
: ${WORKSPACE_CWD:=$(pwd)}
: ${GPU:="cpu"}

NUM_THREADS=8

if [[ ${GPU} == cuda* ]]; then
	TARGET=cuda
	pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cu121
	export LD_LIBRARY_PATH=/usr/local/cuda/compat/:$LD_LIBRARY_PATH
elif [[ ${GPU} == rocm* ]]; then
	TARGET=rocm
	pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-rocm57
else
	TARGET=vulkan
	pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly
fi
pip install wheels/*.whl
python tests/python/integration/test_model_compile.py $TARGET $NUM_THREADS
