#!/bin/bash
set -eo pipefail
set -x
: ${NUM_THREADS:=$(nproc)}
: ${WORKSPACE_CWD:=$(pwd)}
: ${GPU:="cpu"}

pip install --force-reinstall wheels/*.whl

if [[ ${GPU} == cuda* ]]; then
	TARGET=cuda
	pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cu121
	export LD_LIBRARY_PATH=/usr/local/cuda/compat/:$LD_LIBRARY_PATH
elif [[ ${GPU} == rocm* ]]; then
	TARGET=rocm
	pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-rocm57
elif [[ ${GPU} == metal ]]; then
	TARGET=metal
	pip install --pre -U --force-reinstal -f https://mlc.ai/wheels mlc-ai-nightly
elif [[ ${GPU} == wasm* ]]; then
	TARGET=wasm
	pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly
	export TVM_HOME=$(dirname $(python -c 'import tvm; print(tvm.__file__)'))
	export MLC_LLM_SOURCE_DIR=$(pwd)
	cd $TVM_HOME/web/ && make -j${NUM_THREADS} && cd -
	cd $MLC_LLM_SOURCE_DIR/web/ && make -j${NUM_THREADS} && cd -
elif [[ ${GPU} == ios ]]; then
	TARGET=ios
	pip install --pre -U --force-reinstal -f https://mlc.ai/wheels mlc-ai-nightly
elif [[ ${GPU} == android* ]]; then
	TARGET=android
	pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly
	source /android_env_vars.sh
else
	TARGET=vulkan
	pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly
fi

python tests/python/integration/test_model_compile.py $TARGET $NUM_THREADS
