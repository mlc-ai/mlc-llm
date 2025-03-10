#!/bin/bash
set -eo pipefail
set -x
: ${NUM_THREADS:=$(nproc)}
: ${WORKSPACE_CWD:=$(pwd)}
: ${GPU:="cpu"}

pip install --force-reinstall wheels/*.whl

if [[ ${GPU} == cuda* ]]; then
    TARGET=cuda
    pip install --pre -U --no-index -f https://mlc.ai/wheels mlc-ai-nightly-cu123
    export LD_LIBRARY_PATH=/usr/local/cuda/compat/:$LD_LIBRARY_PATH
elif [[ ${GPU} == rocm* ]]; then
    TARGET=rocm
    pip install --pre -U --no-index -f https://mlc.ai/wheels mlc-ai-nightly-rocm57
elif [[ ${GPU} == metal ]]; then
    TARGET=metal
    pip install --pre -U --force-reinstall -f https://mlc.ai/wheels mlc-ai-nightly-cpu
elif [[ ${GPU} == wasm* ]]; then
    TARGET=wasm
    pip install --pre -U --no-index -f https://mlc.ai/wheels mlc-ai-nightly-cpu
    export TVM_SOURCE_DIR=$(dirname $(python -c 'import tvm; print(tvm.__file__)'))
    export TVM_HOME=${TVM_SOURCE_DIR}
    export MLC_LLM_SOURCE_DIR=$(pwd)
    cd $TVM_SOURCE_DIR/web/ && make -j${NUM_THREADS} && cd -
    cd $MLC_LLM_SOURCE_DIR/web/ && make -j${NUM_THREADS} && cd -
elif [[ ${GPU} == ios ]]; then
    TARGET=ios
    pip install --pre -U --force-reinstall -f https://mlc.ai/wheels mlc-ai-nightly-cpu
elif [[ ${GPU} == android* ]]; then
    TARGET=android
    pip install --pre -U --no-index -f https://mlc.ai/wheels mlc-ai-nightly-cpu
else
    TARGET=vulkan
    pip install --pre -U --no-index -f https://mlc.ai/wheels mlc-ai-nightly-cpu
fi

python tests/python/integration/test_model_compile.py $TARGET $NUM_THREADS
