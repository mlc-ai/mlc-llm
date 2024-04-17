#!/bin/bash
set -eo pipefail
set -x
: ${NUM_THREADS:=$(nproc)}
: ${WORKSPACE_CWD:=$(pwd)}
: ${GPU:="cpu"}
export PYTHONPATH="./python":${PYTHONPATH:-""}

# TVM Unity is a dependency to this testing
pip install --quiet --pre -U -f https://mlc.ai/wheels mlc-ai-nightly
pip install --quiet --pre -U cuda-python

pylint --jobs $NUM_THREADS ./python/
pylint --jobs $NUM_THREADS --recursive=y ./tests/python/
