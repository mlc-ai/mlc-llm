#!/bin/bash
set -eo pipefail

source ~/.bashrc
micromamba activate ci-lint
export NUM_THREADS=$(nproc)
export PYTHONPATH="./python:$PYTHONPATH"

set -x

# TVM Unity is a dependency to this testing
pip install --quiet --pre -U -f https://mlc.ai/wheels mlc-ai-nightly requests

pylint --jobs $NUM_THREADS ./python/
pylint --jobs $NUM_THREADS --recursive=y ./tests/python/
