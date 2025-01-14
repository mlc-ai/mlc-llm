#!/bin/bash
set -eo pipefail
set -x
: ${NUM_THREADS:=$(nproc)}
: ${WORKSPACE_CWD:=$(pwd)}
: ${GPU:="cpu"}

isort --check-only -j $NUM_THREADS --profile black \
    ./python/ \
    ./tests/python/ \
    ./examples/python
