#!/bin/bash
set -eo pipefail
set -x
: ${NUM_THREADS:=$(nproc)}
: ${WORKSPACE_CWD:=$(pwd)}
: ${GPU:="cpu"}

black --diff --check --workers $NUM_THREADS \
    ./python/ \
    ./tests/python \
    ./examples/python
