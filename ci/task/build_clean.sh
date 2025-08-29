#!/bin/bash
set -eo pipefail
set -x
: ${NUM_THREADS:=$(nproc)}
: ${WORKSPACE_CWD:=$(pwd)}
: ${GPU:="cpu"}

rm -rf ${WORKSPACE_CWD}/build/ \
    ${WORKSPACE_CWD}/dist/
