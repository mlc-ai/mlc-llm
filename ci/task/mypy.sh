#!/bin/bash
: ${NUM_THREADS:=$(nproc)}
: ${WORKSPACE_CWD:=$(pwd)}
: ${GPU:="cpu"}

mypy --install-types --non-interactive ./python/ ./tests/python/ ./examples/python/
