#!/bin/bash
set -eo pipefail

source ~/.bashrc
micromamba activate ci-lint
export NUM_THREADS=$(nproc)
export PYTHONPATH="./python:$PYTHONPATH"

set -x

mypy --install-types --non-interactive ./python/ ./tests/python/
