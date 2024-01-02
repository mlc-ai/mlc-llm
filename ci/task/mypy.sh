#!/bin/bash
export NUM_THREADS=$(nproc)
export PYTHONPATH="./python:$PYTHONPATH"

set -eo pipefail
set -x

mypy --install-types --non-interactive ./python/ ./tests/python/
