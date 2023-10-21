#!/bin/bash
set -eo pipefail

source ~/.bashrc
micromamba activate ci-lint
export NUM_THREADS=$(nproc)
export PYTHONPATH="./python:$PYTHONPATH"

black --check --workers $NUM_THREADS ./python/
black --check --workers $NUM_THREADS ./tests/python
