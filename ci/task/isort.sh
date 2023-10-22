#!/bin/bash
set -eo pipefail

source ~/.bashrc
micromamba activate ci-lint
export NUM_THREADS=$(nproc)
export PYTHONPATH="./python:$PYTHONPATH"

set -x

isort --check-only -j $NUM_THREADS --profile black \
	./python/ \
	./tests/python/
