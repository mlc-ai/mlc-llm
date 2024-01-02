#!/bin/bash
export NUM_THREADS=$(nproc)
export PYTHONPATH="./python:$PYTHONPATH"

set -eo pipefail
set -x

isort --check-only -j $NUM_THREADS --profile black \
	./python/ \
	./tests/python/
