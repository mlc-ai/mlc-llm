#!/bin/bash
export NUM_THREADS=$(nproc)
export PYTHONPATH="./python:$PYTHONPATH"

set -eo pipefail
set -x

black --check --workers $NUM_THREADS \
	./python/ \
	./tests/python
