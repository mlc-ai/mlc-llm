#!/bin/bash
set -eo pipefail

source ~/.bashrc
micromamba activate ci-lint
export NUM_THREADS=$(nproc)
export PYTHONPATH="./python:$PYTHONPATH"

set -x

mypy ./python/mlc_chat/compiler \
	./python/mlc_chat/support \
	./tests/python/model \
	./tests/python/parameter
