#!/bin/bash
set -eo pipefail

source ~/.bashrc
micromamba activate ci-lint
NUM_THREADS=$(nproc)

isort --check-only -j $NUM_THREADS --profile black ./python/
isort --check-only -j $NUM_THREADS --profile black ./tests/python/
