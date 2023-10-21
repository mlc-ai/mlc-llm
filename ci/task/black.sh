#!/bin/bash
set -eo pipefail

source ~/.bashrc
micromamba activate ci-lint
NUM_THREADS=$(nproc)

black --check --workers $NUM_THREADS ./python/
black --check --workers $NUM_THREADS ./tests/python
