#!/bin/bash
set -eo pipefail
set -x

# Install dependency
python -m pip install --force-reinstall wheels/*.whl
python -m pip install pytest
python -m pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cu121
export LD_LIBRARY_PATH=/usr/local/cuda/compat/:$LD_LIBRARY_PATH

# run all tests that are categorized as "unittest"
# add pytestmarker = [pytest.mark.unittest] in the test file
# so they will be run here
python -m pytest -v tests/python/ -m unittest
