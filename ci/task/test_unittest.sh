#!/bin/bash
set -eo pipefail
set -x

# run all tests that are categorized as "unittest"
# add pytestmarker = [pytest.mark.unittest] in the test file
# so they will be run here
python -m pip install pytest
python -m pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cu121
python -m pytest -v tests/python/ -m unittest
