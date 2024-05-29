#!/bin/bash
set -eo pipefail
set -x

# run all tests that are categorized as "unittest"
# add pytestmarker = [pytest.mark.unittest] in the test file
# so they will be run here
python -m pytest -v tests/python/ -m unittest
