#!/bin/bash
set -eo pipefail
set -x

# this scripts only triggers in CI_ENV where these environment variable are passed
if [[ -n ${MLC_CI_SETUP_DEPS:-} ]]; then
    echo "MLC_CI_SETUP_DEPS=1 start setup deps.."
    # Install dependency
    pip install --force-reinstall wheels/*.whl
    pip install "ml_dtypes>=0.5.1" --no-binary ml_dtypes
    pip install --quiet pytest
    pip install --pre -U --no-index -f https://mlc.ai/wheels mlc-ai-nightly-cu123
    export LD_LIBRARY_PATH=/usr/local/cuda/compat/:$LD_LIBRARY_PATH
fi

# run all tests that are categorized as "unittest"
# add pytestmarker = [pytest.mark.unittest] in the test file
# so they will be run here
python -m pytest -v tests/python/ -m unittest
