#!/bin/bash
# This file prepares all the necessary dependencies for the web build.
set -euxo pipefail

emcc --version
npm --version

TVM_SOURCE_DIR_SET="${TVM_SOURCE_DIR:-}"

git submodule update --init --recursive

CURR_DIR=`pwd`

if [[ -z "${TVM_SOURCE_DIR_SET}" ]]; then
    echo "Do not find TVM_SOURCE_DIR env variable, use 3rdparty/tvm".
    echo "Make sure you set TVM_SOURCE_DIR in your env variable to use emcc build correctly"
    export TVM_SOURCE_DIR="${TVM_SOURCE_DIR:-${CURR_DIR}/3rdparty/tvm}"
fi

# Build mlc_wasm_runtime
cd web && make
cd -

# Build tvm's web runtime
cd ${TVM_SOURCE_DIR}/web && TVM_HOME=${TVM_SOURCE_DIR} make
cd -
