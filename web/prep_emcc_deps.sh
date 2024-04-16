#!/bin/bash
# This file prepares all the necessary dependencies for the web build.
set -euxo pipefail

emcc --version
npm --version

TVM_HOME_SET="${TVM_HOME:-}"

git submodule update --init --recursive

# Build mlc_wasm_runtime
cd web && make
cd -

# Build tvm's web runtime
if [[ -z ${TVM_HOME_SET} ]]; then
    echo "Do not find TVM_HOME env variable, use 3rdparty/tvm".
    echo "Make sure you set TVM_HOME in your env variable to use emcc build correctly"
    export TVM_HOME="${TVM_HOME:-3rdparty/tvm}"
fi

cd ${TVM_HOME}/web && make
cd -
