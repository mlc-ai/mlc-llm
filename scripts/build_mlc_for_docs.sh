#!/bin/bash
set -euxo pipefail

mkdir -p build
cd build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j$(nproc)
cd -
