#!/bin/bash
set -euxo pipefail

rustup target add aarch64-apple-ios

MODEL="RedPajama-INCITE-Chat-3B-v1"
QUANTIZATION="q4f16_0"

MODEL_KERNEL_LIB="../dist/${MODEL}-${QUANTIZATION}/${MODEL}-${QUANTIZATION}-iphone.a"

rm -rf build/ && mkdir -p build/ && cd build/
ln -s ${TVM_HOME} ./tvm_home
cmake ../..\
  -DCMAKE_BUILD_TYPE=Release\
  -DCMAKE_SYSTEM_NAME=iOS\
  -DCMAKE_SYSTEM_VERSION=14.0\
  -DCMAKE_OSX_SYSROOT=iphoneos\
  -DCMAKE_OSX_ARCHITECTURES=arm64\
  -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0\
  -DCMAKE_BUILD_WITH_INSTALL_NAME_DIR=ON\
  -DCMAKE_SKIP_INSTALL_ALL_DEPENDENCY=ON\
  -DCMAKE_INSTALL_PREFIX=.\
  -DCMAKE_CXX_FLAGS="-O3"\
  -DMLC_LLM_INSTALL_STATIC_LIB=ON\
  -DUSE_METAL=ON
make mlc_llm_static
cmake --build . --target install --config release -j
cp ../${MODEL_KERNEL_LIB} ./lib/libmodel_iphone.a
cd ..
