#!/bin/bash
set -euxo pipefail

rustup target add aarch64-apple-ios

mkdir -p build
cd build
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

cd ..
rm -rf build/tvm_home
ln -s  ${TVM_HOME} build/tvm_home
cp ../dist/vicuna-v1-7b/float16/vicuna-v1-7b_iphone_float16.a build/lib/libmodel_iphone.a
