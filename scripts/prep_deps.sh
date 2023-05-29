#!/bin/bash
# This file prepares all the necessary dependencies for the web build.
set -uo pipefail

# need rust for tokenizers
cargo version
CARGO_RESULT=$?
if [ $CARGO_RESULT -eq 0 ]; then
  echo "Cargo installed"
else
  printf "Cargo is required to compile tokenizers in MLC-LLM, do you want to install cargo (y/n)?"
  read answer
  if [ "$answer" != "${answer#[Yy]}" ] ;then 
    curl https://sh.rustup.rs -sSf | sh
  else
    echo "Failed installation: the dependency cargo not installed."
    exit 1
  fi
fi

TVM_HOME_SET="${TVM_HOME:-}"

if [[ -z ${TVM_HOME_SET} ]]; then
    if [[ ! -d "3rdparty/tvm" ]]; then
        echo "Do not find TVM_HOME env variable, cloning a version as source".
        git clone https://github.com/apache/tvm 3rdparty/tvm --branch unity --recursive
    fi
    export TVM_HOME="${TVM_HOME:-3rdparty/tvm}"
fi

export SENTENCEPIECE_JS_HOME="3rdparty/sentencepiece-js"

mkdir -p dist
