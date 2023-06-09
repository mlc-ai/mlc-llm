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

export SENTENCEPIECE_JS_HOME="3rdparty/sentencepiece-js"

mkdir -p dist
