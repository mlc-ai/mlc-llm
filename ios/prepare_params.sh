#!/bin/bash
set -euxo pipefail

rustup target add aarch64-apple-ios

MODEL="RedPajama-INCITE-Chat-3B-v1"
QUANTIZATION="q4f16_0"

MODEL_PARAMS="../dist/${MODEL}-${QUANTIZATION}/params/"

rm -rf dist && mkdir -p dist
cp -rf ${MODEL_PARAMS} ./dist/params
