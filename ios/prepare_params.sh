#!/bin/bash
set -euxo pipefail

rm -rf dist
mkdir -p dist
cp -rf ../dist/vicuna-v1-7b-q3f16_0/params dist/params
