#!/bin/bash
set -euxo pipefail

rm -rf dist
mkdir -p dist
cp -rf ../dist/models/vicuna-v1-7b/tokenizer.model dist/tokenizer.model
#cp -rf ../dist/models/stablelm-3b-v0/tokenizer.json dist/tokenizer.json
cp -rf ../dist/vicuna-v1-7b/float16/params dist/params
