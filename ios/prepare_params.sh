#!/bin/bash
set -euxo pipefail

# NOTE: this is optional, prepackage weight into app
rm -r dist
mkdir -p dist

declare -a builtin_list=(
    "RedPajama-INCITE-Chat-3B-v1-q4f16_0"
    # "vicuna-v1-7b-q3f16_0"
    # "rwkv-raven-1b5-q8f16_0"
    # "rwkv-raven-3b-q8f16_0"
    # "rwkv-raven-7b-q8f16_0"
)

for model in "${builtin_list[@]}"
do
   cp -r ../dist/$model/params dist/$model
done

