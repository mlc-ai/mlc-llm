#!/bin/bash
set -euxo pipefail

# NOTE: this is optional, prepackage weight into app
rm -rf dist
mkdir -p dist

declare -a builtin_list=(
        "Mistral-7B-Instruct-v0.2-q3f16_1"
        #"OpenHermes-2.5-Mistral-7B-q3f16_1"
	# "Llama-2-7b-chat-hf-q3f16_1"
	# "RedPajama-INCITE-Chat-3B-v1-q4f16_1"
	# "vicuna-v1-7b-q3f16_0"
	# "rwkv-raven-1b5-q8f16_0"
	# "rwkv-raven-3b-q8f16_0"
	# "rwkv-raven-7b-q8f16_0"
)

for model in "${builtin_list[@]}"; do
	if [ -d ../dist/$model/params ]; then
		cp -r ../dist/$model/params dist/$model
	elif [ -d ../dist/prebuilt/$model ]; then
		cp -r ../dist/prebuilt/$model dist/$model
	elif [ -d ../dist/prebuilt/mlc-chat-$model ]; then
		cp -r ../dist/prebuilt/mlc-chat-$model dist/$model
	else
		echo "Cannot find prebuilt weights for " $model
		exit 1
	fi
done
