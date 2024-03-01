#!/bin/bash
set -e

MODEL_PATH=/opt/models/llama-2/llama-2-7b-chat-hf/
OUTPUT_PATH=./dist/new-llama-awq/
QUANT=q4f16_autoawq

# python -m mlc_chat gen_config $MODEL_PATH/config.json --quantization $QUANT -o $OUTPUT_PATH --conv-template llama-2

# python -m mlc_chat compile $OUTPUT_PATH -o $OUTPUT_PATH/llama.so

python -m mlc_chat convert_weight $MODEL_PATH --quantization $QUANT -o $OUTPUT_PATH --source-format awq --source ../Llama-2-7B-Chat-AWQ/model.safetensors

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m mlc_chat.cli.benchmark --model $OUTPUT_PATH --model-lib $OUTPUT_PATH/llama.so --device "cuda:0" --prompt "What is the meaning of life?" --generate-length 256
