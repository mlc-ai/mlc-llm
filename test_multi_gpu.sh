#!/bin/bash
set -e

NUM_SHARDS=2
MODEL_PATH=/opt/models/llama-2/llama-2-7b-chat-hf/
OUTPUT_PATH=./dist/new-llama-2-GPU/

# python -m mlc_chat gen_config $MODEL_PATH/config.json --quantization q4f16_1 --conv-template llama-2 --context-window-size 4096 -o $OUTPUT_PATH --tensor-parallel-shards $NUM_SHARDS

# python -m mlc_chat compile $OUTPUT_PATH -o $OUTPUT_PATH/llama.so

python -m mlc_chat convert_weight $MODEL_PATH --quantization q4f16_1 -o $OUTPUT_PATH

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m mlc_chat.cli.benchmark --model $OUTPUT_PATH --model-lib $OUTPUT_PATH/llama.so --device "cuda:0" --prompt "What is the meaning of life?" --generate-length 256
