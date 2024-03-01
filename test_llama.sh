#!/bin/bash
set -e

# QUANT=q0f16
QUANT=q4f16_1
# QUANT=q4f16_0
# QUANT=q4f16_ft
MODEL_PATH=/opt/models/llama-2/llama-2-7b-chat-hf
OUTPUT_PATH=./dist/new-llama-${QUANT}/
NUM_SHARDS=2

mkdir -p $OUTPUT_PATH

# python -m mlc_chat gen_config $MODEL_PATH/config.json --quantization $QUANT --conv-template llama-2 -o $OUTPUT_PATH/params --tensor-parallel-shards $NUM_SHARDS

# python -m mlc_chat compile $OUTPUT_PATH/params -o $OUTPUT_PATH/model.so --opt "O2"

# PRESHARD_NUM=$NUM_SHARDS python -m mlc_chat convert_weight $MODEL_PATH --quantization $QUANT -o $OUTPUT_PATH/params

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m mlc_chat.cli.benchmark --model $OUTPUT_PATH/params --model-lib $OUTPUT_PATH/model.so --device "cuda:0" --prompt "What is the meaning of life?" --generate-length 256 --num-shards $NUM_SHARDS
