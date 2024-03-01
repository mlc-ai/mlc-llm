#!/bin/bash
set -e

# QUANT=q0f16
QUANT=q4f16_1
NUM_SHARDS=$1

MODEL_PATH=/opt/scratch/lesheng/models/gpt2-medium
OUTPUT_PATH=./dist/GPT-${QUANT}/

mkdir -p $OUTPUT_PATH

python -m mlc_chat gen_config $MODEL_PATH/config.json --quantization $QUANT --conv-template phi-2 -o $OUTPUT_PATH/params --tensor-parallel-shards $NUM_SHARDS

python -m mlc_chat compile $OUTPUT_PATH/params -o $OUTPUT_PATH/model.so --debug-dump $OUTPUT_PATH/debug

# python -m mlc_chat convert_weight $MODEL_PATH --quantization $QUANT -o $OUTPUT_PATH/params

#"What is the meaning of life?"
# CUDA_VISIBLE_DEVICES=4,5,6,7

python -m mlc_chat.cli.benchmark --model $OUTPUT_PATH/params --model-lib $OUTPUT_PATH/model.so --device "cuda:0" --prompt "a + b" --generate-length 256
