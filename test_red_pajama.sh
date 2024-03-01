#!/bin/bash
set -e

# QUANT=q0f16
QUANT=q4f16_1
MODEL_PATH=/opt/scratch/lesheng/models/RedPajama-INCITE-Chat-3B-v1
OUTPUT_PATH=./dist/new-red-pajama-${QUANT}/

mkdir -p $OUTPUT_PATH

# python -m mlc_chat gen_config $MODEL_PATH/config.json --quantization $QUANT --conv-template redpajama_chat -o $OUTPUT_PATH/params --tensor-parallel-shards 4

# python -m mlc_chat compile $OUTPUT_PATH/params -o $OUTPUT_PATH/red-pajama.so --debug-dump $OUTPUT_PATH/debug

# python -m mlc_chat convert_weight $MODEL_PATH --quantization $QUANT -o $OUTPUT_PATH/params

# CUDA_VISIBLE_DEVICES=4,5,6,7
python -m mlc_chat.cli.benchmark --model $OUTPUT_PATH/params --model-lib $OUTPUT_PATH/red-pajama.so --device "cuda:0" --prompt "What is the meaning of life?" --generate-length 256 # --tensor-parallel-shards 2

# python -m mlc_chat.cli.benchmark --model ./dist/new-red-pajama-q4f16_1/params --model-lib ./dist/new-red-pajama-q4f16_1/red-pajama.so --device "cuda:0" --prompt "What is the meaning of life?" --generate-length 256
