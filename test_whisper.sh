#!/bin/bash
set -ex

QUANT=q0f16
# QUANT=q4f16_1
# NUM_SHARDS=$1

MODEL_PATH=/opt/scratch/lesheng/models/whisper-large-v3
OUTPUT_PATH=./dist/whisper-${QUANT}/

mkdir -p $OUTPUT_PATH

# python -m mlc_chat gen_config $MODEL_PATH/config.json --quantization $QUANT --conv-template phi-2 -o $OUTPUT_PATH/params

python -m mlc_chat compile $OUTPUT_PATH/params -o $OUTPUT_PATH/model.so --debug-dump $OUTPUT_PATH/debug --opt="cublas_gemm=0"

# python -m mlc_chat convert_weight $MODEL_PATH --quantization $QUANT -o $OUTPUT_PATH/params

# CUDA_VISIBLE_DEVICES=4,5,6,7

# python -m mlc_chat.cli.benchmark --model $OUTPUT_PATH/params --model-lib $OUTPUT_PATH/model.so --device "cuda:0" --prompt "Print a + b" --generate-length 256
