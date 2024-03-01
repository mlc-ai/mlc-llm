#!/bin/bash
set -e

# QUANT=q0f16
QUANT=q4f16_1
MODEL_PATH=/opt/scratch/lesheng/models/Mistral-7B-Instruct-v0.2
OUTPUT_PATH=./dist/mistral-${QUANT}/

mkdir -p $OUTPUT_PATH

# python -m mlc_chat gen_config $MODEL_PATH/config.json --quantization $QUANT --conv-template mistral_default -o $OUTPUT_PATH/params

# python -m mlc_chat compile $OUTPUT_PATH/params -o $OUTPUT_PATH/model.so --opt "flashinfer=0;cublas_gemm=1;cudagraph=0" --overrides "context_window_size=32768;sliding_window_size=4096;prefill_chunk_size=4096;attention_sink_size=4"

# python -m mlc_chat convert_weight $MODEL_PATH --quantization $QUANT -o $OUTPUT_PATH/params

# CUDA_VISIBLE_DEVICES=6,7 python -m mlc_chat.cli.benchmark --model $OUTPUT_PATH/params --model-lib $OUTPUT_PATH/model.so --device "cuda:0" --prompt "What is the meaning of life?" --generate-length 256

CUDA_VISIBLE_DEVICES=6,7 python -m mlc_chat.cli.benchmark \
    --model /opt/scratch/lesheng/mlc-llm/dist/tmp/model_weights/junrushao/Mistral-7B-Instruct-v0.2-q4f16_1-MLC \
    --model-lib /opt/scratch/lesheng/mlc-llm/dist/tmp/model_lib/a4e842ad2814e4e95c171a2685bdbab7.so \
    --device "cuda:0" --prompt "What is the meaning of life?" --generate-length 256
