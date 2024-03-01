set -e

MODEL_PATH=/opt/models/llama-2/llama-2-7b-chat-hf/

CUDA_VISIBLE_DEVICES=4,5,6,7 python build.py --model $MODEL_PATH --quantization q4f16_1 --num-shards 2 --build-model-only --max-seq-len 4096 --use-cache 0 --use-presharded-weights

# CUDA_VISIBLE_DEVICES=4,5,6,7 python build.py --model $MODEL_PATH --quantization q4f16_1 --num-shards 2 --convert-weight-only --max-seq-len 4096 --use-cache 0 --use-presharded-weights
