NUM_SHARDS=2
MODEL_PATH=/opt/models/llama-2/llama-2-7b-chat-hf/
OUTPUT_PATH=./dist/Mixtral-8x7B-Instruct-v0.1-q0f16/params

# python -m mlc_chat gen_config --model $MODEL_PATH/config.json --quantization q4f16_1 --conv-template llama-2 --context-window-size 4096 --num-shards $NUM_SHARDS -o $OUTPUT_PATH

# python -m mlc_chat compile --model $OUTPUT_PATH -o $OUTPUT_PATH/llama.so

# python -m mlc_chat convert_weight --model $MODEL_PATH --quantization q4f16_1 -o $OUTPUT_PATH

CUDA_VISIBLE_DEVICES=6,7 python -m mlc_chat.cli.benchmark --model ./dist/Mixtral-8x7B-Instruct-v0.1-q0f16/params --model-lib ./dist/Mixtral-8x7B-Instruct-v0.1-q0f16/Mixtral-8x7B-Instruct-v0.1-q0f16-cuda.so --device "cuda:0" --prompt "What is the meaning of life?" --generate-length 256 --num-shards 2
