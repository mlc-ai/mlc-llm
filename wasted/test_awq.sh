MODEL_PATH=/opt/models/llama-2/llama-2-7b-chat-hf/
OUTPUT_PATH=./dist/new-llama-awq-mit/

# python -m mlc_chat compile --model llama2_7b --quantization q4f16_awq -o $OUTPUT_PATH/llama.so --context-window-size 4096

python -m mlc_chat convert_weight $MODEL_PATH --quantization q4f16_awq -o $OUTPUT_PATH --source-format awq --source dist/models/llama-2-7b-chat-w4-g128-awq.pt

# python -m mlc_chat gen_mlc_chat_config --model $MODEL_PATH/config.json --quantization q4f16_awq -o $OUTPUT_PATH --conv-template llama-2

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m mlc_chat.cli.benchmark --model $OUTPUT_PATH --model-lib $OUTPUT_PATH/llama.so --device "cuda:0" --prompt "What is the meaning of life?" --generate-length 256
