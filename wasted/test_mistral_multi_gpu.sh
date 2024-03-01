QUANT=q4f16_1
MODEL_PATH=/opt/models/llama-2/llama-2-7b-chat-hf
OUTPUT_PATH=./dist/new-llama-${QUANT}/

mkdir -p $OUTPUT_PATH

python -m mlc_chat gen_config $MODEL_PATH/config.json --quantization $QUANT --conv-template llama-2 --num-shards $NUM_SHARDS -o $OUTPUT_PATH/params

# python -m mlc_chat compile $OUTPUT_PATH/params -o $OUTPUT_PATH/llama.so

# python -m mlc_chat convert_weight $MODEL_PATH --quantization $QUANT -o $OUTPUT_PATH/params

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m mlc_chat.cli.benchmark --model $OUTPUT_PATH/params --model-lib $OUTPUT_PATH/llama.so --device "cuda:0" --prompt "What is the meaning of life?" --generate-length 256 --num-shards 1

# python -m mlc_chat compile --model $OUTPUT_PATH -o $OUTPUT_PATH/llama.so

# python -m mlc_chat convert_weight --model $MODEL_PATH --quantization q4f16_1 -o $OUTPUT_PATH

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m mlc_chat.cli.benchmark --model $OUTPUT_PATH --model-lib $OUTPUT_PATH/llama.so --device "cuda:0" --prompt "What is the meaning of life?" --generate-length 256
