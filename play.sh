# python3 build.py --hf-path decapoda-research/llama-7b-hf --target cuda
# python3 ./tests/chat.py --model llama-7b-hf --device-name cuda
# python3 ./tests/evaluate.py --model-path llama-7b-hf --device-name cuda --debug-dump

# python3 build.py --model-path dist/models/llama-7b-hf-l1 --target cuda --use-cache=0 --quantization q4f16_1 | tee build-llama-7b-hf-l1.log
# python3 ./tests/evaluate.py --model llama-7b-hf-l1 --quantization q4f16_1 --device-name cuda --debug-dump


# python3 build.py --model-path dist/models/llama-7b-hf --target cuda --use-cache=0 --quantization q4f16_1 | tee build-llama-7b-hf.log
# python3 ./tests/evaluate.py --model llama-7b-hf --quantization q4f16_1 --device-name cuda --debug-dump

python3 quantization/build.py --model-path quantization/models/llama-7b-l1-4bit-mlc --target cuda --use-cache=0 --quantization q4f16_1 --quantized-model | tee build-llama-7b-hf-l1.log
python3 ./tests/evaluate.py --model llama-7b-hf-l1 --quantization q4f16_1 --device-name cuda --debug-dump