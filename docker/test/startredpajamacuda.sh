docker run --gpus all --rm --network host -v ./mlcllm:/mlcllm mlc-cuda121:v0.1 --model RedPajama-INCITE-Chat-3B-v1-q4f16_1 --device cuda --host 0.0.0.0 --port 8000
