MODEL_PATH=/opt/scratch/lesheng/models/RedPajama-INCITE-Chat-3B-v1

CUDA_VISIBLE_DEVICES=4,5,6,7 python build.py --model $MODEL_PATH --quantization q4f16_1 --use-cache 0 --debug-dump # --build-model-only
