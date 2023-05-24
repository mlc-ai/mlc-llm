import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, TextGenerationPipeline, AutoModelForCausalLM

DEV = 'cuda'

def benchmark(model, inputs):
    torch.cuda.synchronize()
    print('Benchmarking ...')
    print("Input is ", inputs)
    iterations = 10
    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()
    with torch.no_grad():
        times = []
        for i in range(iterations):
            tick = time.time()
            out = model(inputs)
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
        sync()
        import numpy as np
        print('Median:', np.median(times) * 1000, 'ms')
        print("Output is ", out)


import argparse

pretrained_model_dir = "/workspace/v-leiwang3/lowbit_workspace/mlc-llm/dist/models/llama-7b-hf-l1"

model = AutoModelForCausalLM.from_pretrained(pretrained_model_dir, torch_dtype='auto')
model = model.cuda()

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
inputs = tokenizer("The capital of Canada is", return_tensors="pt").to(model.device)
inputs = inputs['input_ids']
benchmark(model, inputs)
