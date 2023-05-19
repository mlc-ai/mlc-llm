import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import torch
import numpy as np

def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)
    
    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({'input_ids':inp,'attention_mask': attention_mask})
    return traindataset, testenc

bits = 4
pretrained_model_dir = "/workspace/v-leiwang3/lowbit_workspace/mlc-llm/dist/models/llama-7b-hf"
quantized_model_dir = f"quantization/models/llama-7b-{bits}bit"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)

quantize_config = BaseQuantizeConfig(
    bits=bits,  # quantize model to 3-bit
    # group_size=-1,
    sym=False,
    desc_act=True,    
)

# load un-quantized model, the model will always be force loaded into cpu
model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

# 
traindataset,testenc = get_wikitext2(128, 0, 2048, pretrained_model_dir)
# quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask" 
# with value under torch.LongTensor type.
model.quantize(traindataset, export_mlc=True)

# save quantized model
model.save_quantized(quantized_model_dir)

# save tokenizer
tokenizer.save_pretrained(quantized_model_dir)

# load quantized model to the first GPU
model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0")
for name, tensor in model.state_dict().items():
    print("name: ", name, "tensor: ", tensor.shape, tensor.dtype)
    
inputs = tokenizer("MLC LLM is", return_tensors="pt")
if 'token_type_ids' in inputs:
    del inputs['token_type_ids']

print(inputs)
outputs = model(**inputs, labels=inputs["input_ids"])
print(f"output logits {outputs.logits.shape}:", outputs.logits)


# -- simple token evaluate --
input_ids = torch.ones((1, 1), dtype=torch.long, device="cuda:0")
outputs = model(input_ids=input_ids)
print(f"output logits {outputs.logits.shape}:", outputs.logits)

# 
# or you can also use pipeline
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
print(pipeline("The capital of Canada is")[0]["generated_text"])
