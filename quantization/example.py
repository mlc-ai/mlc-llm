import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import torch

enable_quant = True
export_mlc = False

bits = 4
# pretrained_model_dir = "dist/models/llama-7b-hf-l1"
# quantized_model_dir = f"quantization/models/llama-7b-l1-{bits}bit"

pretrained_model_dir = "dist/models/llama-7b-hf"
quantized_model_dir = f"quantization/models/llama-7b-{bits}bit"

if export_mlc:
    quantized_model_dir+= "-mlc"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
examples = [
    tokenizer(
        "mlc-llm is a universal solution that allows any language models to be deployed natively on a diverse set of hardware backends and native applications."
    )
]

quantize_config = BaseQuantizeConfig(
    bits=bits,
    group_size=-1,
    sym=False,
    desc_act=True,    
)

if enable_quant:
    # load un-quantized model, the model will always be force loaded into cpu
    model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask" 
    # with value under torch.LongTensor type.
    model.quantize(examples, export_mlc=export_mlc)

    # save quantized model
    model.save_quantized(quantized_model_dir)

    # save tokenizer
    tokenizer.save_pretrained(quantized_model_dir)

# load quantized model to the first GPU
model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", export_mlc=export_mlc)

if not export_mlc:
    # -- simple token evaluate --
    input_ids = torch.ones((1, 1), dtype=torch.long, device="cuda:0")
    outputs = model(input_ids=input_ids)
    print(f"output logits of simple token {outputs.logits.shape}:", outputs.logits)

    # 
    # or you can also use pipeline
    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)

    print(pipeline("The capital of Canada is", max_new_tokens=48)[0]["generated_text"])
    # output is: The capital of Canada is Ottawa. Canada is a country in North America. 
    # It is the second largest country in the world. Canada has a population of 35,154,139.

    print(pipeline("mlc-llm is a universal solution that allows any language models to be deployed", max_new_tokens=32)[0]["generated_text"])
    # output is: mlc-llm is a universal solution that allows any language models to be deployed on any hardware.
