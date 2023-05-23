## Quantization

This folder contains the GPTQ-based quantization tools for `mlc-llm`. Currently supports: bloom, gpt2, gpt_neox, gptj, llama, moss and opt; There're two ways to quantize the model:

- **Quantize from given prompts**

    we provide a simple script to quantize a given model from a given prompt. The script is located in `example.py`:

    ```python

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

    ```

    - The quantized model will be saved in `quantized_model_dir` and the tokenizer will be saved in `quantized_model_dir` as well.
    - if `enable_quant` is set to `True`, the model will be quantized and saved in `quantized_model_dir`. Otherwise, the model will be loaded from `quantized_model_dir`.
    - if `export-mlc` is set to `True`, the model will be exported to MLC format which suitable for `mlc-llm` compute kernels. Otherwise, the model will be exported to other format which suitable for naive kernel implementation inference for correctness evaluation.

- **Quantize from given datasets**: Please refer to [quant_wikitext2.py](quant_wikitext2.py) 

### Buil mlc-llm executable and evaluate 

- build
    ```python
    python3 quantization/build.py --model-path quantization/models/llama-7b-l1-4bit-mlc --target cuda --use-cache=0 --quantization q4f16_1 --quantized-model
    ```

- evaluate
    ```python
    python3 ./tests/evaluate.py --model llama-7b-hf-l1 --device-name cuda --debug-dump --local-id llama-7b-hf-l1-q4f16_1
    ```

###  Reference

- [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ): An easy-to-use LLMs quantization package with user-friendly apis, based on GPTQ.