from typing import Optional, Tuple
import numpy as np

import torch

import tvm
from tvm import relax, te

from .mpt_config import MPTConfig


def create_encoding_func(bb: relax.BlockBuilder, config: MPTConfig) -> None:
  pass


def get_model(args, hf_config):
  from transformers import AutoModelForCausalLM # type: ignore[import]

  model_name = args.model
  # TODO: download model and use model_path instead of args for from_pretrained
  # model_path = args.model_path
  dtype = args.quantization.model_dtype
  # Recommendation from https://huggingface.co/mosaicml/mpt-7b-instruct
  max_seq_len = args.max_seq_len if args.max_seq_len is not None else 4096  # 4096 recommended

  config.update({"max_seq_len": max_seq_len})
  config.update({"max_new_tokens": args.seq_len})

  if model_name.startswith("mpt-"):
    config = MPTConfig(**hf_config)

    bb = relax.BlockBuilder()
    create_encoding_func(bb, config)

    mod = bb.get()

    device = tvm.cpu()
    # TODO: get default mpt-7b-instruct from HF. Possibly it should be downloaded earlier
    # and use model_path instead
    hf_model = AutoModelForCausalLM.from_pretrained(
      'mosaicml/mpt-7b-instruct',
      config=config,
      torch_dtype=torch.bfloat16,
      trust_remote_code=True
    )
    # Get a list of parameters in advance, then delete the model to save memory
    # param_list = [param for _, param in hf_model.named_parameters()]
    for name, param in hf_model.named_parameters():
      print(name, param.shape)
    # Get a list of parameters in advance, then delete the model to save memory
    param_list = [param for _, param in hf_model.named_parameters()]

    for i, param in enumerate(param_list):
      # TODO: dtype? what is about mix-precision?
      param_list[i] = tvm.nd.array(
        param.detach().cpu().numpy().astype(dtype), device
      )
    del hf_model

    print(mod)
    return mod, param_list

  raise ValueError(f"Unsupported model: {model_name}")