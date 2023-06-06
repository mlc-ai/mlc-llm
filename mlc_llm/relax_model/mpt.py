from typing import Optional, Tuple
import numpy as np

import torch

import tvm
from tvm import relax, te
from tvm.relax.testing import nn
from tvm.script import relax as R

from .mpt_config import MPTConfig, attn_config_defaults


def _cast_if_autocast_enabled(tensor):
  if torch.is_autocast_enabled():
    if tensor.device.type == 'cuda':
      dtype = torch.get_autocast_gpu_dtype()
    elif tensor.device.type == 'cpu':
      dtype = torch.get_autocast_cpu_dtype()
    else:
      raise NotImplementedError()
    return tensor.to(dtype=dtype)
  return tensor

class LPLayerNorm(torch.nn.LayerNorm):
  def __init__(self, normalized_shape, eps=1e-05, dtype=None):
    self.weight = nn.Parameter((normalized_shape,), dtype=dtype, name="low_precision_layernorm_weight")
    self.bias = nn.Parameter((normalized_shape,), dtype=dtype, name="low_precision_layernorm_bias")
    # TODO: check
    self.weight = relax.op.ones((normalized_shape,), dtype)
    self.bias = relax.op.zeros((normalized_shape,), dtype)
    self.variance_epsilon = tvm.tir.const(eps, dtype)

  def forward(self, x):
    module_device = x.device
    downcast_x = _cast_if_autocast_enabled(x)
    downcast_weight = _cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
    downcast_bias = _cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
    with torch.autocast(enabled=False, device_type=module_device.type):
      return torch.nn.functional.layer_norm(downcast_x, self.normalized_shape, downcast_weight, downcast_bias, self.eps)

NORM_CLASS_REGISTRY = {'low_precision_layernorm': LPLayerNorm}


# TODO: it is identical to Linear from llama.py
class Linear(nn.Module):
  def __init__(self, in_features, out_features, dtype: str, bias=True):
    self.in_features = in_features
    self.out_features = out_features
    self.weight = nn.Parameter(
        (out_features, in_features), dtype=dtype, name="linear_weight"
    )
    if bias:
        self.bias = nn.Parameter((out_features,), dtype=dtype, name="linear_bias")
    else:
        self.bias = None

  def forward(self, input: relax.Expr) -> relax.Var:
    return nn.emit(relax.op.linear(input, self.weight, self.bias))


class MPTMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dtype: str):
        self.down_proj = Linear(intermediate_size, hidden_size, dtype=dtype)
        self.up_proj = Linear(hidden_size, intermediate_size, dtype=dtype)

    def forward(self, x):
        return self.down_proj(relax.op.nn.gelu(self.up_proj(x)))


class MPTBlock(nn.Module):
  def __init__(self, config: MPTConfig):
    # Get values from config or defaults
    attn_config = config.attn_config if config.attn_config is not None else attn_config_defaults
    norm_type = config.norm_type if config.norm_type is not None else 'low_precision_layernorm'
    verbose = config.verbose if config.verbose is not None else 0
    # Define layer norm and attention classes
    norm_class = NORM_CLASS_REGISTRY[norm_type.lower()]
    attn_class = ATTN_CLASS_REGISTRY[attn_config['attn_type']]

    self.hidden_size = config.d_model
    # Init layers
    self.self_attn = attn_class(
        attn_impl=attn_config['attn_impl'],
        clip_qkv=attn_config['clip_qkv'],
        qk_ln=attn_config['qk_ln'],
        softmax_scale=attn_config['softmax_scale'],
        attn_pdrop=attn_config['attn_pdrop'],
        d_model=self.hidden_size,
        n_heads=config.n_heads,
        verbose=verbose,
    )
    self.mlp = MPTMLP(
        hidden_size=self.hidden_size,
        intermediate_size=config.expansion_ratio*self.hidden_size,
        dtype=config.dtype,
    )
    self.input_layernorm = norm_class(self.hidden_size)
    self.post_attention_layernorm = norm_class(self.hidden_size)

  def forward(
      self,
      hidden_states: relax.Expr,
      past_key_value: Tuple[relax.Expr],
      attn_bias: Optional[relax.Expr] = None,
      attention_mask: Optional[relax.Expr] = None,
      is_causal: bool=True,
  ) -> Tuple[relax.Expr, relax.Expr, Optional[Tuple[relax.Expr, relax.Expr]]]:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    (hidden_states, attn_weights, present_key_value) = self.self_attn(
      hidden_states,
      past_key_value=past_key_value,
      attn_bias=attn_bias,
      attention_mask=attention_mask,
      is_causal=is_causal
    )
    residual = nn.emit(residual + hidden_states)

    # Fully Connected
    hidden_states = self.post_attention_layernorm(residual)
    hidden_states = self.mlp(hidden_states)
    hidden_states = nn.emit(residual + hidden_states)

    return (hidden_states, attn_weights, present_key_value)


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