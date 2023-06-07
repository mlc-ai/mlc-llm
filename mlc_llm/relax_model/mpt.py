import math                   # TODO: replace
from einops import rearrange  # TODO: replace
import warnings
from typing import Optional, Tuple
import numpy as np

import torch

import tvm
from tvm import relax, te
from tvm.relax.testing import nn
from tvm.script import relax as R

from .mpt_config import MPTConfig, attn_config_defaults
from .modules import (
    Embedding,
    LayerNorm,
    Linear,
    ModuleList,
)

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


def _reset_is_causal(num_query_tokens: int, num_key_tokens: int, original_is_causal: bool):
    if original_is_causal and num_query_tokens != num_key_tokens:
        if num_query_tokens != 1:
            raise NotImplementedError('MPT does not support query and key with different number of tokens, unless number of query tokens is 1.')
        else:
            return False
    return original_is_causal


def scaled_multihead_dot_product_attention(
    query,
    key,
    value,
    n_heads,
    past_key_value=None,
    softmax_scale=None,
    attn_bias=None,
    key_padding_mask=None,
    is_causal=False,
    needs_weights=False,
    multiquery=False
):
  q = rearrange(query, 'b s (h d) -> b h s d', h=n_heads)
  kv_n_heads = 1 if multiquery else n_heads
  k = rearrange(key, 'b s (h d) -> b h d s', h=kv_n_heads)
  v = rearrange(value, 'b s (h d) -> b h s d', h=kv_n_heads)
  if past_key_value is not None:
      if len(past_key_value) != 0:
          k = torch.cat([past_key_value[0], k], dim=3)
          v = torch.cat([past_key_value[1], v], dim=2)
      past_key_value = (k, v)
  (b, _, s_q, d) = q.shape
  s_k = k.size(-1)
  if softmax_scale is None:
      softmax_scale = 1 / math.sqrt(d)
  attn_weight = q.matmul(k) * softmax_scale
  if attn_bias is not None:
      _s_q = max(0, attn_bias.size(2) - s_q)
      _s_k = max(0, attn_bias.size(3) - s_k)
      attn_bias = attn_bias[:, :, _s_q:, _s_k:]
      if attn_bias.size(-1) != 1 and attn_bias.size(-1) != s_k or (attn_bias.size(-2) != 1 and attn_bias.size(-2) != s_q):
          raise RuntimeError(f'attn_bias (shape: {attn_bias.shape}) is expected to broadcast to shape: {attn_weight.shape}.')
      attn_weight = attn_weight + attn_bias
  min_val = torch.finfo(q.dtype).min
  if key_padding_mask is not None:
      if attn_bias is not None:
          warnings.warn('Propogating key_padding_mask to the attention module ' + 'and applying it within the attention module can cause ' + 'unneccessary computation/memory usage. Consider integrating ' + 'into attn_bias once and passing that to each attention ' + 'module instead.')
      attn_weight = attn_weight.masked_fill(~key_padding_mask.view((b, 1, 1, s_k)), min_val)
  if is_causal and (not q.size(2) == 1):
      s = max(s_q, s_k)
      causal_mask = attn_weight.new_ones(s, s, dtype=torch.float16)
      causal_mask = causal_mask.tril()
      causal_mask = causal_mask.to(torch.bool)
      causal_mask = ~causal_mask
      causal_mask = causal_mask[-s_q:, -s_k:]
      attn_weight = attn_weight.masked_fill(causal_mask.view(1, 1, s_q, s_k), min_val)
  attn_weight = torch.softmax(attn_weight, dim=-1)
  out = attn_weight.matmul(v)
  out = rearrange(out, 'b h s d -> b s (h d)')
  if needs_weights:
      return (out, attn_weight, past_key_value)
  return (out, None, past_key_value)


def check_valid_inputs(*tensors, valid_dtypes=[torch.float16, torch.bfloat16]):
    for tensor in tensors:
        if tensor.dtype not in valid_dtypes:
            raise TypeError(f'tensor.dtype={tensor.dtype!r} must be in valid_dtypes={valid_dtypes!r}.')
        if not tensor.is_cuda:
            raise TypeError(f'Inputs must be cuda tensors (tensor.is_cuda={tensor.is_cuda!r}).')


def flash_attn_fn(
    query,
    key,
    value,
    n_heads,
    past_key_value=None,
    softmax_scale=None,
    attn_bias=None,
    key_padding_mask=None,
    is_causal=False,
    needs_weights=False,
    multiquery=False
):
  try:
    from flash_attn import bert_padding, flash_attn_interface
  except:
    raise RuntimeError('Please install flash-attn==1.0.3.post0')
  check_valid_inputs(query, key, value)
  if past_key_value is not None:
    if len(past_key_value) != 0:
      key = torch.cat([past_key_value[0], key], dim=1)
      value = torch.cat([past_key_value[1], value], dim=1)
    past_key_value = (key, value)
  if attn_bias is not None:
    _s_q = max(0, attn_bias.size(2) - query.size(1))
    _s_k = max(0, attn_bias.size(3) - key.size(1))
    attn_bias = attn_bias[:, :, _s_q:, _s_k:]
  if attn_bias is not None:
    raise NotImplementedError(f'attn_bias not implemented for flash attn.')
  (batch_size, seqlen) = query.shape[:2]
  if key_padding_mask is None:
    key_padding_mask = torch.ones_like(key[:, :, 0], dtype=torch.bool)
  query_padding_mask = key_padding_mask[:, -query.size(1):]
  (query_unpad, indices_q, cu_seqlens_q, max_seqlen_q) = bert_padding.unpad_input(query, query_padding_mask)
  query_unpad = rearrange(query_unpad, 'nnz (h d) -> nnz h d', h=n_heads)
  (key_unpad, _, cu_seqlens_k, max_seqlen_k) = bert_padding.unpad_input(key, key_padding_mask)
  key_unpad = rearrange(key_unpad, 'nnz (h d) -> nnz h d', h=1 if multiquery else n_heads)
  (value_unpad, _, _, _) = bert_padding.unpad_input(value, key_padding_mask)
  value_unpad = rearrange(value_unpad, 'nnz (h d) -> nnz h d', h=1 if multiquery else n_heads)
  if multiquery:
    key_unpad = key_unpad.expand(key_unpad.size(0), n_heads, key_unpad.size(-1))
    value_unpad = value_unpad.expand(value_unpad.size(0), n_heads, value_unpad.size(-1))
  reset_is_causal = _reset_is_causal(query.size(1), key.size(1), is_causal)
  output_unpad = flash_attn_interface.flash_attn_unpadded_func(query_unpad, key_unpad, value_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, 0.0, softmax_scale=softmax_scale, causal=reset_is_causal, return_attn_probs=needs_weights)
  output = bert_padding.pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'), indices_q, batch_size, seqlen)
  return (output, None, past_key_value)


def triton_flash_attn_fn(
    query,
    key,
    value,
    n_heads,
    past_key_value=None,
    softmax_scale=None,
    attn_bias=None,
    key_padding_mask=None,
    is_causal=False,
    needs_weights=False,
    multiquery=False):
  try:
    from .flash_attn_triton import flash_attn_func
  except:
    _installed = False
    if version.parse(torch.__version__) < version.parse('2.0.0'):
      _installed = True
      try:
        from flash_attn.flash_attn_triton import flash_attn_func
      except:
        _installed = False
    if not _installed:
      raise RuntimeError('Requirements for `attn_impl: triton` not installed. Either (1) have a CUDA-compatible GPU and `pip install .[gpu]` if installing from llm-foundry source or `pip install triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python` if installing from pypi, or (2) use torch attn model.attn_config.attn_impl=torch (torch attn_impl will be slow). Note: (1) requires you have CMake and PyTorch already installed.')
  check_valid_inputs(query, key, value)
  if past_key_value is not None:
    if len(past_key_value) != 0:
      key = torch.cat([past_key_value[0], key], dim=1)
      value = torch.cat([past_key_value[1], value], dim=1)
    past_key_value = (key, value)
  if attn_bias is not None:
    _s_q = max(0, attn_bias.size(2) - query.size(1))
    _s_k = max(0, attn_bias.size(3) - key.size(1))
    attn_bias = attn_bias[:, :, _s_q:, _s_k:]
  if needs_weights:
    raise NotImplementedError(f'attn_impl: triton cannot return attn weights.')
  if key_padding_mask is not None:
    warnings.warn('Propagating key_padding_mask to the attention module ' + 'and applying it within the attention module can cause ' + 'unnecessary computation/memory usage. Consider integrating ' + 'into attn_bias once and passing that to each attention ' + 'module instead.')
    (b_size, s_k) = key_padding_mask.shape[:2]
    if attn_bias is None:
      attn_bias = query.new_zeros(b_size, 1, 1, s_k)
    attn_bias = attn_bias.masked_fill(~key_padding_mask.view((b_size, 1, 1, s_k)), torch.finfo(query.dtype).min)
  query = rearrange(query, 'b s (h d) -> b s h d', h=n_heads)
  key = rearrange(key, 'b s (h d) -> b s h d', h=1 if multiquery else n_heads)
  value = rearrange(value, 'b s (h d) -> b s h d', h=1 if multiquery else n_heads)
  if multiquery:
    key = key.expand(*key.shape[:2], n_heads, key.size(-1))
    value = value.expand(*value.shape[:2], n_heads, value.size(-1))
  reset_is_causal = _reset_is_causal(query.size(1), key.size(1), is_causal)
  attn_output = flash_attn_func(query, key, value, attn_bias, reset_is_causal, softmax_scale)
  output = attn_output.view(*attn_output.shape[:2], -1)
  return (output, None, past_key_value)


class MultiheadAttention(nn.Module):
  """Multi-head self attention.
  Using torch or triton attention implemetation enables user to also use
  additive bias.
  """

  def __init__(
      self,
      d_model: int,
      n_heads: int,
      attn_impl: str='triton',
      clip_qkv: Optional[float]=None,
      qk_ln: bool=False,
      softmax_scale: Optional[float]=None,
      low_precision_layernorm: bool=False,
      device: Optional[str]=None
  ):
    # Init fields
    self.d_model = d_model
    self.n_heads = n_heads
    self.attn_impl = attn_impl
    self.clip_qkv = clip_qkv
    self.qk_ln = qk_ln
    self.softmax_scale = softmax_scale

    if self.softmax_scale is None:
      self.softmax_scale = 1 / math.sqrt(self.d_model / self.n_heads)
    self.Wqkv = Linear(self.d_model, 3 * self.d_model, device=device)
    fuse_splits = (d_model, 2 * d_model)
    self.Wqkv._fused = (0, fuse_splits)
    if self.qk_ln:
      layernorm_class = LPLayerNorm if low_precision_layernorm else LayerNorm
      self.q_ln = layernorm_class(self.d_model, device=device)
      self.k_ln = layernorm_class(self.d_model, device=device)
    if self.attn_impl == 'flash':
      self.attn_fn = flash_attn_fn
    elif self.attn_impl == 'triton':
      # While `attn_impl: triton` can be faster than `attn_impl: flash` it uses more memory.
      # When training larger models this can trigger alloc retries which hurts performance.
      # If encountered, we recommend using `attn_impl: flash` if your model does not use `alibi` or `prefix_lm`.
      self.attn_fn = triton_flash_attn_fn
    elif self.attn_impl == 'torch':
      # Using `attn_impl: torch`. If your model does not use `alibi` or `prefix_lm` we recommend using `attn_impl: flash`
      # otherwise we recommend using `attn_impl: triton`.
      self.attn_fn = scaled_multihead_dot_product_attention
    else:
      raise ValueError(f'attn_impl={attn_impl!r} is an invalid setting.')
    self.out_proj = Linear(self.d_model, self.d_model, device=device)
    # TODO: Does field _is_residual exist?
    self.out_proj._is_residual = True

  def forward(self, x, past_key_value=None, attn_bias=None, attention_mask=None, is_causal=True, needs_weights=False):
    qkv = self.Wqkv(x)
    if self.clip_qkv:
      qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
    (query, key, value) = qkv.chunk(3, dim=2)
    key_padding_mask = attention_mask
    if self.qk_ln:
      dtype = query.dtype
      query = self.q_ln(query).to(dtype)
      key = self.k_ln(key).to(dtype)
    (context, attn_weights, past_key_value) = self.attn_fn(
        query,
        key,
        value,
        self.n_heads,
        past_key_value=past_key_value,
        softmax_scale=self.softmax_scale,
        attn_bias=attn_bias,
        key_padding_mask=key_padding_mask,
        is_causal=is_causal,
        needs_weights=needs_weights
    )
    return (self.out_proj(context), attn_weights, past_key_value)

ATTN_CLASS_REGISTRY = {'multihead_attention': MultiheadAttention}


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
    # Define layer norm and attention classes
    norm_class = NORM_CLASS_REGISTRY[norm_type.lower()]
    attn_class = ATTN_CLASS_REGISTRY[attn_config['attn_type']]

    self.hidden_size = config.d_model
    # Init layers
    self.self_attn = attn_class(
        d_model=self.hidden_size,
        n_heads=config.n_heads,
        attn_impl=attn_config['attn_impl'],
        clip_qkv=attn_config['clip_qkv'],
        qk_ln=attn_config['qk_ln'],
        softmax_scale=attn_config['softmax_scale'],
        attn_pdrop=attn_config['attn_pdrop'],
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
    config = MPTConfig(**hf_config, dtype=dtype)

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