import math
import warnings
from typing import Optional, Tuple, List, Dict
import numpy as np

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
    named_parameters,
)


def _cast_if_autocast_enabled(tensor):
  # # TODO: how to check device?
  # if tensor.device.type == 'cuda':
  #   dtype = "float16"
  # elif tensor.device.type == 'cpu':
  #   dtype = "bfloat16"
  # else:
  #   raise NotImplementedError()
  dtype = "float32" # TODO: temporal workaround
  return nn.emit(tvm.tir.Cast(dtype, tensor))

# Low-precision layer norm for mpt-7b-instruct, where are no biases expected
class LPLayerNormWOBias(nn.Module):
  def __init__(self, normalized_shape, dtype, eps=1e-05):
    self.weight = nn.Parameter((normalized_shape,), dtype=dtype, name="low_precision_layernorm_weight")
    # TODO: check default filling of weights
    self.weight = relax.op.ones((normalized_shape,), dtype)
    self.bias = relax.op.zeros((normalized_shape,), dtype)
    self.eps = relax.const(eps, dtype)

  def forward(self, x):
    downcast_x = _cast_if_autocast_enabled(x)
    downcast_weight = _cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
    downcast_bias = _cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
    return nn.emit(relax.op.nn.layer_norm(downcast_x, downcast_weight, downcast_bias, axes=-1, epsilon=self.eps))

NORM_CLASS_REGISTRY = {'low_precision_layernorm': LPLayerNormWOBias}


def _reset_is_causal(num_query_tokens: int, num_key_tokens: int, original_is_causal: bool):
  if original_is_causal and num_query_tokens != num_key_tokens:
    if num_query_tokens != 1:
      raise NotImplementedError('MPT does not support query and key with different number of tokens, unless number of query tokens is 1.')
    else:
      return False
  return original_is_causal


def reshape_and_permute(hidden_states: relax.Expr, n_heads: int, d_model: int, indeces: List[int] = [0, 2, 1, 3]):
  '''
  Transform shape of input: b s (h d) -> b h d s
  '''
  batch_size, seqlen, _ = hidden_states.struct_info.shape
  inter = nn.emit(relax.op.reshape(
      hidden_states,
      (batch_size, seqlen, n_heads, d_model),
  ))
  return nn.emit(relax.op.permute_dims(inter, indeces))


def reverse_reshape_and_permute(hidden_states: relax.Expr):
  '''
  Transform shape of input: b h s d -> b s (h d)
  '''
  batch_size, n_heads, seqlen, d_model = hidden_states.struct_info.shape
  inter = nn.emit(relax.op.permute_dims(hidden_states, [0, 2, 1, 3]))
  return nn.emit(relax.op.reshape(
      inter,
      (batch_size, seqlen, n_heads*d_model),
  ))


######################### FLASH ATTENTION IMPLEMENTATION TYPE TORCH (BEGIN) ##########################

def scaled_multihead_dot_product_attention(
    query,
    key,
    value,
    n_heads,
    d_model,
    past_key_value=None,
    softmax_scale=None,
    attn_bias=None,
    key_padding_mask=None,
    is_causal=False,
    needs_weights=False,
    multiquery=False
):
  q = reshape_and_permute(query, n_heads, d_model)
  kv_n_heads = 1 if multiquery else n_heads
  k = reshape_and_permute(key, kv_n_heads, d_model, [0, 2, 3, 1])
  v = reshape_and_permute(value, kv_n_heads, d_model)
  if past_key_value is not None:
    if len(past_key_value) != 0:
      k = nn.emit(relax.op.concat([past_key_value[0], k], axis=3))
      v = nn.emit(relax.op.concat([past_key_value[1], v], axis=2))
    past_key_value = (k, v)
  (b, _, s_q, d) = q.struct_info.shape
  s_k = k.struct_info.shape[-1]
  if softmax_scale is None:
      softmax_scale = 1 / math.sqrt(d)
  attn_weight = nn.emit(relax.op.matmul(q, k) * softmax_scale)
  if attn_bias is not None:
    _s_q = relax.op.maximum(relax.const(0), attn_bias.struct_info.shape[2] - s_q)
    _s_k = relax.op.maximum(relax.const(0), attn_bias.struct_info.shape[3] - s_k)
    # slicing attn_bias[:, :, _s_q:, _s_k:]
    s_q_end, s_k_end = attn_bias.struct_info.shape[-2:]
    attn_bias = nn.emit(relax.op.strided_slice(attn_bias, [2, 3], [_s_q, _s_k], [s_q_end, s_k_end]))
    if (attn_bias.struct_info.shape[-1] != 1 and
        attn_bias.struct_info.shape[-1] != s_k or # dynamic condition?
        (attn_bias.struct_info.shape[-2] != 1 and
          attn_bias.struct_info.shape[-2] != s_q)): # dynamic condition?
      raise RuntimeError(f'attn_bias (shape: {attn_bias.struct_info.shape}) is expected to broadcast to shape: {attn_weight.struct_info.shape}.')
    attn_weight = attn_weight + attn_bias
  min_val = tvm.tir.min_value(q.struct_info.dtype)
  if key_padding_mask is not None:
    if attn_bias is not None:
      warnings.warn('Propogating key_padding_mask to the attention module ' + 'and applying it within the attention module can cause ' + 'unneccessary computation/memory usage. Consider integrating ' + 'into attn_bias once and passing that to each attention ' + 'module instead.')
    key_mask = nn.emit(tvm.tir.bitwise_not(relax.op.reshape(key_padding_mask, (b, 1, 1, s_k))))
    attn_weight = nn.emit(relax.op.masked_fill(attn_weight, key_mask, min_val))
  if is_causal and (not q.struct_info.shape[2] == 1):
      s = relax.op.maximum(s_q, s_k)
      causal_mask = nn.emit(relax.op.ones((s, s,), dtype="float16"))
      causal_mask = nn.emit(relax.op.tril(causal_mask))
      causal_mask = tvm.tir.Cast("bool", causal_mask)
      causal_mask = tvm.tir.bitwise_not(causal_mask)
      # slicing causal_mask[-s_q:, -s_k:]
      s_q_end, s_k_end = causal_mask.struct_info.shape
      causal_mask = nn.emit(relax.op.strided_slice(causal_mask, [0, 1], [s_q_end - s_q, s_k_end - s_k], [s_q_end, s_k_end]))
      causal_mask = nn.emit(relax.op.reshape(causal_mask, (1, 1, s_q, s_k)))
      attn_weight = nn.emit(relax.op.masked_fill(attn_weight, causal_mask, min_val))
  attn_weight = nn.emit(relax.op.nn.softmax(attn_weight))
  out = nn.emit(relax.op.matmul(attn_weight, v))
  out = reverse_reshape_and_permute(out)
  if needs_weights:
    return (out, attn_weight, past_key_value)
  return (out, None, past_key_value)

######################### FLASH ATTENTION IMPLEMENTATION TYPE TORCH (END) ##########################


def check_valid_inputs(*tensors, valid_dtypes=["float16", "bfloat16"]):
  for tensor in tensors:
    if tensor.struct_info.dtype not in valid_dtypes:
      raise TypeError(f'tensor.dtype={tensor.struct_info.dtype!r} must be in valid_dtypes={valid_dtypes!r}.')
    # TODO: check on relax that CUDA is used
    # if not tensor.is_cuda:
    #   raise TypeError(f'Inputs must be cuda tensors (tensor.is_cuda={tensor.is_cuda!r}).')


######################### FLASH ATTENTION IMPLEMENTATION TYPE FLASH (BEGIN) ##########################

# class IndexFirstAxis(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, input, indices):
#         ctx.save_for_backward(indices)
#         assert input.ndim >= 2
#         ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
#         second_dim = other_shape.numel()
#         # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
#         # return input[indices]
#         return torch.gather(rearrange(input, 'b ... -> b (...)'), 0,
#                             repeat(indices, 'z -> z d', d=second_dim)).reshape(-1, *other_shape)


# index_first_axis = IndexFirstAxis.apply

# import torch
# class IndexPutFirstAxis(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, values, indices, first_axis_dim):
#         ctx.save_for_backward(indices)
#         assert indices.ndim == 1
#         assert values.ndim >= 2
#         output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device,
#                              dtype=values.dtype)
#         # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
#         output[indices] = values
#         # output.scatter_(0, repeat(indices, 'z -> z d', d=values.shape[1]), values)
#         return output


# index_put_first_axis = IndexPutFirstAxis.apply


# def bert_padding_unpad_input(hidden_states, attention_mask):
#   seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
#   indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
#   max_seqlen_in_batch = seqlens_in_batch.max().item()
#   cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
#   # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
#   # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
#   # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
#   # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
#   # so we write custom forward and backward to make it a bit faster.
#   return (index_first_axis(rearrange(hidden_states, 'b s ... -> (b s) ...'), indices), indices,
#           cu_seqlens, max_seqlen_in_batch)


# def bert_padding_pad_input(hidden_states, indices, batch, seqlen):
#     """
#     Arguments:
#         hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
#         indices: (total_nnz)
#     Return:
#         hidden_states: (batch, seqlen, ...)
#     """
#     dim = hidden_states.shape[-1]
#     # output = torch.zeros((batch * seqlen), dim, device=hidden_states.device, dtype=hidden_states.dtype)
#     # output[indices] = hidden_states
#     output = index_put_first_axis(hidden_states, indices, batch * seqlen)
#     return rearrange(output, '(b s) ... -> b s ...', b=batch)

# import flash_attn_cuda
# def _flash_attn_forward(q, k, v, out, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
#                         dropout_p, softmax_scale, causal, return_softmax, num_splits=0,
#                         generator=None):
#     """
#     num_splits: how much to parallelize over the seqlen_q dimension. num_splits=0 means
#     it will be set by an internal heuristic. We're exposing num_splits mostly for benchmarking.
#     Don't change it unless you know what you're doing.
#     """
#     softmax_lse, rng_state, *rest = flash_attn_cuda.fwd(
#         q, k, v, out, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
#         softmax_scale, False, causal, return_softmax, num_splits, generator
#     )
#     # if out.isnan().any() or softmax_lse.isnan().any():
#     #     breakpoint()
#     S_dmask = rest[0] if return_softmax else None
#     return out, softmax_lse, rng_state, S_dmask


# class FlashAttnFunc(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
#                 softmax_scale, causal, return_softmax, deterministic):
#         if softmax_scale is None:
#             softmax_scale = q.shape[-1] ** (-0.5)
#         out, softmax_lse, rng_state, S_dmask = _flash_attn_forward(
#             q, k, v, torch.empty_like(q), cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
#             dropout_p, softmax_scale, causal=causal, return_softmax=return_softmax
#         )
#         ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state)
#         ctx.dropout_p = dropout_p
#         ctx.max_seqlen_q = max_seqlen_q
#         ctx.max_seqlen_k = max_seqlen_k
#         ctx.softmax_scale = softmax_scale
#         ctx.causal = causal
#         ctx.deterministic = deterministic
#         return out if not return_softmax else (out, softmax_lse, S_dmask)


# def flash_attn_interface_flash_attn_unpadded_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
#     dropout_p, softmax_scale=None, causal=False, return_attn_probs=False,
#     deterministic=False
# ):
#   return FlashAttnFunc.apply(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
#                               dropout_p, softmax_scale, causal, return_attn_probs, deterministic)


# def flash_attn_fn(
#     query,
#     key,
#     value,
#     n_heads,
#     d_model,
#     past_key_value=None,
#     softmax_scale=None,
#     attn_bias=None,
#     key_padding_mask=None,
#     is_causal=False,
#     needs_weights=False,
#     multiquery=False
# ):
#   check_valid_inputs(query, key, value)
#   if past_key_value is not None:
#     if len(past_key_value) != 0:
#       key = nn.emit(relax.op.concat([past_key_value[0], key], axis=1))
#       value = nn.emit(relax.op.concat([past_key_value[1], value], axis=1))
#     past_key_value = (key, value)
#   if attn_bias is not None:
#     _s_q = relax.op.maximum(relax.const(0), attn_bias.struct_info.shape[2] - query.struct_info.shape[1])
#     _s_k = relax.op.maximum(relax.const(0), attn_bias.struct_info.shape[3] - key.struct_info.shape[1])
#     # slicing attn_bias[:, :, _s_q:, _s_k:]
#     s_q_end, s_k_end = attn_bias.struct_info.shape[-2:]
#     attn_bias = nn.emit(relax.op.strided_slice(attn_bias, [2, 3], [_s_q, _s_k], [s_q_end, s_k_end]))
#   if attn_bias is not None:
#     raise NotImplementedError(f'attn_bias not implemented for flash attn.')
#   batch_size, seqlen = query.struct_info.shape[:2]
#   if key_padding_mask is None:
#     key_shape = key.struct_info.shape[:2]
#     key_padding_mask = nn.emit(relax.op.ones(Tuple(*key_shape, 1), dtype="bool"))
#   # slicing key_padding_mask[:, -query.struct_info.shape[1]:]
#   dim1_length = key_padding_mask.struct_info.shape[1]
#   query_padding_mask = nn.emit(relax.op.strided_slice(key_padding_mask, [1], [dim1_length - seqlen], [dim1_length]))
#   (query_unpad, indices_q, cu_seqlens_q, max_seqlen_q) = bert_padding_unpad_input(query, query_padding_mask)

#   qnnz, _, _ = query_unpad.struct_info.shape
#   query_unpad = nn.emit(relax.op.reshape(
#       query_unpad,
#       (qnnz, n_heads, d_model),
#   ))  # (nnz, (h d)) -> (nnz, h, d)

#   kv_nnz, _, _ = key_unpad.struct_info.shape
#   kv_n_heads = 1 if multiquery else n_heads
#   (key_unpad, _, cu_seqlens_k, max_seqlen_k) = bert_padding_unpad_input(key, key_padding_mask)
#   key_unpad = nn.emit(relax.op.reshape(
#       key_unpad,
#       (kv_nnz, kv_n_heads, d_model),
#   ))  # (nnz, (h d)) -> (nnz, h, d)
#   (value_unpad, _, _, _) = bert_padding_unpad_input(value, key_padding_mask)
#   value_unpad = nn.emit(relax.op.reshape(
#       value_unpad,
#       (kv_nnz, kv_n_heads, d_model),
#   ))  # (nnz, (h d)) -> (nnz, h, d)

#   if multiquery:
#     key_unpad = relax.op.broadcast_to(key_unpad, (kv_nnz, n_heads, d_model))
#     value_unpad = relax.op.broadcast_to(value_unpad, (kv_nnz, n_heads, d_model))
#   reset_is_causal = _reset_is_causal(query.struct_info.shape[1], key.struct_info.shape[1], is_causal)
#   output_unpad = flash_attn_interface_flash_attn_unpadded_func(query_unpad, key_unpad, value_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, 0.0, softmax_scale=softmax_scale, causal=reset_is_causal, return_attn_probs=needs_weights)
#   nnz, _, _ = output_unpad.struct_info.shape
#   output_unpad = nn.emit(relax.op.reshape(
#       output_unpad,
#       (nnz, n_heads*d_model),
#   ))  # (nnz, h, d)) -> (nnz, (h d))
#   output = bert_padding_pad_input(output_unpad, indices_q, batch_size, seqlen)
#   return (output, None, past_key_value)

######################### FLASH ATTENTION IMPLEMENTATION TYPE FLASH (END) ##########################


######################### FLASH ATTENTION IMPLEMENTATION TYPE TRITON (BEGIN) ##########################

# def triton_flash_attn_fn(
#     query,
#     key,
#     value,
#     n_heads,
#     d_model,
#     past_key_value=None,
#     softmax_scale=None,
#     attn_bias=None,
#     key_padding_mask=None,
#     is_causal=False,
#     needs_weights=False,
#     multiquery=False):
#   try:
#     from .flash_attn_triton import flash_attn_func
#   except:
#     _installed = False
#     if version.parse(torch.__version__) < version.parse('2.0.0'):
#       _installed = True
#       try:
#         from flash_attn.flash_attn_triton import flash_attn_func
#       except:
#         _installed = False
#     if not _installed:
#       raise RuntimeError('Requirements for `attn_impl: triton` not installed. Either (1) have a CUDA-compatible GPU and `pip install .[gpu]` if installing from llm-foundry source or `pip install triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python` if installing from pypi, or (2) use torch attn model.attn_config.attn_impl=torch (torch attn_impl will be slow). Note: (1) requires you have CMake and PyTorch already installed.')
#   check_valid_inputs(query, key, value)
#   if past_key_value is not None:
#     if len(past_key_value) != 0:
#       key = nn.emit(relax.op.concat([past_key_value[0], key], axis=1))
#       value = nn.emit(relax.op.concat([past_key_value[1], value], axis=1))
#     past_key_value = (key, value)
#   if attn_bias is not None:
#     _s_q = relax.op.maximum(relax.const(0), attn_bias.struct_info.shape[2] - query.struct_info.shape[1])
#     _s_k = relax.op.maximum(relax.const(0), attn_bias.struct_info.shape[3] - key.struct_info.shape[1])
#     # slicing attn_bias[:, :, _s_q:, _s_k:]
#     s_q_end, s_k_end = attn_bias.struct_info.shape[-2:]
#     attn_bias = nn.emit(relax.op.strided_slice(attn_bias, [2, 3], [_s_q, _s_k], [s_q_end, s_k_end]))
#   if needs_weights:
#     raise NotImplementedError(f'attn_impl: triton cannot return attn weights.')
#   if key_padding_mask is not None:
#     warnings.warn('Propagating key_padding_mask to the attention module ' + 'and applying it within the attention module can cause ' + 'unnecessary computation/memory usage. Consider integrating ' + 'into attn_bias once and passing that to each attention ' + 'module instead.')
#     (b_size, s_k) = key_padding_mask.struct_info.shape[:2]
#     if attn_bias is None:
#       attn_bias = nn.emit(relax.op.zeros((b_size, 1, 1, s_k), dtype=query.struct_info.dtype))
#     key_mask = nn.emit(tvm.tir.bitwise_not(relax.op.reshape(key_padding_mask, (b_size, 1, 1, s_k))))
#     attn_bias = nn.emit(relax.op.masked_fill(attn_bias, key_mask, tvm.tir.min_value(query.struct_info.dtype)))

#   batch_size, seq_len, _ = query.struct_info.shape
#   query = nn.emit(relax.op.reshape(
#       query,
#       (batch_size, seq_len, n_heads, d_model),
#   ))  # b s (h d) -> b s h d

#   batch_size, seq_len, _ = key.struct_info.shape
#   kv_n_heads = 1 if multiquery else n_heads
#   key = nn.emit(relax.op.reshape(
#       key,
#       (batch_size, seq_len, kv_n_heads, d_model),
#   ))  # b s (h d) -> b s h d
#   value = nn.emit(relax.op.reshape(
#       value,
#       (batch_size, seq_len, kv_n_heads, d_model),
#   ))  # b s (h d) -> b s h d
#   if multiquery:
#     key = relax.op.broadcast_to(key, (batch_size, seq_len, n_heads, d_model))
#     value = relax.op.broadcast_to(value, (batch_size, seq_len, n_heads, d_model))
#   reset_is_causal = _reset_is_causal(query.struct_info.shape[1], key.struct_info.shape[1], is_causal)
#   attn_output = flash_attn_func(query, key, value, attn_bias, reset_is_causal, softmax_scale)
#   batch_size, seq_len, _, _ = attn_output.struct_info.shape
#   output = nn.emit(relax.op.reshape(
#       attn_output,
#       (batch_size, seq_len, n_heads*d_model),
#   ))  # (b, s, h, d)) -> (b, s, (h d))
#   return (output, None, past_key_value)

######################### FLASH ATTENTION IMPLEMENTATION TYPE TRITON (END) ##########################


class MultiheadAttention(nn.Module):
  """Multi-head self attention.
  Using torch or triton attention implemetation enables user to also use
  additive bias.
  """

  def __init__(
      self,
      d_model: int,
      n_heads: int,
      dtype: str,
      attn_impl: str='triton',
      clip_qkv: Optional[float]=None,
      qk_ln: bool=False,
      softmax_scale: Optional[float]=None,
      low_precision_layernorm: bool=False
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
    self.Wqkv = Linear(self.d_model, 3 * self.d_model, dtype)
    fuse_splits = (d_model, 2 * d_model)
    self.Wqkv._fused = (0, fuse_splits)
    if self.qk_ln:
      layernorm_class = LPLayerNormWOBias if low_precision_layernorm else LayerNorm
      self.q_ln = layernorm_class(self.d_model, dtype)
      self.k_ln = layernorm_class(self.d_model, dtype)
    if self.attn_impl == 'flash':
      raise NotImplemented("Flash type of flash attention has not been implemented yet")
      # self.attn_fn = flash_attn_fn
    elif self.attn_impl == 'triton':
      # While `attn_impl: triton` can be faster than `attn_impl: flash` it uses more memory.
      # When training larger models this can trigger alloc retries which hurts performance.
      # If encountered, we recommend using `attn_impl: flash` if your model does not use `alibi` or `prefix_lm`.
      raise NotImplemented("Triton type of flash attention has not been implemented yet")
      # self.attn_fn = triton_flash_attn_fn
    elif self.attn_impl == 'torch':
      # Using `attn_impl: torch`. If your model does not use `alibi` or `prefix_lm` we recommend using `attn_impl: flash`
      # otherwise we recommend using `attn_impl: triton`.
      self.attn_fn = scaled_multihead_dot_product_attention
    else:
      raise ValueError(f'attn_impl={attn_impl!r} is an invalid setting.')
    self.out_proj = Linear(self.d_model, self.d_model, dtype)
    # TODO: Does field _is_residual exist?
    # self.out_proj._is_residual = True

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
        self.d_model,
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
        dtype=config.dtype,
        attn_impl=attn_config['attn_impl'],
        clip_qkv=attn_config['clip_qkv'],
        qk_ln=attn_config['qk_ln'],
        softmax_scale=attn_config['softmax_scale'],
    )
    self.mlp = MPTMLP(
        hidden_size=self.hidden_size,
        intermediate_size=config.expansion_ratio*self.hidden_size,
        dtype=config.dtype,
    )
    self.input_layernorm = norm_class(self.hidden_size, config.dtype)
    self.post_attention_layernorm = norm_class(self.hidden_size, config.dtype)

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


def attn_bias_shape(attn_impl, n_heads, seq_len, alibi, prefix_lm, causal, use_sequence_id):
  if attn_impl == 'flash':
    return None
  elif attn_impl in ['torch', 'triton']:
    if alibi:
      if (prefix_lm or not causal) or use_sequence_id:
        return (1, n_heads, seq_len, seq_len)
      return (1, n_heads, 1, seq_len)
    elif prefix_lm or use_sequence_id:
      return (1, 1, seq_len, seq_len)
    return None
  else:
    raise ValueError(f'attn_impl={attn_impl!r} is an invalid setting.')


def gen_slopes(n_heads, alibi_bias_max=8):
    _n_heads = 2 ** math.ceil(math.log2(n_heads))
    m = nn.emit(relax.op.arange(1, _n_heads + 1, dtype="float32"))
    m = nn.emit(m * (alibi_bias_max / _n_heads))
    slopes = 1.0 / math.pow(2, m)
    if _n_heads != n_heads:
      slopes_len = slopes.struct_info.shape[0]
      slopes = nn.emit(relax.op.strided_slice(
          relax.op.concat(
            [relax.op.strided_slice(slopes, [0], [relax.const(1)], [slopes_len], [relax.const(2)]), # [1::2]
             relax.op.strided_slice(slopes, [0], [relax.const(0)], [slopes_len], [relax.const(2)])] # [::2]
          ), [0], [relax.const(0)], [relax.const(n_heads)]) # slicing [:n_heads]
      )
    return nn.emit(relax.op.reshape(slopes, (1, n_heads, 1, 1)))


def build_alibi_bias(n_heads, seq_len, full=False, alibi_bias_max=8, dtype=None):
  alibi_bias = nn.emit(relax.op.reshape(relax.op.arange(1 - seq_len, 1, dtype="int32"), (1, 1, 1, seq_len)))
  if full:
    alibi_bias = nn.emit(alibi_bias - relax.op.reshape(relax.op.arange(1 - seq_len, 1, dtype="int32"), (1, 1, seq_len, 1)))
    alibi_bias = nn.emit(relax.op.negative(relax.op.abs(alibi_bias)))
  slopes = gen_slopes(n_heads, alibi_bias_max)
  alibi_bias = nn.emit(alibi_bias * slopes)
  if dtype is not None:
    alibi_bias = nn.emit(tvm.tir.Cast(dtype, alibi_bias))
  return alibi_bias


def build_attn_bias(attn_impl, attn_bias, n_heads, seq_len, causal=False, alibi=False, alibi_bias_max=8):
  if attn_impl == 'flash':
    return None
  elif attn_impl in ['torch', 'triton']:
    if alibi:
      attn_bias = nn.emit(relax.op.add(attn_bias, build_alibi_bias(
        n_heads, seq_len, full=not causal, alibi_bias_max=alibi_bias_max, dtype=attn_bias.struct_info.dtype
      )))
    return attn_bias
  else:
    raise ValueError(f'attn_impl={attn_impl!r} is an invalid setting.')


class MPTModel(nn.Module):
  def __init__(self, config: MPTConfig):
    config._validate_config()
    # Init fields from config
    self.attn_impl = config.attn_config['attn_impl']
    self.prefix_lm = config.attn_config['prefix_lm']
    self.attn_uses_sequence_id = config.attn_config['attn_uses_sequence_id']
    self.alibi = config.attn_config['alibi']
    self.alibi_bias_max = config.attn_config['alibi_bias_max']
    self.is_causal = not self.prefix_lm

    self.n_heads = config.n_heads
    self.n_layers = config.n_layers
    self.max_seq_len = config.max_seq_len
    self.return_dict = config.return_dict
    self.use_cache = config.use_cache

    self._attn_bias_initialized = False
    self.attn_bias = None
    self.attn_bias_shape = attn_bias_shape(
        self.attn_impl,
        self.n_heads,
        self.max_seq_len,
        self.alibi,
        prefix_lm=self.prefix_lm,
        causal=self.is_causal,
        use_sequence_id=self.attn_uses_sequence_id
    )

    # Define layer norm type
    if config.norm_type.lower() not in NORM_CLASS_REGISTRY.keys():
      norm_options = ' | '.join(NORM_CLASS_REGISTRY.keys())
      raise NotImplementedError(f'Requested norm type ({config.norm_type}) is not implemented within this repo (Options: {norm_options}).')
    norm_class = NORM_CLASS_REGISTRY[config.norm_type.lower()]

    # Init layers
    self.wte = Embedding(config.vocab_size, config.d_model, dtype=config.dtype)
    if not self.alibi:
      self.wpe = Embedding(config.max_seq_len, config.d_model, dtype=config.dtype)
    self.blocks = ModuleList([MPTBlock(config) for _ in range(config.n_layers)])
    self.norm_f = norm_class(config.d_model, dtype=config.dtype)

  def get_input_embeddings(self):
    return self.wte

  def set_input_embeddings(self, value):
    self.wte = value

  def _attn_bias(self, dtype, attention_mask: Optional[relax.Expr]=None, prefix_mask: Optional[relax.Expr]=None, sequence_id: Optional[relax.Expr]=None):
    if not self._attn_bias_initialized:
      if self.attn_bias_shape:
        self.attn_bias = nn.emit(relax.op.zeros(self.attn_bias_shape, dtype=dtype))
        self.attn_bias = build_attn_bias(
            self.attn_impl, self.attn_bias, self.n_heads, self.max_seq_len, causal=self.is_causal, alibi=self.alibi, alibi_bias_max=self.alibi_bias_max
        )
      self._attn_bias_initialized = True
    if self.attn_impl == 'flash':
      return (self.attn_bias, attention_mask)
    if self.attn_bias is not None:
      self.attn_bias = nn.emit(tvm.tir.Cast(dtype, self.attn_bias))
    attn_bias = self.attn_bias
    if self.prefix_lm:
        assert isinstance(attn_bias, relax.Expr)
        assert isinstance(prefix_mask, relax.Expr)
        attn_bias = self._apply_prefix_mask(attn_bias, prefix_mask)
    if self.attn_uses_sequence_id and sequence_id is not None:
        assert isinstance(attn_bias, relax.Expr)
        attn_bias = self._apply_sequence_id(attn_bias, sequence_id)
    if attention_mask is not None:
      s_k = attention_mask.struct_info.shape[-1]
      if attn_bias is None:
        attn_bias = nn.emit(relax.op.zeros((1, 1, 1, s_k), dtype=dtype))
      else:
        _s_k = relax.op.maximum(relax.const(0), attn_bias.struct_info.shape[-1] - s_k)
        # slicing attn_bias[:, :, :, _s_k:]
        s_k_end = attn_bias.struct_info.shape[3]
        attn_bias = nn.emit(relax.op.strided_slice(attn_bias, [3], [_s_k], [s_k_end]))
      if prefix_mask is not None and attention_mask.struct_info.shape != prefix_mask.struct_info.shape:
        raise ValueError(f'attention_mask shape={attention_mask.shape} ' + f'and prefix_mask shape={prefix_mask.shape} are not equal.')
      min_val = tvm.tir.min_value(attn_bias.struct_info.dtype)
      attn_mask = nn.emit(tvm.tir.bitwise_not(relax.op.reshape(attention_mask, (-1, 1, 1, s_k))))
      attn_bias = nn.emit(relax.op.masked_fill(attn_bias, attn_mask, min_val))
    return (attn_bias, None)

  def _apply_prefix_mask(self, attn_bias: relax.Expr, prefix_mask: relax.Expr):
    (s_k, s_q) = attn_bias.struct_info.shape[-2:]
    if s_k != self.max_seq_len or s_q != self.max_seq_len:
      raise ValueError('attn_bias does not match the expected shape. ' + f'The last two dimensions should both be {self.max_seq_len} ' + f'but are {s_k} and {s_q}.')
    seq_len = prefix_mask.struct_info.shape[-1]
    if seq_len > self.max_seq_len:
      raise ValueError(f'prefix_mask sequence length cannot exceed max_seq_len={self.max_seq_len}')
    # slicing attn_bias[..., :seq_len, :seq_len]
    dims_len = attn_bias.struct_info.ndim
    attn_bias = nn.emit(relax.op.strided_slice(attn_bias, [dims_len - 2, dims_len - 1], [relax.const(0), relax.const(0)], [seq_len, seq_len]))
    causal = nn.emit(relax.op.reshape(relax.op.tril(relax.op.ones((seq_len, seq_len), dtype="bool")), (1, 1, seq_len, seq_len)))
    prefix = nn.emit(relax.op.reshape(prefix_mask, (-1, 1, 1, seq_len)))
    cannot_attend = nn.emit(tvm.tir.bitwise_not(relax.op.logical_or(causal, tvm.tir.Cast("bool", prefix))))
    min_val = tvm.tir.min_value(attn_bias.struct_info.dtype)
    attn_bias = nn.emit(relax.op.masked_fill(attn_bias, cannot_attend, min_val))
    return attn_bias

  def _apply_sequence_id(self, attn_bias: relax.Expr, sequence_id: relax.Expr):
    seq_len = sequence_id.struct_info.shape[-1]
    if seq_len > self.max_seq_len:
      raise ValueError(f'sequence_id sequence length cannot exceed max_seq_len={self.max_seq_len}')
    # slicing attn_bias[..., :seq_len, :seq_len]
    dims_len = attn_bias.struct_info.ndim
    attn_bias = nn.emit(relax.op.strided_slice(attn_bias, [dims_len - 2, dims_len - 1], [relax.const(0), relax.const(0)], [seq_len, seq_len]))
    seq_id_l = nn.emit(relax.op.reshape(sequence_id, (-1, seq_len, 1)))
    seq_id_r = nn.emit(relax.op.reshape(sequence_id, (-1, 1, seq_len)))
    cannot_attend = nn.emit(relax.op.expand_dims(relax.op.logical_not(relax.op.equal(seq_id_l, seq_id_r)), axis=1))
    min_val = tvm.tir.min_value(attn_bias.struct_info.dtype)
    attn_bias = nn.emit(relax.op.masked_fill(attn_bias, cannot_attend, min_val))
    return attn_bias

  def forward(
      self,
      input_ids: relax.Expr,
      past_key_values: Optional[List[Tuple[relax.Expr]]]=None,
      attention_mask: Optional[relax.Expr]=None,
      prefix_mask: Optional[relax.Expr]=None,
      sequence_id: Optional[relax.Expr]=None,
      return_dict: Optional[bool]=None,
      output_attentions: Optional[bool]=None,
      output_hidden_states: Optional[bool]=None,
      use_cache: Optional[bool]=None
  ):
    return_dict = return_dict if return_dict is not None else self.return_dict
    use_cache = use_cache if use_cache is not None else self.use_cache
    if attention_mask is not None:
      attention_mask = nn.emit(tvm.tir.Cast("bool", attention_mask))
    if prefix_mask is not None:
      prefix_mask = nn.emit(tvm.tir.Cast("bool", prefix_mask))
    if not return_dict:
      raise NotImplementedError('return_dict False is not implemented yet for MPT')
    if output_attentions:
      if self.attn_impl != 'torch':
        raise NotImplementedError('output_attentions is not implemented for MPT when using attn_impl `flash` or `triton`.')
    if self.prefix_lm and prefix_mask is None:
      raise ValueError('prefix_mask is a required argument when MPT is configured with prefix_lm=True.')

    S = input_ids.struct_info.shape[1]
    assert S <= self.max_seq_len, f'Cannot forward input with seq_len={S}, this model only supports seq_len<={self.max_seq_len}'

    tok_emb = self.wte(input_ids)
    if self.alibi:
      x = tok_emb
    else:
      past_position = 0
      if past_key_values is not None:
        if len(past_key_values) != self.n_layers:
          raise ValueError(f'past_key_values must provide a past_key_value for each attention ' + f'layer in the network (len(past_key_values)={len(past_key_values)!r}; self.config.n_layers={self.n_layers!r}).')
        past_position = past_key_values[0][0].struct_info.shape[1]
        if self.attn_impl == 'torch':
          past_position = past_key_values[0][0].struct_info.shape[3]
      if S + past_position > self.max_seq_len:
        raise ValueError(f'Cannot forward input with past sequence length {past_position} and current sequence length {S + 1}, this model only supports total sequence length <= {self.max_seq_len}.')
      pos = nn.emit(relax.op.expand_dims(relax.op.arange(past_position, S + past_position, dtype="long"), axis=0))
      if attention_mask is not None:
        pos_diff_to_slice = nn.emit(relax.op.cumsum(tvm.tir.Cast("int32", tvm.tir.bitwise_not(attention_mask)), axis=1))
        dim1_len = pos_diff_to_slice.struct_info.shape[1]
        # slicing [:, past_position:]
        pos_diff = nn.emit(relax.op.strided_slice(pos_diff_to_slice, [1], [past_position], [dim1_len]))
        pos = nn.emit(relax.op.clip(pos - pos_diff, min=0))
      pos_emb = self.wpe(pos)
      x = tok_emb + pos_emb
    (attn_bias, attention_mask) = self._attn_bias(dtype=x.dtype, attention_mask=attention_mask, prefix_mask=prefix_mask, sequence_id=sequence_id)
    if use_cache and past_key_values is None:
      past_key_values = [() for _ in range(self.n_layers)]
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    for (b_idx, block) in enumerate(self.blocks):
      if output_hidden_states:
        assert all_hidden_states is not None
        all_hidden_states = all_hidden_states + (x,)
      past_key_value = past_key_values[b_idx] if past_key_values is not None else None
      (x, attn_weights, past_key_value) = block(x, past_key_value=past_key_value, attn_bias=attn_bias, attention_mask=attention_mask, is_causal=self.is_causal)
      if past_key_values is not None:
        past_key_values[b_idx] = past_key_value
      if output_attentions:
        assert all_self_attns is not None
        all_self_attns = all_self_attns + (attn_weights,)
    x = self.norm_f(x)
    if output_hidden_states:
      assert all_hidden_states is not None
      all_hidden_states = all_hidden_states + (x,)
    return x, past_key_values, all_hidden_states, all_self_attns

  def fsdp_wrap_fn(self, module):
    return isinstance(module, MPTBlock)

  def activation_checkpointing_fn(self, module):
    return isinstance(module, MPTBlock)


class MPTForCausalLM(nn.Module):
  def __init__(self, config: MPTConfig):
    if not config.tie_word_embeddings:
      raise ValueError('MPTForCausalLM only supports tied word embeddings')
    self.transformer = MPTModel(config)
    self.dtype = config.dtype

    self.return_dict = config.return_dict
    self.use_cache = config.use_cache

  def get_input_embeddings(self):
    return self.transformer.wte

  def set_input_embeddings(self, value):
    self.transformer.wte = value

  def get_output_embeddings(self):
    return self.transformer.wte

  def set_output_embeddings(self, new_embeddings):
    self.transformer.wte = new_embeddings

  def set_decoder(self, decoder):
    self.transformer = decoder

  def get_decoder(self):
    return self.transformer

  def forward(
      self,
      input_ids: relax.Expr,
      past_key_values: Optional[List[Tuple[relax.Expr]]]=None,
      attention_mask: Optional[relax.Expr]=None,
      prefix_mask: Optional[relax.Expr]=None,
      sequence_id: Optional[relax.Expr]=None,
      return_dict: Optional[bool]=None,
      output_attentions: Optional[bool]=None,
      output_hidden_states: Optional[bool]=None,
      use_cache: Optional[bool]=None
  ):
    return_dict = return_dict if return_dict is not None else self.return_dict
    use_cache = use_cache if use_cache is not None else self.use_cache
    outputs = self.transformer(
        input_ids=input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        prefix_mask=prefix_mask,
        sequence_id=sequence_id,
        return_dict=return_dict,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        use_cache=use_cache
    )
    logits = nn.emit(relax.op.matmul(outputs[0], self.transformer.wte.weight))

    return logits, outputs[1]

  def fsdp_wrap_fn(self, module):
    return isinstance(module, MPTBlock)

  def activation_checkpointing_fn(self, module):
    return isinstance(module, MPTBlock)

  def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
    if inputs_embeds is not None:
      raise NotImplementedError('inputs_embeds is not implemented for MPT yet')
    attention_mask = kwargs['attention_mask'].bool()
    if attention_mask[:, -1].sum() != attention_mask.shape[0]:
      raise NotImplementedError('MPT does not support generation with right padding.')
    if self.transformer.attn_uses_sequence_id and self.training:
      # slicing input_ids[:1]
      input_ids_slice = nn.emit(relax.op.strided_slice(input_ids, [0], [relax.const(0)], [relax.const(1)]))
      sequence_id = nn.emit(relax.op.zeros_like(input_ids_slice))
    else:
      sequence_id = None
    if past_key_values is not None:
      # slicing input_ids[:, -1]
      dim1_len = input_ids.struct_info.shape[1]
      input_ids_slice = nn.emit(relax.op.strided_slice(input_ids, [1], [dim1_len - 1], [dim1_len]))
      input_ids = nn.emit(relax.op.expand_dims(input_ids_slice, axis=-1))
    if self.transformer.prefix_lm:
      prefix_mask = nn.emit(relax.op.ones_like(attention_mask, self.dtype))
      if kwargs.get('use_cache') == False:
        raise NotImplementedError('MPT with prefix_lm=True does not support use_cache=False.')
    else:
      prefix_mask = None
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'prefix_mask': prefix_mask,
        'sequence_id': sequence_id,
        'past_key_values': past_key_values,
        'use_cache': kwargs.get('use_cache', True)
    }

  @staticmethod
  def _reorder_cache(past_key_values, beam_idx):
      """Used by HuggingFace generate when using beam search with kv-caching.
      See https://github.com/huggingface/transformers/blob/3ec7a47664ebe40c40f4b722f6bb1cd30c3821ec/src/transformers/models/gpt2/modeling_gpt2.py#L1122-L1133
      for an example in transformers.
      """
      reordered_past = []
      for layer_past in past_key_values:
        reordered_past += [tuple((nn.emit(relax.op.take(past_state, beam_idx, 0)) for past_state in layer_past))]
      return reordered_past


def create_decoding_func(bb: relax.BlockBuilder, config: MPTConfig) -> Dict[int, str]:
  pidx2pname: Dict[int, str] = {}
  with bb.function("decode"):
    model = MPTForCausalLM(config)
    input_ids = nn.Placeholder((1, 1), dtype="int32", name="input_ids")
    # Placeholder for compatibility to LLAMA
    all_seq_len_shape = relax.Var("place_holder", R.Object())
    state = relax.Var("state", R.Tuple([R.Object()] * config.n_layers * 5))
    with bb.dataflow():
      logits, states = model(input_ids, state)
      params = [
          input_ids,
          all_seq_len_shape,
          state,
      ] + model.parameters()

      named_params = named_parameters(model)
      for i, (name, param) in enumerate(named_params.items()):
        pidx2pname[i] = name
        assert param.same_as(params[i + 3])

      gv = bb.emit_output((logits, relax.Tuple(states)))
    bb.emit_func_output(gv, params)

  mod = bb.get()
  gv = mod.get_global_var("decode")
  bb.update_func(gv, mod[gv].with_attr("num_input", 3))

  return pidx2pname

def get_model(args, hf_config):
  from transformers import AutoModelForCausalLM # type: ignore[import]
  import torch                                  # type: ignore[import]

  model_name = args.model
  # TODO: download model and use model_path instead of args for from_pretrained
  # model_path = args.model_path
  dtype = args.quantization.model_dtype
  # Recommendation from https://huggingface.co/mosaicml/mpt-7b-instruct
  max_seq_len = args.max_seq_len if args.max_seq_len is not None else 4096  # 4096 recommended

  hf_config.update({"max_seq_len": max_seq_len})
  # hf_config.update({"max_new_tokens": args.seq_len})

  if model_name.startswith("mpt-"):
    config = MPTConfig(**hf_config, dtype=dtype)

    bb = relax.BlockBuilder()
    create_decoding_func(bb, config)

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
      param_list[i] = tvm.nd.array(
        param.detach().cpu().numpy().astype(dtype), device
      )
    del hf_model

    print(mod)
    return mod, param_list

  raise ValueError(f"Unsupported model: {model_name}")