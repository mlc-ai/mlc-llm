import math
from einops import rearrange  # TODO: replace
import warnings
from typing import Optional, Tuple, List
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

# Low-precision layer norm for mpt-7b-instruct, where are no biases expected
class LPLayerNormWOBias(nn.Module):
  def __init__(self, normalized_shape, eps=1e-05, dtype=None):
    self.weight = nn.Parameter((normalized_shape,), dtype=dtype, name="low_precision_layernorm_weight")
    # TODO: check
    self.weight = relax.op.ones((normalized_shape,), dtype)
    self.variance_epsilon = tvm.tir.const(eps, dtype)

  def forward(self, x):
    module_device = x.device
    downcast_x = _cast_if_autocast_enabled(x)
    downcast_weight = _cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
    downcast_bias = _cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
    with torch.autocast(enabled=False, device_type=module_device.type):
      return torch.nn.functional.layer_norm(downcast_x, self.normalized_shape, downcast_weight, downcast_bias, self.eps)

NORM_CLASS_REGISTRY = {'low_precision_layernorm': LPLayerNormWOBias}


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
      layernorm_class = LPLayerNormWOBias if low_precision_layernorm else LayerNorm
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

    self._attn_bias_initialized = False
    self.attn_bias = None
    self.attn_bias_shape = attn_bias_shape(
        self.attn_impl,
        config.n_heads,
        config.max_seq_len,
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

  def _attn_bias(self, device, dtype, attention_mask: Optional[relax.Expr]=None, prefix_mask: Optional[relax.Expr]=None, sequence_id: Optional[relax.Expr]=None):
    if not self._attn_bias_initialized:
        if self.attn_bias_shape:
            self.attn_bias = torch.zeros(self.attn_bias_shape, device=device, dtype=dtype)
            self.attn_bias = build_attn_bias(self.attn_impl, self.attn_bias, self.config.n_heads, self.config.max_seq_len, causal=self.is_causal, alibi=self.alibi, alibi_bias_max=self.alibi_bias_max)
        self._attn_bias_initialized = True
    if self.attn_impl == 'flash':
        return (self.attn_bias, attention_mask)
    if self.attn_bias is not None:
        self.attn_bias = self.attn_bias.to(dtype=dtype, device=device)
    attn_bias = self.attn_bias
    if self.prefix_lm:
        assert isinstance(attn_bias, torch.Tensor)
        assert isinstance(prefix_mask, torch.Tensor)
        attn_bias = self._apply_prefix_mask(attn_bias, prefix_mask)
    if self.attn_uses_sequence_id and sequence_id is not None:
        assert isinstance(attn_bias, torch.Tensor)
        attn_bias = self._apply_sequence_id(attn_bias, sequence_id)
    if attention_mask is not None:
        s_k = attention_mask.shape[-1]
        if attn_bias is None:
            attn_bias = torch.zeros((1, 1, 1, s_k), device=device, dtype=dtype)
        else:
            _s_k = max(0, attn_bias.size(-1) - s_k)
            attn_bias = attn_bias[:, :, :, _s_k:]
        if prefix_mask is not None and attention_mask.shape != prefix_mask.shape:
            raise ValueError(f'attention_mask shape={attention_mask.shape} ' + f'and prefix_mask shape={prefix_mask.shape} are not equal.')
        min_val = torch.finfo(attn_bias.dtype).min
        attn_bias = attn_bias.masked_fill(~attention_mask.view(-1, 1, 1, s_k), min_val)
    return (attn_bias, None)

  def _apply_prefix_mask(self, attn_bias: torch.Tensor, prefix_mask: torch.Tensor):
    (s_k, s_q) = attn_bias.shape[-2:]
    if s_k != self.config.max_seq_len or s_q != self.config.max_seq_len:
      raise ValueError('attn_bias does not match the expected shape. ' + f'The last two dimensions should both be {self.config.max_length} ' + f'but are {s_k} and {s_q}.')
    seq_len = prefix_mask.shape[-1]
    if seq_len > self.config.max_seq_len:
      raise ValueError(f'prefix_mask sequence length cannot exceed max_seq_len={self.config.max_seq_len}')
    attn_bias = attn_bias[..., :seq_len, :seq_len]
    causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=prefix_mask.device)).view(1, 1, seq_len, seq_len)
    prefix = prefix_mask.view(-1, 1, 1, seq_len)
    cannot_attend = ~torch.logical_or(causal, prefix.bool())
    min_val = torch.finfo(attn_bias.dtype).min
    attn_bias = attn_bias.masked_fill(cannot_attend, min_val)
    return attn_bias

  def _apply_sequence_id(self, attn_bias: torch.Tensor, sequence_id: torch.LongTensor):
    seq_len = sequence_id.shape[-1]
    if seq_len > self.config.max_seq_len:
        raise ValueError(f'sequence_id sequence length cannot exceed max_seq_len={self.config.max_seq_len}')
    attn_bias = attn_bias[..., :seq_len, :seq_len]
    cannot_attend = torch.logical_not(torch.eq(sequence_id.view(-1, seq_len, 1), sequence_id.view(-1, 1, seq_len))).unsqueeze(1)
    min_val = torch.finfo(attn_bias.dtype).min
    attn_bias = attn_bias.masked_fill(cannot_attend, min_val)
    return attn_bias

  def forward(self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple[torch.FloatTensor]]]=None, attention_mask: Optional[torch.ByteTensor]=None, prefix_mask: Optional[torch.ByteTensor]=None, sequence_id: Optional[torch.LongTensor]=None, return_dict: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, use_cache: Optional[bool]=None):
    return_dict = return_dict if return_dict is not None else self.config.return_dict
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    if attention_mask is not None:
        attention_mask = attention_mask.bool()
    if prefix_mask is not None:
        prefix_mask = prefix_mask.bool()
    if not return_dict:
        raise NotImplementedError('return_dict False is not implemented yet for MPT')
    if output_attentions:
        if self.attn_impl != 'torch':
            raise NotImplementedError('output_attentions is not implemented for MPT when using attn_impl `flash` or `triton`.')
    if attention_mask is not None and attention_mask[:, 0].sum() != attention_mask.shape[0] and self.training:
        raise NotImplementedError('MPT does not support training with left padding.')
    if self.prefix_lm and prefix_mask is None:
        raise ValueError('prefix_mask is a required argument when MPT is configured with prefix_lm=True.')
    if self.training:
        if self.attn_uses_sequence_id and sequence_id is None:
            raise ValueError('sequence_id is a required argument when MPT is configured with attn_uses_sequence_id=True ' + 'and the model is in train mode.')
        elif self.attn_uses_sequence_id is False and sequence_id is not None:
            warnings.warn('MPT received non-None input for `sequence_id` but is configured with attn_uses_sequence_id=False. ' + 'This input will be ignored. If you want the model to use `sequence_id`, set attn_uses_sequence_id to True.')
    S = input_ids.size(1)
    assert S <= self.config.max_seq_len, f'Cannot forward input with seq_len={S}, this model only supports seq_len<={self.config.max_seq_len}'
    tok_emb = self.wte(input_ids)
    if self.alibi:
        x = tok_emb
    else:
        past_position = 0
        if past_key_values is not None:
            if len(past_key_values) != self.config.n_layers:
                raise ValueError(f'past_key_values must provide a past_key_value for each attention ' + f'layer in the network (len(past_key_values)={len(past_key_values)!r}; self.config.n_layers={self.config.n_layers!r}).')
            past_position = past_key_values[0][0].size(1)
            if self.attn_impl == 'torch':
                past_position = past_key_values[0][0].size(3)
        if S + past_position > self.config.max_seq_len:
            raise ValueError(f'Cannot forward input with past sequence length {past_position} and current sequence length {S + 1}, this model only supports total sequence length <= {self.config.max_seq_len}.')
        pos = torch.arange(past_position, S + past_position, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        if attention_mask is not None:
            pos = torch.clamp(pos - torch.cumsum((~attention_mask).to(torch.int32), dim=1)[:, past_position:], min=0)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
    (attn_bias, attention_mask) = self._attn_bias(device=x.device, dtype=x.dtype, attention_mask=attention_mask, prefix_mask=prefix_mask, sequence_id=sequence_id)
    if use_cache and past_key_values is None:
        past_key_values = [() for _ in range(self.config.n_layers)]
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
    return BaseModelOutputWithPast(last_hidden_state=x, past_key_values=past_key_values, hidden_states=all_hidden_states, attentions=all_self_attns)

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
    return_dict = return_dict if return_dict is not None else self.config.return_dict
    use_cache = use_cache if use_cache is not None else self.config.use_cache
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
    logits = nn.emit(relax.op.matmul(outputs.last_hidden_state, self.transformer.wte.weight))

    return logits, outputs.past_key_values

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
      # TODO: [:1] in Relax?
      sequence_id = nn.emit(relax.op.zeros_like(input_ids[:1]))
    else:
      sequence_id = None
    if past_key_values is not None:
      # TODO: Relax implementation?
      input_ids = input_ids[:, -1].unsqueeze(-1)
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
        # TODO: Relax implementation?
        reordered_past += [tuple((past_state.index_select(0, beam_idx) for past_state in layer_past))]
      return reordered_past


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