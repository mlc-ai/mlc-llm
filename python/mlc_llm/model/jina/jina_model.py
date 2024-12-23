"""
Ref

post-ln: https://arxiv.org/pdf/2002.04745
"""

import dataclasses
from typing import Any, Dict, Optional, Union
from functools import partial

from mlc_llm.support.config import ConfigBase
from mlc_llm import op as op_ext
from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.relax.frontend.nn.llm import llama_rope


@dataclasses.dataclass
class JinaConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the jina-embedding-v3 model."""

    vocab_size: int
    hidden_act = "gelu"
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    layer_norm_eps: float = 1e-05
    layer_norm_type: str = "layer_norm"
    lora_alpha: float = 1.
    lora_num_task: int = 5
    lora_rank: int = 4
    rotary_emb_base: float = 10000.0
    max_position_embeddings: int = 8194
    
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        pass


class LoRAEmbedding(nn.Module):
    """
    Module for embedding layer.
    """

    def __init__(
        self,
        num: Union[int, str, tir.PrimExpr],
        dim: Union[int, str, tir.PrimExpr],
        num_task: int,
        lora_alpha: float,
        lora_rank: int,
        dtype: Optional[str] = None,
    ):
        self.num = num
        self.dim = dim
        self.weight = nn.Parameter((num, dim), dtype=dtype)
        self.lora_a = nn.Parameter((num_task, num, lora_rank), dtype=dtype)
        self.lora_b = nn.Parameter((num_task, lora_rank, dim), dtype=dtype)
        self.scaling = lora_alpha / lora_rank

    def forward(self, x: Tensor, i: Tensor):
        lora = op.matmul(
            op.take(self.lora_a, i, axis=0),
            op.take(self.lora_b, i, axis=0),
        ) * self.scaling
        w = op.add(self.weight, lora)

        if x.ndim == 1:
            return op.take(w, x, axis=0)
        return op.reshape(
            op.take(
                w,
                op.reshape(x, shape=[-1]),
                axis=0,
            ),
            shape=[*x.shape, self.dim],  # TODO(@junrushao): revisit and remove self.dim
        )


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: Union[int, str, tir.PrimExpr],
        out_features: Union[int, str, tir.PrimExpr],
        num_task: int,
        lora_alpha: float,
        lora_rank: int,
        bias: bool = True,
        dtype: Optional[str] = None,
        out_dtype: Optional[str] = None,
    ):
        #  config.lora_num_task, config.lora_alpha, config.lora_rank,
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype
        self.weight = nn.Parameter((out_features, in_features), dtype)
        if bias:
            self.bias = nn.Parameter((out_features,), dtype=dtype if out_dtype is None else out_dtype)
        else:
            self.bias = None
        self.lora_a = nn.Parameter((num_task, lora_rank, in_features), dtype=dtype)
        self.lora_b = nn.Parameter((num_task, out_features, lora_rank), dtype=dtype)
        self.scaling = lora_alpha / lora_rank

    def forward(self, x: Tensor, i: Tensor) -> Tensor:
        lora = op.matmul(
            op.take(self.lora_b, i, axis=0),
            op.take(self.lora_a, i, axis=0),
        ) * self.scaling
        w = op.add(self.weight, lora)

        # x: [*B, in_features]
        # w: [in_features, out_features]
        w = op.permute_dims(w)
        # x: [*B, out_features]
        x = op.matmul(x, w, out_dtype=self.out_dtype)
        if self.bias is not None:
            x = x + self.bias
        return x

    def to(self, dtype: Optional[str] = None) -> None:
        """
        Override to() such that we do not convert bias if there is `out_dtype`.
        Otherwise, we might run into dtype mismatch when computing `x + self.bias`
        since x is of type `out_dtype` and bias becomes `dtype`, potentially different.
        """
        self.lora_a.to(dtype=dtype)
        self.lora_b.to(dtype=dtype)
        self.weight.to(dtype=dtype)
        if self.bias is not None and self.out_dtype is None:
            self.bias.to(dtype=dtype)
        if dtype is not None and isinstance(getattr(self, "dtype", None), str):
            self.dtype = dtype  # pylint: disable=attribute-defined-outside-init


class JinaEmbeddings(nn.Module):
    def __init__(self, config: JinaConfig):
        self.word_embeddings = LoRAEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.lora_num_task,
            config.lora_alpha,
            config.lora_rank,
        )
        type_vocab_size = 1
        self.token_type_embeddings = LoRAEmbedding(
            type_vocab_size,
            config.hidden_size,
            config.lora_num_task,
            config.lora_alpha,
            config.lora_rank,
            )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids: Tensor, task_id: Tensor):
        words_embeddings = self.word_embeddings(input_ids, task_id)
        token_type_ids = op.zeros(input_ids.shape, dtype="int32")
        token_type_embeddings = self.token_type_embeddings(token_type_ids, task_id)
        embeddings = words_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        return embeddings


class JinaMultiHeadAttention(nn.Module):
    def __init__(self, config: JinaConfig):
        if config.num_attention_heads % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split {config.num_attention_heads} attention heads"
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.rotary_emb_base = config.rotary_emb_base
        self.num_heads = config.num_attention_heads // config.tensor_parallel_shards
        self.head_dim = config.hidden_size // self.num_heads

        self.qkv = LoRALinear(
            in_features=config.hidden_size,
            out_features=3 * self.num_heads * self.head_dim,
            num_task=config.lora_num_task,
            lora_alpha=config.lora_alpha,
            lora_rank=config.lora_rank,
            bias=True,
        )
        self.softmax_scale = 1.0 / config.hidden_size / config.num_attention_heads
        self.out_proj = LoRALinear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            num_task=config.lora_num_task,
            lora_alpha=config.lora_alpha,
            lora_rank=config.lora_rank,
            bias=True,
        )

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, task_id: Tensor):
        d, h = self.head_dim, self.num_heads
        b, s, _ = hidden_states.shape

        qkv = self.qkv(hidden_states, task_id)
        qkv = op.reshape(qkv, (b, s, 3 * h, d))
        q_rot, k_rot, v = llama_rope(qkv, s, self.rotary_emb_base, 1., h, h, {})
        output = op_ext.attention(q_rot, k_rot, v, attention_mask, attn_score_scaling_factor=self.softmax_scale)
        output = self.out_proj(output, task_id)

        return output

ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.silu,
    "swish": nn.silu,
    "gelu_new": partial(nn.gelu, approximate=True),
}


class JinaFFN(nn.Module):
    def __init__(self, config: JinaConfig):
        self.fc1 = LoRALinear(
            in_features=config.hidden_size,
            out_features=config.hidden_size * 4,
            num_task=config.lora_num_task,
            lora_alpha=config.lora_alpha,
            lora_rank=config.lora_rank,
            bias=True,
        )
        self.intermediate_act_fn = ACT2FN[config.hidden_act]
        self.fc2 = LoRALinear(
            in_features=config.hidden_size * 4,
            out_features=config.hidden_size,
            num_task=config.lora_num_task,
            lora_alpha=config.lora_alpha,
            lora_rank=config.lora_rank,
            bias=True,
        )

    def forward(self, hidden_states: Tensor, input_tensor: Tensor):
        hidden_states = self.fc1(hidden_states, input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.fc2(hidden_states, input_tensor)
        return hidden_states


class JinaLayer(nn.Module):
    def __init__(self, config: JinaConfig):
        self.mha = JinaMultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = JinaFFN(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, task_id: Tensor):
        residual = hidden_states
        output = self.mha(hidden_states, attention_mask, task_id)
        output = output + residual
        output = self.layer_norm1(output)
        residual = output
        output = self.mlp(output, task_id)
        output = output + residual
        output = self.layer_norm2(output)
        return output


class JinaEncoder(nn.Module):
    def __init__(self, config: JinaConfig):
        self.layers = nn.ModuleList([JinaLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, task_id: Tensor):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, task_id)
        return hidden_states


class JinaPooler(nn.Module):
    def __init__(self, config: JinaConfig):
        self.hidden_size = config.hidden_size

    def forward(self, outputs: Tensor, attention_mask: Tensor):
        import numpy as np

        # Mean pool: https://huggingface.co/jinaai/jina-embeddings-v3#why-use-mean-pooling
        attention_mask_expanded = op.repeat(op.unsqueeze(attention_mask, 2).astype(outputs.dtype), self.hidden_size, axis=2)
        sum = op.sum(op.multiply(outputs, attention_mask_expanded), axis=1)
        count = op.sum(attention_mask_expanded, axis=1)
        eps = nn.Tensor.from_const(np.ones([], dtype=count.dtype) * 1e-9)
        count = op.maximum(count, eps)
        avg = op.divide(sum, count)
        # Return l2 norm
        scale = op.sqrt(op.sum(op.square(avg)))
        return avg / scale


class JinaModel(nn.Module):
    def __init__(self, config: JinaConfig):
        self.embeddings = JinaEmbeddings(config)
        self.encoder = JinaEncoder(config)
        self.pooler = JinaPooler(config)
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def forward(self, inputs: Tensor, attention_mask: Tensor, task_id: Tensor):
        embeddings = self.embeddings(inputs, task_id)
        encoder_output = self.encoder(embeddings, attention_mask, task_id)
        output = self.pooler(encoder_output, attention_mask)
        return output

    def prefill(self, inputs: Tensor, attention_mask: Tensor, task_id: Tensor):
        return self.forward(inputs, attention_mask, task_id)

    def get_default_spec(self):
        mod_spec = {
            "prefill": {
                "inputs": nn.spec.Tensor(["batch_size", "seq_len"], "int32"),
                "attention_mask": nn.spec.Tensor(["batch_size", "seq_len"], "int32"),
                "task_id": nn.spec.Tensor([], "int32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
