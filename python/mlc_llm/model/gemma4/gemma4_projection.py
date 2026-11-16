"""Pre-RoPE text attention projections for Gemma 4's larger variants.

This component does not enable the full 31B or 26B-A4B decoder. Source:
https://github.com/huggingface/transformers/blob/e7200cd9a98f0c874edcdf8f7263b7d19530b403/src/transformers/models/gemma4/modeling_gemma4.py
"""

from dataclasses import dataclass

from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm.loader import ExternMapping

from .gemma4_audio import Gemma4RMSNorm


@dataclass
class Gemma4ProjectionConfig:
    """Only the projection fields; deliberately separate from full-model config."""

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    global_head_dim: int
    num_global_key_value_heads: int | None
    attention_k_eq_v: bool
    layer_types: list[str]
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    num_kv_shared_layers: int = 0

    def __post_init__(self):
        if self.num_kv_shared_layers != 0:
            raise ValueError("Projection component requires physical, unshared KV layers")
        if self.attention_bias:
            raise ValueError("Projection component requires bias-free attention")
        for value in (
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.global_head_dim,
        ):
            if type(value) is not int or value <= 0:
                raise ValueError("Projection dimensions must be positive integers")
        if self.num_global_key_value_heads is not None and (
            type(self.num_global_key_value_heads) is not int or self.num_global_key_value_heads <= 0
        ):
            raise ValueError("Global KV heads must be positive")
        if not self.layer_types or set(self.layer_types) - {"sliding_attention", "full_attention"}:
            raise ValueError("Unsupported attention layer types")
        if not 0 < self.rms_norm_eps < 1:
            raise ValueError("Invalid RMSNorm epsilon")
        for index in range(len(self.layer_types)):
            if self.num_attention_heads % self.dimensions(index)[1]:
                raise ValueError("Query heads must be divisible by KV heads")

    @classmethod
    def from_text_config(cls, config):
        """Extract projection fields without treating other decoder fields as supported."""
        return cls(**{key: config[key] for key in cls.__dataclass_fields__ if key in config})

    def dimensions(self, layer_idx: int) -> tuple[int, int, bool]:
        if not 0 <= layer_idx < len(self.layer_types):
            raise ValueError("Invalid projection layer index")
        is_global = self.layer_types[layer_idx] == "full_attention"
        shared_projection = is_global and self.attention_k_eq_v
        kv_heads = (
            self.num_global_key_value_heads
            if shared_projection and self.num_global_key_value_heads is not None
            else self.num_key_value_heads
        )
        return self.global_head_dim if is_global else self.head_dim, kv_heads, shared_projection


class Gemma4TextProjections(nn.Module):
    """Return normalized Q/K/V before RoPE, attention or cache insertion.

    K=V shares the raw projection only: K has learned scaling and later receives
    RoPE, while V has unscaled RMS normalization and never receives RoPE.
    """

    def __init__(self, config: Gemma4ProjectionConfig, layer_idx: int):
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim, self.num_kv_heads, self.shared_projection = config.dimensions(layer_idx)
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        if not self.shared_projection:
            self.v_proj = nn.Linear(
                config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
            )
        self.q_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps, with_scale=False)

    def forward(self, hidden: Tensor):
        batch, length, _ = hidden.shape
        query = op.reshape(
            self.q_proj(hidden), (batch, length, self.config.num_attention_heads, self.head_dim)
        )
        key = op.reshape(self.k_proj(hidden), (batch, length, self.num_kv_heads, self.head_dim))
        value = (
            key
            if self.shared_projection
            else op.reshape(self.v_proj(hidden), (batch, length, self.num_kv_heads, self.head_dim))
        )
        return self.q_norm(query), self.k_norm(key), self.v_norm(value)

    def get_default_spec(self):
        return nn.spec.ModuleSpec.from_raw(
            {
                "forward": {
                    "hidden": nn.spec.Tensor((1, "seq_len", self.config.hidden_size), "float32"),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                }
            },
            self,
        )


def projection_huggingface(config: Gemma4ProjectionConfig, layer_idx: int) -> ExternMapping:
    """Map only this component's real checkpoint tensors; never invent V weights."""
    model = Gemma4TextProjections(config, layer_idx)
    _, parameters, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    mapping = ExternMapping()
    for name, _ in parameters:
        mapping.add_mapping(
            name, [f"model.language_model.layers.{layer_idx}.self_attn.{name}"], _float32
        )
    return mapping


def _float32(value):
    return value.astype("float32", copy=False)
