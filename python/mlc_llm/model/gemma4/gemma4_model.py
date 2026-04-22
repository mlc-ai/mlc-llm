"""Implementation for Gemma4 text architecture."""

import dataclasses
import functools
from typing import Any, Dict, List, Optional, Tuple

from tvm import te, tirx
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.relax.frontend.nn.llm import position_embedding

from mlc_llm import op as op_ext
from mlc_llm.model.gemma.gemma_model import GemmaEmbedding
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.support import logging
from mlc_llm.support import tensor_parallel as tp
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

logger = logging.getLogger(__name__)


ACT2FN = {
    "gelu": functools.partial(op.gelu, approximate="none"),
    "gelu_pytorch_tanh": functools.partial(op.gelu, approximate="tanh"),
}


@dataclasses.dataclass
class Gemma4TextConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the Gemma4 text model."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    vocab_size: int
    attention_bias: bool = False
    attention_dropout: float = 0.0
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    hidden_activation: Optional[str] = "gelu_pytorch_tanh"
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    sliding_window_size: Optional[int] = None
    layer_types: Optional[List[str]] = None
    position_embedding_base: float = 0
    global_position_embedding_base: float = 0
    hidden_size_per_layer_input: int = 0
    vocab_size_per_layer_input: int = 0
    num_global_key_value_heads: Optional[int] = None
    global_head_dim: int = 0
    attention_k_eq_v: bool = False
    num_kv_shared_layers: int = 0
    enable_moe_block: bool = False
    use_double_wide_mlp: bool = False
    num_experts: Optional[int] = None
    top_k_experts: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    final_logit_softcapping: Optional[float] = None
    tensor_parallel_shards: int = 1
    max_batch_size: int = 1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.hidden_activation is None:
            self.hidden_activation = self.kwargs.get("hidden_act", None)
        if self.hidden_activation not in ACT2FN:
            raise ValueError("Only GeLU is supported as the activation for gemma4.")
        if self.attention_bias:
            raise ValueError('Only "False" attention_bias is supported for gemma4')

        # Guard: use_double_wide_mlp requires num_kv_shared_layers > 0.
        # Without shared-KV layers, layer_uses_double_wide_mlp() always
        # returns False, silently compiling all layers with the wrong
        # intermediate_size.  Catch this early.
        if self.use_double_wide_mlp and self.num_kv_shared_layers <= 0:
            raise ValueError(
                "Gemma4 config has use_double_wide_mlp=True but "
                f"num_kv_shared_layers={self.num_kv_shared_layers}.  "
                "Shared-KV layers are required for double-wide MLP.  "
                "Set num_kv_shared_layers to the correct value from the "
                "HuggingFace config.json (typically 20 for Gemma 4 E2B)."
            )

        if self.sliding_window_size is None:
            self.sliding_window_size = self.kwargs.get("sliding_window", None)

        rope_parameters = self.kwargs.get("rope_parameters", {})
        if self.position_embedding_base == 0:
            if isinstance(rope_parameters, dict):
                sliding_params = rope_parameters.get("sliding_attention", rope_parameters)
                if isinstance(sliding_params, dict) and "rope_theta" in sliding_params:
                    self.position_embedding_base = sliding_params["rope_theta"]
            if self.position_embedding_base == 0 and "rope_theta" in self.kwargs:
                self.position_embedding_base = self.kwargs["rope_theta"]
            if self.position_embedding_base == 0:
                self.position_embedding_base = 10000

        if self.global_position_embedding_base == 0:
            if isinstance(rope_parameters, dict):
                full_params = rope_parameters.get("full_attention", {})
                if isinstance(full_params, dict) and "rope_theta" in full_params:
                    self.global_position_embedding_base = full_params["rope_theta"]
            if self.global_position_embedding_base == 0:
                self.global_position_embedding_base = self.position_embedding_base

        if self.context_window_size == 0:
            for name in ["max_position_embeddings", "max_sequence_length"]:
                if name in self.kwargs:
                    self.context_window_size = self.kwargs.pop(name)
                    logger.info(
                        "%s not found in config.json. Falling back to %s (%d)",
                        bold("context_window_size"),
                        bold(name),
                        self.context_window_size,
                    )
                    break
            else:
                raise ValueError(
                    "Unable to determine the maximum sequence length, because none of "
                    "`context_window_size`, `max_position_embeddings` or `max_sequence_length` is "
                    "provided in `config.json`."
                )
        assert self.num_attention_heads % self.num_key_value_heads == 0
        if self.num_global_key_value_heads is None:
            self.num_global_key_value_heads = self.num_key_value_heads
        if self.global_head_dim == 0:
            self.global_head_dim = self.head_dim
        assert self.num_attention_heads % self.num_global_key_value_heads == 0

        if self.prefill_chunk_size == 0:
            logger.info(
                "%s defaults to %d",
                bold("prefill_chunk_size"),
                min(self.context_window_size, 8192),
            )
            self.prefill_chunk_size = min(self.context_window_size, 8192)
        elif self.prefill_chunk_size > self.context_window_size:
            logger.info(
                "Overriding %s from %d to %d",
                bold("prefill_chunk_size"),
                self.prefill_chunk_size,
                min(self.context_window_size, 8192),
            )
            self.prefill_chunk_size = min(self.context_window_size, 8192)

        if self.layer_types is None:
            self.layer_types = ["full_attention" for _ in range(self.num_hidden_layers)]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"Gemma4 layer_types length mismatch: expected {self.num_hidden_layers}, "
                f"got {len(self.layer_types)}"
            )

    @property
    def max_head_dim(self) -> int:
        return max(self.head_dim, self.global_head_dim)

    @property
    def max_num_key_value_heads(self) -> int:
        return max(self.num_key_value_heads, self.num_global_key_value_heads)

    @property
    def per_layer_input_total_size(self) -> int:
        return self.num_hidden_layers * self.hidden_size_per_layer_input

    @property
    def input_embed_size(self) -> int:
        return self.hidden_size + self.per_layer_input_total_size

    def layer_type(self, layer_idx: int) -> str:
        return self.layer_types[layer_idx]

    def layer_head_dim(self, layer_idx: int) -> int:
        if self.layer_type(layer_idx) == "full_attention":
            return self.global_head_dim
        return self.head_dim

    def layer_num_key_value_heads(self, layer_idx: int) -> int:
        if self.layer_type(layer_idx) == "full_attention":
            return self.num_global_key_value_heads
        return self.num_key_value_heads

    def layer_rope_theta(self, layer_idx: int) -> float:
        if self.layer_type(layer_idx) == "full_attention":
            return self.global_position_embedding_base
        return self.position_embedding_base

    def layer_rope_scaling(self, layer_idx: int) -> Dict[str, Any]:
        rope_parameters = self.kwargs.get("rope_parameters", {})
        if not isinstance(rope_parameters, dict):
            return {}
        value = rope_parameters.get(self.layer_type(layer_idx), {})
        if not isinstance(value, dict):
            return {}
        # Gemma 4 uses GPT-J-style RoPE interleaving (alternating pairs at
        # [2i, 2i+1]), not NeoX-style (first-half / second-half split). In TVM
        # this corresponds to ``rope_type="gptj"``, which both computes the
        # correct frequency (``2 * (d // 2)``) and applies the correct
        # interleaving.
        #
        # The HF config uses "default" / "proportional" as rope_type labels,
        # so we normalize both to "gptj" here so that `switch_rope_freq_func`
        # routes to ``rope_freq_gptj``.  For full-attention layers with
        # ``partial_rotary_factor < 1`` we additionally set ``freq_dim_base``
        # so ``rope_freq_gptj`` uses ``head_dim`` (the HF denominator) rather
        # than ``rotary_dim`` as the frequency base --- matching HF's
        # ``inv_freq = 1 / theta ** (arange(0, rotary_dim, 2) / head_dim)``.
        result = dict(value)
        orig_type = result.get("rope_type", "")
        if orig_type in ("default", "proportional"):
            result["rope_type"] = "gptj"
        prf = result.get("partial_rotary_factor", 1.0)
        if prf < 1.0:
            head_dim = self.layer_head_dim(layer_idx)
            result["freq_dim_base"] = head_dim
        return result

    def first_kv_shared_layer_idx(self) -> int:
        return self.num_hidden_layers - self.num_kv_shared_layers

    def layer_uses_shared_cache(self, layer_idx: int) -> bool:
        first_shared = self.first_kv_shared_layer_idx()
        return layer_idx >= first_shared > 0

    def layer_cache_source(self, layer_idx: int) -> Optional[int]:
        if not self.layer_uses_shared_cache(layer_idx):
            return None
        first_shared = self.first_kv_shared_layer_idx()
        prev_layers = self.layer_types[:first_shared]
        layer_type = self.layer_type(layer_idx)
        return len(prev_layers) - 1 - prev_layers[::-1].index(layer_type)

    def layer_uses_double_wide_mlp(self, layer_idx: int) -> bool:
        return self.use_double_wide_mlp and self.layer_uses_shared_cache(layer_idx)

    def unsupported_runtime_features(self) -> List[str]:
        """Return Gemma4 features that still need native support."""
        issues: List[str] = []
        layer_types = set(self.layer_types)
        unknown_layer_types = sorted(layer_types - {"full_attention", "sliding_attention"})
        if unknown_layer_types:
            issues.append(f"unknown layer_types={unknown_layer_types}")
        if (
            "full_attention" in layer_types
            and "sliding_attention" in layer_types
            and self.num_global_key_value_heads != self.num_key_value_heads
        ):
            issues.append(
                "mixed full/sliding key-value head counts "
                f"({self.num_global_key_value_heads} vs {self.num_key_value_heads})"
            )
        if self.attention_k_eq_v:
            issues.append("shared K/V projection weights")
        if self.enable_moe_block:
            issues.append("Mixture-of-Experts blocks")
        return issues


@dataclasses.dataclass
class Gemma4Config(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the Gemma4 model."""

    text_config: Gemma4TextConfig = None
    vocab_size: int = 262_144
    tensor_parallel_shards: int = 1
    max_batch_size: int = 1
    context_window_size: int = -1
    sliding_window_size: int = -1
    prefill_chunk_size: int = -1
    is_text_model: bool = False
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.text_config is None:
            self.is_text_model = True
            self.text_config = Gemma4TextConfig.from_dict(self.kwargs)

        text_config_dict: Dict[str, Any]
        if isinstance(self.text_config, Gemma4TextConfig):
            text_config_dict = dataclasses.asdict(self.text_config)
        else:
            text_config_dict = dict(self.text_config)

        for k, v in text_config_dict.pop("kwargs", {}).items():
            text_config_dict[k] = v

        self.text_config = Gemma4TextConfig.from_dict(text_config_dict)

        for k in ["context_window_size", "prefill_chunk_size", "sliding_window_size"]:
            if getattr(self, k) <= 0 and hasattr(self.text_config, k):
                setattr(self, k, getattr(self.text_config, k))


class Gemma4ScaledEmbedding(GemmaEmbedding):
    """Embedding module with a fixed post-lookup scale."""

    def __init__(self, num_embeddings, embedding_dim, embed_scale: float):
        super().__init__(num_embeddings, embedding_dim)
        self.embed_scale = embed_scale

    def forward(self, input_ids: Tensor):
        return super().forward(input_ids) * self.embed_scale


# Maximum bytes allowed for a single GPU buffer allocation in WebGPU.
# Apple WebGPU enforces ~1 GB per-buffer; we use 512 MB as a safe target
# so that quantized uint32 records stay well under the limit.
_MAX_EMBED_SHARD_BYTES = 512 * 1024 * 1024


class Gemma4SplitScaledEmbedding(nn.Module):
    """Per-layer embedding split into shards along the embedding dimension.

    WebGPU has per-buffer allocation limits (~1 GB on Apple, similar on other
    vendors).  Gemma 4's ``embed_tokens_per_layer`` weight is ~1.17 GB after
    quantisation, which exceeds that limit.  This module splits the single
    embedding table into N smaller shards so that each shard's quantised weight
    fits comfortably within the GPU buffer budget.

    The split is along the *embedding* dimension (axis 1), not the vocab
    dimension.  Because ``per_layer_input_total_size == num_hidden_layers *
    hidden_size_per_layer_input``, a clean split boundary every K layers keeps
    each shard aligned to the per-layer stride.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        embed_scale: float,
        hidden_size_per_layer_input: int,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embed_scale = embed_scale

        # Compute how many shards we need so each stays under the byte budget.
        # After q4f16_1 quantisation the storage ratio is
        #   uint32_cols = float_cols / 8   (4-bit packing into 32-bit words)
        # and each element is 4 bytes (uint32).
        quant_ratio = 4 / 8  # bytes-per-float-dim after q4f16 packing
        total_bytes = num_embeddings * embedding_dim * quant_ratio
        n_shards = max(1, int(-(-total_bytes // _MAX_EMBED_SHARD_BYTES)))  # ceil div

        # Round shard boundaries to multiples of hidden_size_per_layer_input so
        # that each shard covers a whole number of layers.
        stride = hidden_size_per_layer_input
        n_layers_total = embedding_dim // stride
        layers_per_shard_base = n_layers_total // n_shards
        remainder = n_layers_total % n_shards

        self.shard_dims: List[int] = []
        for i in range(n_shards):
            n_layers = layers_per_shard_base + (1 if i < remainder else 0)
            self.shard_dims.append(n_layers * stride)

        self.shards = nn.ModuleList([
            Gemma4ScaledEmbedding(num_embeddings, dim, embed_scale)
            for dim in self.shard_dims
        ])

    def forward(self, input_ids: Tensor) -> Tensor:
        parts = [shard(input_ids) for shard in self.shards]
        return op.concat(parts, dim=-1)


class Gemma4RMSNormNoScale(nn.Module):
    """Gemma4 RMSNorm variant without a learned scale."""

    def __init__(self, hidden_size: int, eps: float):
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, hidden_states: Tensor):
        weight = op.ones((self.hidden_size,), dtype=hidden_states.dtype)
        return op.rms_norm(hidden_states, weight=weight, axes=[-1], epsilon=self.eps)


class Gemma4MLP(nn.Module):
    """Gemma4 MLP with optional double-width hidden size on shared-KV layers."""

    def __init__(self, config: Gemma4Config, layer_idx: int):
        super().__init__()
        hidden_size = config.text_config.hidden_size
        intermediate_size = config.text_config.intermediate_size
        if config.text_config.layer_uses_double_wide_mlp(layer_idx):
            intermediate_size *= 2
        if intermediate_size % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split MLP intermediate size {intermediate_size} evenly to "
                f"{config.tensor_parallel_shards} GPUs."
            )
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size // config.tensor_parallel_shards
        self.gate_up_proj = nn.Linear(
            in_features=hidden_size,
            out_features=2 * self.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            in_features=self.intermediate_size,
            out_features=hidden_size,
            bias=False,
        )
        self.act_fn = ACT2FN[config.text_config.hidden_activation]

    def forward(self, hidden_states: Tensor):
        concat_gate_up = self.gate_up_proj(hidden_states)
        gate, up = op.split(concat_gate_up, 2, axis=-1)
        return self.down_proj(self.act_fn(gate) * up)


class Gemma4Attention(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Gemma4 text attention with layer-type-specific head dimensions and shared KV reuse."""

    def __init__(self, config: Gemma4Config, layer_idx: int):
        self.config = config
        self.text_config = config.text_config
        self.layer_idx = layer_idx
        self.layer_type = self.text_config.layer_type(layer_idx)
        self.local_head_dim = self.text_config.layer_head_dim(layer_idx)
        self.max_head_dim = self.text_config.max_head_dim
        self.shared_cache_source = self.text_config.layer_cache_source(layer_idx)
        self.rope_theta = self.text_config.layer_rope_theta(layer_idx)
        self.rope_scaling = self.text_config.layer_rope_scaling(layer_idx)
        self.scaling = 1.0

        self.num_q_heads = self.text_config.num_attention_heads // config.tensor_parallel_shards
        num_kv_heads = self.text_config.layer_num_key_value_heads(layer_idx)
        assert (
            num_kv_heads % config.tensor_parallel_shards == 0
        ), f"num_kv_heads({num_kv_heads}) must be divisible by tensor_parallel_shards"
        assert (
            num_kv_heads >= config.tensor_parallel_shards
        ), f"Too large tensor_parallel_shards, must be smaller than {num_kv_heads}"
        self.num_kv_heads = num_kv_heads // config.tensor_parallel_shards

        hidden_size = self.text_config.hidden_size
        self.q_proj = nn.Linear(
            in_features=hidden_size,
            out_features=self.num_q_heads * self.local_head_dim,
            bias=self.text_config.attention_bias,
        )
        self.k_proj = nn.Linear(
            in_features=hidden_size,
            out_features=self.num_kv_heads * self.local_head_dim,
            bias=self.text_config.attention_bias,
        )
        self.v_proj = nn.Linear(
            in_features=hidden_size,
            out_features=self.num_kv_heads * self.local_head_dim,
            bias=self.text_config.attention_bias,
        )
        self.o_proj = nn.Linear(
            in_features=self.num_q_heads * self.local_head_dim,
            out_features=hidden_size,
            bias=self.text_config.attention_bias,
        )
        self.q_norm = nn.RMSNorm(
            self.local_head_dim,
            -1,
            self.text_config.rms_norm_eps,
            bias=False,
        )
        self.k_norm = nn.RMSNorm(
            self.local_head_dim,
            -1,
            self.text_config.rms_norm_eps,
            bias=False,
        )
        self.v_norm = Gemma4RMSNormNoScale(self.local_head_dim, self.text_config.rms_norm_eps)

        def _set_tp():
            def _set(layer, hint):
                layer.weight.attrs["shard_strategy"] = hint

            q = self.num_q_heads * self.local_head_dim
            k = self.num_kv_heads * self.local_head_dim
            v = self.num_kv_heads * self.local_head_dim
            _set(self.q_proj, tp.ShardSingleDim(f"_shard_q_{layer_idx}", dim=0))
            _set(self.k_proj, tp.ShardSingleDim(f"_shard_k_{layer_idx}", dim=0))
            _set(self.v_proj, tp.ShardSingleDim(f"_shard_v_{layer_idx}", dim=0))
            _set(self.q_norm, tp.ShardSingleDim(f"_shard_q_norm_{layer_idx}", dim=0))
            _set(self.k_norm, tp.ShardSingleDim(f"_shard_k_norm_{layer_idx}", dim=0))
            _set(self.o_proj, tp.ShardSingleDim(f"_shard_o_{layer_idx}", dim=1))
            _set(
                self.q_proj,
                tp.ShardSingleDim(f"_shard_q_{layer_idx}", segs=[q], dim=0),
            )
            _set(
                self.k_proj,
                tp.ShardSingleDim(f"_shard_k_{layer_idx}", segs=[k], dim=0),
            )
            _set(
                self.v_proj,
                tp.ShardSingleDim(f"_shard_v_{layer_idx}", segs=[v], dim=0),
            )

        _set_tp()

    def _apply_rope(self, q: Tensor, k: Tensor, v: Tensor, cache_positions: Tensor):
        b, s, _, _ = q.shape
        fused_qkv = op.concat([q, k, v], dim=2)
        fused_qkv = op.reshape(
            fused_qkv,
            (b * s, self.num_q_heads + 2 * self.num_kv_heads, self.local_head_dim),
        )
        # Compute rotary_dim from partial_rotary_factor if present.
        # Full-attention layers use partial_rotary_factor=0.25 (rotate only 25% of dims).
        # Sliding-attention layers rotate all dimensions (partial_rotary_factor=1.0 or absent).
        partial_rotary_factor = self.rope_scaling.get("partial_rotary_factor", 1.0)
        rotary_dim = int(partial_rotary_factor * self.local_head_dim)
        rotary = position_embedding.llama_rope_with_position_map(
            theta=self.rope_theta,
            scale=1.0,
            head_dim=self.local_head_dim,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            dtype=q.dtype,
            rope_scaling=self.rope_scaling,
            rotary_dim=rotary_dim,
        )
        apply_rope = tirx.IntImm("int64", 1)
        q_out, k_out, v_out = op.tensor_ir_op(
            rotary,
            "llama_rope_with_position_map",
            args=[fused_qkv, cache_positions, apply_rope],
            out=(
                Tensor.placeholder((b * s, self.num_q_heads, self.local_head_dim), q.dtype),
                Tensor.placeholder((b * s, self.num_kv_heads, self.local_head_dim), q.dtype),
                Tensor.placeholder((b * s, self.num_kv_heads, self.local_head_dim), q.dtype),
            ),
        )
        return (
            op.reshape(q_out, (b, s, self.num_q_heads, self.local_head_dim)),
            op.reshape(k_out, (b, s, self.num_kv_heads, self.local_head_dim)),
            op.reshape(v_out, (b, s, self.num_kv_heads, self.local_head_dim)),
        )

    def _pad_head_dim(self, x: Tensor, num_heads: int):
        if self.local_head_dim == self.max_head_dim:
            return x
        pad_width = self.max_head_dim - self.local_head_dim
        padding = op.zeros((x.shape[0], x.shape[1], num_heads, pad_width), dtype=x.dtype)
        return op.concat([x, padding], dim=3)

    def _crop_output_head_dim(self, x: Tensor):
        if self.local_head_dim == self.max_head_dim:
            return op.reshape(x, (x.shape[0], x.shape[1], self.num_q_heads * self.local_head_dim))
        # The attention output has layout [batch, seq, num_q_heads, max_head_dim].
        # Crop each head's dim from max_head_dim back to local_head_dim, then flatten.
        x = op.reshape(x, (x.shape[0], x.shape[1], self.num_q_heads, self.max_head_dim))
        x, _ = op.split(x, [self.local_head_dim], axis=3)
        return op.reshape(x, (x.shape[0], x.shape[1], self.num_q_heads * self.local_head_dim))

    def forward(
        self,
        hidden_states: Tensor,
        paged_kv_cache: PagedKVCache,
        cache_positions: Tensor,
    ):
        b, s, _ = hidden_states.shape
        q = op.reshape(
            self.q_proj(hidden_states),
            (b, s, self.num_q_heads, self.local_head_dim),
        )
        k = op.reshape(
            self.k_proj(hidden_states),
            (b, s, self.num_kv_heads, self.local_head_dim),
        )
        v = op.reshape(
            self.v_proj(hidden_states),
            (b, s, self.num_kv_heads, self.local_head_dim),
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        q, k, v = self._apply_rope(q, k, v, cache_positions)
        q = self._pad_head_dim(q, self.num_q_heads)
        k = self._pad_head_dim(k, self.num_kv_heads)
        v = self._pad_head_dim(v, self.num_kv_heads)

        if self.shared_cache_source is None:
            # Re-fuse Q/K/V after the manual RoPE + head-dim padding and route
            # through ``attention_with_fused_qkv``, which performs the KV append
            # AND the cross-attention against previously-cached tokens in the
            # same call.  The plain ``self_attention`` entry point only covers
            # ragged self-attention on the new tokens and would skip cache
            # interaction, producing incorrect output at decode time.
            fused_qkv = op.concat([q, k, v], dim=2)  # [b, s, 10, max_head_dim]
            output = paged_kv_cache.attention_with_fused_qkv(
                self.layer_idx,
                fused_qkv,
                num_qo_heads=self.num_q_heads,
                sm_scale=self.scaling,
            )
        else:
            output, _ = paged_kv_cache.cross_attention(
                self.shared_cache_source,
                q=q,
                v_head_dim=self.max_head_dim,
                sm_scale=self.scaling,
            )

        output = self._crop_output_head_dim(output)
        return self.o_proj(output)


class Gemma4DecoderLayer(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Gemma4 decoder layer with per-layer inputs and layer scalar."""

    def __init__(self, config: Gemma4Config, layer_idx: int):
        self.layer_idx = layer_idx
        self.config = config
        self.text_config = config.text_config
        self.hidden_size = self.text_config.hidden_size
        self.self_attn = Gemma4Attention(config, layer_idx)
        self.mlp = Gemma4MLP(config, layer_idx)
        self.input_layernorm = nn.RMSNorm(
            self.hidden_size,
            -1,
            self.text_config.rms_norm_eps,
            bias=False,
        )
        self.post_attention_layernorm = nn.RMSNorm(
            self.hidden_size,
            -1,
            self.text_config.rms_norm_eps,
            bias=False,
        )
        self.pre_feedforward_layernorm = nn.RMSNorm(
            self.hidden_size,
            -1,
            self.text_config.rms_norm_eps,
            bias=False,
        )
        self.post_feedforward_layernorm = nn.RMSNorm(
            self.hidden_size,
            -1,
            self.text_config.rms_norm_eps,
            bias=False,
        )
        # WebGPU requires storage buffers to be ≥4 bytes and a multiple of 4.
        # A single float16 scalar is only 2 bytes, which triggers validation
        # errors.  Pad to (2,) so the buffer is 4 bytes.  Only element [0] is
        # used; element [1] is always zero.
        self.layer_scalar = nn.Parameter((2,))

        self.hidden_size_per_layer_input = self.text_config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.act_fn = ACT2FN[self.text_config.hidden_activation]
            self.per_layer_input_gate = nn.Linear(
                self.hidden_size,
                self.hidden_size_per_layer_input,
                bias=False,
                )
            self.per_layer_projection = nn.Linear(
                self.hidden_size_per_layer_input,
                self.hidden_size,
                bias=False,
                )
            self.post_per_layer_input_norm = nn.RMSNorm(
                self.hidden_size,
                -1,
                self.text_config.rms_norm_eps,
                bias=False,
            )

        def _set_tp():
            def _set(layer, hint):
                layer.weight.attrs["shard_strategy"] = hint

            i = self.mlp.intermediate_size
            _set(
                self.mlp.gate_up_proj,
                tp.ShardSingleDim(f"_shard_mlp_up_{layer_idx}", segs=[i, i], dim=0),
            )
            _set(
                self.mlp.down_proj,
                tp.ShardSingleDim(f"_shard_mlp_down_{layer_idx}", dim=1),
            )
            if self.hidden_size_per_layer_input:
                _set(
                    self.per_layer_input_gate,
                    tp.ShardSingleDim(f"_shard_per_layer_gate_{layer_idx}", dim=0),
                )
                _set(
                    self.per_layer_projection,
                    tp.ShardSingleDim(f"_shard_per_layer_proj_{layer_idx}", dim=1),
                )

        self.tensor_parallel_shards = config.tensor_parallel_shards
        _set_tp()

    def _apply_residual(self, out: Tensor, residual: Tensor):
        if self.tensor_parallel_shards > 1:
            return op.ccl_allreduce(out, "sum") + residual
        return out + residual

    def forward(
        self,
        hidden_states: Tensor,
        per_layer_input: Optional[Tensor],
        paged_kv_cache: PagedKVCache,
        cache_positions: Tensor,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, paged_kv_cache, cache_positions)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self._apply_residual(hidden_states, residual)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = self._apply_residual(hidden_states, residual)

        if self.hidden_size_per_layer_input:
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = self.act_fn(hidden_states)
            hidden_states = hidden_states * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = self._apply_residual(hidden_states, residual)

        scalar, _ = op.split(self.layer_scalar, [1], axis=0)
        return hidden_states * scalar


class Gemma4TextModel(nn.Module):
    """Gemma4 text backbone."""

    def __init__(self, config: Gemma4Config):
        self.config = config
        self.text_config = config.text_config
        self.hidden_size = self.text_config.hidden_size
        self.num_hidden_layers = self.text_config.num_hidden_layers
        self.hidden_size_per_layer_input = self.text_config.hidden_size_per_layer_input
        self.input_embed_size = self.text_config.input_embed_size

        self.embed_tokens = Gemma4ScaledEmbedding(
            "vocab_size",
            self.hidden_size,
            embed_scale=self.hidden_size**0.5,
        )
        self.layers = nn.ModuleList(
            [Gemma4DecoderLayer(config, layer_idx) for layer_idx in range(self.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(
            self.hidden_size,
            -1,
            self.text_config.rms_norm_eps,
            bias=False,
        )

        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = Gemma4SplitScaledEmbedding(
                self.text_config.vocab_size_per_layer_input,
                self.text_config.per_layer_input_total_size,
                embed_scale=self.hidden_size_per_layer_input**0.5,
                hidden_size_per_layer_input=self.hidden_size_per_layer_input,
            )
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_model_projection = nn.Linear(
                self.hidden_size,
                self.text_config.per_layer_input_total_size,
                bias=False,
                )
            self.per_layer_model_projection_scale = self.hidden_size**-0.5
            self.per_layer_projection_norm = nn.RMSNorm(
                self.hidden_size_per_layer_input,
                -1,
                self.text_config.rms_norm_eps,
                bias=False,
            )

    def split_input_embed(self, input_embed: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        if not self.hidden_size_per_layer_input:
            return input_embed, None
        input_embed, per_layer_input = op.split(input_embed, [self.hidden_size], axis=2)
        per_layer_input = op.reshape(
            per_layer_input,
            (
                input_embed.shape[0],
                input_embed.shape[1],
                self.num_hidden_layers,
                self.hidden_size_per_layer_input,
            ),
        )
        return input_embed, per_layer_input

    def project_per_layer_inputs(
        self,
        inputs_embeds: Tensor,
        per_layer_inputs: Optional[Tensor],
    ) -> Tensor:
        projection = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        projection = op.reshape(
            projection,
            (
                inputs_embeds.shape[0],
                inputs_embeds.shape[1],
                self.num_hidden_layers,
                self.hidden_size_per_layer_input,
            ),
        )
        projection = self.per_layer_projection_norm(projection)
        if per_layer_inputs is None:
            return projection
        return (projection + per_layer_inputs) * self.per_layer_input_scale

    def forward(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        hidden_states, per_layer_inputs = self.split_input_embed(input_embed)
        layer_per_inputs: Optional[Tuple[Tensor, ...]] = None
        if self.hidden_size_per_layer_input:
            per_layer_inputs = self.project_per_layer_inputs(hidden_states, per_layer_inputs)
            layer_per_inputs = op.split(per_layer_inputs, self.num_hidden_layers, axis=2)
        cache_positions = paged_kv_cache.get_query_positions(
            hidden_states.shape[0] * hidden_states.shape[1]
        )
        for layer_idx, layer in enumerate(self.layers):
            layer_per_input = None
            if layer_per_inputs is not None:
                layer_per_input = op.squeeze(layer_per_inputs[layer_idx], axis=2)
            hidden_states = layer(
                hidden_states,
                layer_per_input,
                paged_kv_cache,
                cache_positions,
            )
        return self.norm(hidden_states)


class Gemma4LanguageModel(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Gemma4 language model wrapper used by the MLC runtime."""

    def __init__(self, config: Gemma4Config):
        self.config = config
        self.text_config = config.text_config
        self.model = Gemma4TextModel(config)
        self.num_hidden_layers = self.text_config.num_hidden_layers
        self.num_attention_heads = self.text_config.num_attention_heads
        self.num_key_value_heads = self.text_config.max_num_key_value_heads
        self.head_dim = self.text_config.max_head_dim
        self.hidden_size = self.text_config.hidden_size
        self.input_embed_size = self.text_config.input_embed_size
        self.vocab_size = config.vocab_size
        self.rope_theta = self.text_config.global_position_embedding_base
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.dtype = "float32"
        # The spec delegates to this class (not Gemma4ForCausalLM), so logit
        # softcapping must be applied here to be included in the compiled module.
        self.final_logit_softcapping = config.text_config.final_logit_softcapping

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def get_logits(self, hidden_states: Tensor):
        logits = self.model.embed_tokens.lm_head_forward(hidden_states)
        # Gemma 4's embed forward multiplies the lookup result by
        # ``sqrt(hidden_size)``. TVM's quantize-embedding fusion absorbs the
        # lookup into a fused ``dequantize+take`` kernel that drops the
        # post-lookup multiply, so ``gemma4_loader.py`` pre-scales the tied
        # embedding weight by ``sqrt(hidden_size)`` at conversion time.
        # Because ``embed_tokens`` and ``lm_head`` share that weight, the
        # lm_head projection is scaled by the same factor; divide it out
        # here so the final logits match the unscaled reference.
        logits = logits / self.text_config.hidden_size**0.5
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        if self.final_logit_softcapping is not None:
            logits = op.tanh(logits / self.final_logit_softcapping) * self.final_logit_softcapping
        return logits

    def batch_forward(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        logit_positions: Optional[Tensor] = None,
    ):
        op_ext.configure()
        hidden_states = self.model(input_embeds, paged_kv_cache)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        return self.get_logits(hidden_states)

    def embed(self, input_ids: Tensor):
        if self.tensor_parallel_shards > 1:
            input_ids = op.ccl_broadcast_from_worker0(input_ids)
        input_embeds = self.model.embed_tokens(input_ids)
        if not self.text_config.hidden_size_per_layer_input:
            return input_embeds
        per_layer_inputs = self.model.embed_tokens_per_layer(input_ids)
        # `embed` is exported with a 1-D `input_ids` (see `get_default_spec`), so
        # both embedding outputs are 2-D and the feature axis is `dim=1`. The
        # result is later reshaped to `[1, seq_len, hidden + per_layer_total]`
        # by the caller, where `split_input_embed` splits it back apart on
        # `axis=2`.
        return op.concat([input_embeds, per_layer_inputs], dim=1)

    def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        def _index(x: te.Tensor):
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.model(input_embed, paged_kv_cache)
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        logits = self.get_logits(hidden_states)
        return logits, paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()
        hidden_states = self.model(input_embed, paged_kv_cache)
        logits = self.get_logits(hidden_states)
        return logits, paged_kv_cache

    def batch_prefill(
        self,
        input_embeds: Tensor,
        logit_positions: Tensor,
        paged_kv_cache: PagedKVCache,
    ):
        if self.tensor_parallel_shards > 1:
            logit_positions = op.ccl_broadcast_from_worker0(logit_positions)
        logits = self.batch_forward(input_embeds, paged_kv_cache, logit_positions)
        return logits, paged_kv_cache

    def batch_decode(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        logits = self.batch_forward(input_embeds, paged_kv_cache)
        return logits, paged_kv_cache

    def batch_verify(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        logits = self.batch_forward(input_embeds, paged_kv_cache)
        return logits, paged_kv_cache

    def create_paged_kv_cache(  # pylint: disable=too-many-arguments
        self,
        max_batch_size: tirx.Var,
        max_total_seq_len: tirx.Var,
        prefill_chunk_size: tirx.Var,
        page_size: tirx.Var,
        support_sliding_window: tirx.Var,
    ) -> PagedKVCache:
        return PagedKVCache.create_generic(
            attn_kind=[
                # Shared-KV layers only do cross-attention (no sliding window),
                # so they must use "mha" to avoid the MLA dispatch path in
                # paged_kv_cache.cc CrossAttention, which requires f_mla_prefill_.
                "mha"
                if layer_type == "full_attention"
                or self.text_config.layer_uses_shared_cache(layer_idx)
                else "mha_sliding"
                for layer_idx, layer_type in enumerate(self.text_config.layer_types)
            ],
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads // self.tensor_parallel_shards,
            # Use max_num_key_value_heads because every layer's KV is padded to
            # max_head_dim before entering the cache (see _pad_head_dim).
            num_key_value_heads=self.text_config.max_num_key_value_heads // self.tensor_parallel_shards,
            # Attention layers pad Q/K/V to max_head_dim before cache ops,
            # so the cache must be sized for the padded dimension.
            qk_head_dim=self.text_config.max_head_dim,
            v_head_dim=self.text_config.max_head_dim,
            rope_mode=RopeMode.NONE,
            rope_scale=1,
            rope_theta=self.rope_theta,
            dtype=self.dtype,
        )

    def get_default_spec(self):
        mod_spec = {
            "embed": {
                "input_ids": nn.spec.Tensor(["seq_len"], "int32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "prefill": {
                "input_embed": nn.spec.Tensor([1, "seq_len", self.input_embed_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode": {
                "input_embed": nn.spec.Tensor([1, 1, self.input_embed_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.input_embed_size], self.dtype),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_decode": {
                "input_embeds": nn.spec.Tensor(["batch_size", 1, self.input_embed_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_verify": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.input_embed_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "create_paged_kv_cache": {
                "max_batch_size": int,
                "max_total_seq_len": int,
                "prefill_chunk_size": int,
                "page_size": int,
                "support_sliding_window": int,
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)


class Gemma4ForCausalLM(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Gemma4 text-only causal LM for mlc-llm."""

    def __init__(self, config: Gemma4Config):
        issues = config.text_config.unsupported_runtime_features()
        if issues:
            raise NotImplementedError(
                "Gemma4 config parsed successfully, but native mlc-llm runtime support is still "
                f"missing for: {'; '.join(issues)}."
            )
        self.config = config
        self.language_model = Gemma4LanguageModel(config)
        self.vocab_size = config.vocab_size
        self.dtype = "float32"
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.final_logit_softcapping = config.text_config.final_logit_softcapping

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        self.language_model.to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def get_logits(self, hidden_states: Tensor):
        # Softcapping is applied inside language_model.get_logits (since the
        # TVM spec delegates to Gemma4LanguageModel, not this class).
        return self.language_model.get_logits(hidden_states)

    def batch_forward(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        logit_positions: Optional[Tensor] = None,
    ):
        op_ext.configure()
        hidden_states = self.language_model.model(input_embeds, paged_kv_cache)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        return self.get_logits(hidden_states)

    def embed(self, input_ids: Tensor):
        return self.language_model.embed(input_ids)

    def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        def _index(x: te.Tensor):
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.language_model.model(input_embed, paged_kv_cache)
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        logits = self.get_logits(hidden_states)
        return logits, paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()
        hidden_states = self.language_model.model(input_embed, paged_kv_cache)
        logits = self.get_logits(hidden_states)
        return logits, paged_kv_cache

    def batch_prefill(
        self,
        input_embeds: Tensor,
        logit_positions: Tensor,
        paged_kv_cache: PagedKVCache,
    ):
        if self.tensor_parallel_shards > 1:
            logit_positions = op.ccl_broadcast_from_worker0(logit_positions)
        logits = self.batch_forward(input_embeds, paged_kv_cache, logit_positions)
        return logits, paged_kv_cache

    def batch_decode(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        logits = self.batch_forward(input_embeds, paged_kv_cache)
        return logits, paged_kv_cache

    def batch_verify(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        logits = self.batch_forward(input_embeds, paged_kv_cache)
        return logits, paged_kv_cache

    def create_paged_kv_cache(  # pylint: disable=too-many-arguments
        self,
        max_batch_size: tirx.Var,
        max_total_seq_len: tirx.Var,
        prefill_chunk_size: tirx.Var,
        page_size: tirx.Var,
        support_sliding_window: tirx.Var,
    ) -> PagedKVCache:
        return self.language_model.create_paged_kv_cache(
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
        )

    def get_default_spec(self):
        return self.language_model.get_default_spec()
