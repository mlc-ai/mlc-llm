"""Implementation for Qwen2.5-VL architecture with MRoPE pre-rotation support."""

# pylint: disable=missing-function-docstring,missing-class-docstring

import dataclasses
from functools import partial
from typing import Any, Dict, Optional, Tuple

from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.op.mrope import (
    MultimodalRotaryEmbedding,
    VisionPositionMetadata,
    apply_multimodal_rotary_pos_emb,
)
from mlc_llm.support import tensor_parallel as tp
from mlc_llm.support.config import ConfigBase

ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.silu,
    "swish": nn.silu,
    "gelu_new": partial(nn.gelu, approximate=True),
}


@dataclasses.dataclass
class Qwen25VLVisionTokenConfig:
    """Vision token IDs used by Qwen2.5-VL."""

    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653


@dataclasses.dataclass
class Qwen25VLVisionGridConfig:
    """Vision grid configuration for multimodal position IDs."""

    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    tokens_per_second: float = 4.0


@dataclasses.dataclass(frozen=True)
class Qwen25VLAttentionState:
    """Derived attention dimensions used across Qwen2.5-VL attention code paths."""

    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    mrope_section: Tuple[int, int, int]
    softmax_scale: float


@dataclasses.dataclass
class Qwen25VLConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration for the Qwen2.5-VL model."""

    hidden_act: str
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int
    tie_word_embeddings: bool = False
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    head_dim: int = 0
    dtype: str = "float32"
    max_batch_size: int = 1
    rope_parameters: Optional[Dict[str, Any]] = None
    mrope_section: Optional[Tuple[int, int, int]] = None
    vision_tokens: Qwen25VLVisionTokenConfig = dataclasses.field(
        default_factory=Qwen25VLVisionTokenConfig
    )
    vision_grid: Qwen25VLVisionGridConfig = dataclasses.field(
        default_factory=Qwen25VLVisionGridConfig
    )
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):  # pylint: disable=too-many-branches
        if self.context_window_size == 0:
            for name in ["max_position_embeddings", "max_sequence_length"]:
                if name in self.kwargs:
                    self.context_window_size = self.kwargs.pop(name)
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.prefill_chunk_size == 0:
            self.prefill_chunk_size = min(self.context_window_size, 8192)
        elif self.prefill_chunk_size > self.context_window_size:
            self.prefill_chunk_size = min(self.context_window_size, 8192)

        rope_scaling = self.kwargs.pop("rope_scaling", None)
        if self.rope_parameters is None:
            self.rope_parameters = rope_scaling or {}
        if self.mrope_section is None:
            section = self.rope_parameters.get("mrope_section")
            if section is None and rope_scaling is not None:
                section = rope_scaling.get("mrope_section")
            if section is None:
                raise ValueError("`mrope_section` must be provided for Qwen2.5-VL.")
            self.mrope_section = tuple(int(i) for i in section)
        if len(self.mrope_section) != 3:
            raise ValueError(f"mrope_section must contain 3 integers, got {self.mrope_section}.")

        for key in [
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        ]:
            if key in self.kwargs:
                setattr(self.vision_tokens, key, int(self.kwargs.pop(key)))
        for key in ["spatial_merge_size", "temporal_patch_size"]:
            if key in self.kwargs:
                setattr(self.vision_grid, key, int(self.kwargs.pop(key)))
        if "tokens_per_second" in self.kwargs:
            self.vision_grid.tokens_per_second = float(self.kwargs.pop("tokens_per_second"))

        vision_cfg = self.kwargs.pop("vision_config", {})
        if vision_cfg:
            if not isinstance(vision_cfg, dict):
                raise ValueError(f"vision_config must be a dict, got {type(vision_cfg)}.")
            self.vision_grid.spatial_merge_size = int(
                vision_cfg.get("spatial_merge_size", self.vision_grid.spatial_merge_size)
            )
            self.vision_grid.temporal_patch_size = int(
                vision_cfg.get("temporal_patch_size", self.vision_grid.temporal_patch_size)
            )
            self.vision_grid.tokens_per_second = float(
                vision_cfg.get("tokens_per_second", self.vision_grid.tokens_per_second)
            )

    @property
    def image_token_id(self) -> int:
        return self.vision_tokens.image_token_id

    @property
    def video_token_id(self) -> int:
        return self.vision_tokens.video_token_id

    @property
    def vision_start_token_id(self) -> int:
        return self.vision_tokens.vision_start_token_id

    @property
    def vision_end_token_id(self) -> int:
        return self.vision_tokens.vision_end_token_id

    @property
    def spatial_merge_size(self) -> int:
        return self.vision_grid.spatial_merge_size

    @property
    def temporal_patch_size(self) -> int:
        return self.vision_grid.temporal_patch_size

    @property
    def tokens_per_second(self) -> float:
        return self.vision_grid.tokens_per_second

    @property
    def vision_metadata(self) -> VisionPositionMetadata:
        return VisionPositionMetadata(
            vision_start_token_id=self.vision_tokens.vision_start_token_id,
            image_token_id=self.vision_tokens.image_token_id,
            video_token_id=self.vision_tokens.video_token_id,
            spatial_merge_size=self.vision_grid.spatial_merge_size,
            tokens_per_second=self.vision_grid.tokens_per_second,
        )


class Qwen25VLEmbedding(nn.Embedding):
    """Embedding module shared with LM head."""

    def lm_head_forward(self, x: Tensor):
        weight = nn.op.permute_dims(self.weight)
        return nn.op.matmul(x, weight, out_dtype="float32")


class Qwen25VLAttention(nn.Module):
    def __init__(self, config: Qwen25VLConfig):
        if config.num_key_value_heads % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split {config.num_key_value_heads} key-value heads "
                f"evenly to {config.tensor_parallel_shards} shards."
            )
        head_dim = config.head_dim
        num_attention_heads = config.num_attention_heads // config.tensor_parallel_shards
        num_key_value_heads = config.num_key_value_heads // config.tensor_parallel_shards
        mrope_section: Tuple[int, int, int] = (
            config.mrope_section if config.mrope_section is not None else (0, 0, 0)
        )
        self.state = Qwen25VLAttentionState(
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            mrope_section=mrope_section,
            softmax_scale=head_dim**-0.5,
        )

        out_features = (num_attention_heads + 2 * num_key_value_heads) * head_dim
        self.c_attn = nn.Linear(
            in_features=config.hidden_size,
            out_features=out_features,
            bias=True,
        )
        self.o_proj = nn.Linear(
            num_attention_heads * head_dim,
            config.hidden_size,
            bias=False,
        )

    @property
    def head_dim(self) -> int:
        return self.state.head_dim

    @property
    def num_attention_heads(self) -> int:
        return self.state.num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.state.num_key_value_heads

    def forward(  # pylint: disable=too-many-locals
        self,
        hidden_states: Tensor,
        paged_kv_cache: PagedKVCache,
        layer_id: int,
        position_embeddings: Tuple[Tensor, Tensor],
    ):
        d, h_q, h_kv = (
            self.state.head_dim,
            self.state.num_attention_heads,
            self.state.num_key_value_heads,
        )
        b, s, _ = hidden_states.shape
        qkv = self.c_attn(hidden_states)
        qkv = op.reshape(qkv, (b, s, h_q + h_kv + h_kv, d))
        q, k, v = op.split(qkv, [h_q, h_q + h_kv], axis=2)
        cos, sin = position_embeddings
        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, self.state.mrope_section)
        output, _ = paged_kv_cache.self_attention(layer_id, q, k, v, self.state.softmax_scale)
        output = op.reshape(output, (b, s, h_q * d))
        return self.o_proj(output)


class Qwen25VLMLP(nn.Module):
    def __init__(self, config: Qwen25VLConfig):
        if config.intermediate_size % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split intermediate size {config.intermediate_size} "
                f"evenly to {config.tensor_parallel_shards} shards."
            )
        self.intermediate_size = config.intermediate_size // config.tensor_parallel_shards
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: Tensor):
        concat_x1_x2 = self.gate_up_proj(x)
        x1, x2 = op.split(concat_x1_x2, 2, axis=-1)
        return self.down_proj(self.act_fn(x1) * x2)


class Qwen25VLDecoderLayer(nn.Module):
    def __init__(self, config: Qwen25VLConfig):
        self.self_attn = Qwen25VLAttention(config)
        self.mlp = Qwen25VLMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, -1, config.rms_norm_eps, bias=False
        )

        self.tensor_parallel_shards = config.tensor_parallel_shards
        self._set_tp(config)

    def _set_tp(self, config: Qwen25VLConfig):
        def _set(layer, hint):
            layer.attrs["shard_strategy"] = hint

        hd = config.head_dim
        q = self.self_attn.num_attention_heads * hd
        k = self.self_attn.num_key_value_heads * hd
        v = self.self_attn.num_key_value_heads * hd
        i = self.mlp.intermediate_size
        _set(
            self.self_attn.c_attn.weight,
            tp.ShardSingleDim("_shard_qkv_weight", dim=0, segs=[q, k, v]),
        )
        _set(
            self.self_attn.c_attn.bias,
            tp.ShardSingleDim("_shard_qkv_bias", dim=0, segs=[q, k, v]),
        )
        _set(self.self_attn.o_proj.weight, tp.ShardSingleDim("_shard_o", dim=1))
        _set(
            self.mlp.gate_up_proj.weight,
            tp.ShardSingleDim("_shard_mlp_up", segs=[i, i], dim=0),
        )
        _set(self.mlp.down_proj.weight, tp.ShardSingleDim("_shard_mlp_down", dim=1))

    def forward(
        self,
        hidden_states: Tensor,
        paged_kv_cache: PagedKVCache,
        layer_id: int,
        position_embeddings: Tuple[Tensor, Tensor],
    ):
        out = self.input_layernorm(hidden_states)
        out = self.self_attn(out, paged_kv_cache, layer_id, position_embeddings)
        hidden_states = self._apply_residual(out, hidden_states)
        out = self.post_attention_layernorm(hidden_states)
        out = self.mlp(out)
        hidden_states = self._apply_residual(out, hidden_states)
        return hidden_states

    def _apply_residual(self, out: Tensor, residual: Tensor) -> Tensor:
        if self.tensor_parallel_shards > 1:
            return op.ccl_allreduce(out, "sum") + residual
        return out + residual


class Qwen25VLModel(nn.Module):
    def __init__(self, config: Qwen25VLConfig):
        self.embed_tokens = Qwen25VLEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen25VLDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        attention_scaling = config.rope_parameters.get("attention_scaling", 1.0)
        self.rotary_emb = MultimodalRotaryEmbedding(
            head_dim=config.head_dim,
            theta=config.rope_theta,
            mrope_section=config.mrope_section,
            attention_scaling=attention_scaling,
        )

    def forward(
        self,
        inputs: Tensor,
        position_ids: Tensor,
        paged_kv_cache: PagedKVCache,
    ):
        hidden_states = inputs
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        for layer_id, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, paged_kv_cache, layer_id, (cos, sin))
        return self.norm(hidden_states)


class Qwen25VLLMHeadModel(nn.Module):
    def __init__(self, config: Qwen25VLConfig):
        self.config = config
        self.model = Qwen25VLModel(config)
        self.tie_word_embeddings = config.tie_word_embeddings
        if not self.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.dtype = config.dtype

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def _apply_lm_head(self, hidden_states: Tensor):
        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.lm_head_forward(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def _set_mrope_delta(self, paged_kv_cache: PagedKVCache, deltas: Tensor):
        setattr(paged_kv_cache, "_mrope_delta", deltas)
        return deltas

    def _get_mrope_delta(self, paged_kv_cache: PagedKVCache, batch: int) -> Tensor:
        delta = getattr(paged_kv_cache, "_mrope_delta", None)
        if delta is None:
            delta = op.zeros((batch, 1), "int32")
            setattr(paged_kv_cache, "_mrope_delta", delta)
        return delta

    def _build_decode_position_ids(
        self,
        seq_len: int,
        paged_kv_cache: PagedKVCache,
        batch: int,
    ) -> Tensor:
        base = paged_kv_cache.get_query_positions(seq_len)
        base = op.reshape(base, (1, seq_len))
        base = op.broadcast_to(base, (batch, seq_len))
        delta = self._get_mrope_delta(paged_kv_cache, batch)
        base = base + delta
        base = op.unsqueeze(base, dim=0)
        return op.broadcast_to(base, (3, batch, seq_len))

    def prefill(
        self,
        input_embed: Tensor,
        position_ids: Tensor,
        mrope_deltas: Tensor,
        paged_kv_cache: PagedKVCache,
    ):
        op_ext.configure()
        self._set_mrope_delta(paged_kv_cache, mrope_deltas)
        hidden_states = self.model(input_embed, position_ids, paged_kv_cache)

        def _index(x: te.Tensor):
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        logits = self._apply_lm_head(hidden_states)
        return logits, paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()
        b, s, _ = input_embed.shape
        position_ids = self._build_decode_position_ids(s, paged_kv_cache, b)
        hidden_states = self.model(input_embed, position_ids, paged_kv_cache)
        logits = self._apply_lm_head(hidden_states)
        return logits, paged_kv_cache

    def batch_prefill(  # pylint: disable=too-many-arguments
        self,
        input_embeds: Tensor,
        position_ids: Tensor,
        mrope_deltas: Tensor,
        logit_positions: Tensor,
        paged_kv_cache: PagedKVCache,
    ):
        if self.config.tensor_parallel_shards > 1:
            logit_positions = op.ccl_broadcast_from_worker0(logit_positions)
        logits = self.batch_forward(
            input_embeds, position_ids, mrope_deltas, logit_positions, paged_kv_cache
        )
        return logits, paged_kv_cache

    def batch_forward(  # pylint: disable=too-many-arguments
        self,
        input_embeds: Tensor,
        position_ids: Tensor,
        mrope_deltas: Tensor,
        logit_positions: Optional[Tensor],
        paged_kv_cache: PagedKVCache,
    ):
        op_ext.configure()
        self._set_mrope_delta(paged_kv_cache, mrope_deltas)
        hidden_states = self.model(input_embeds, position_ids, paged_kv_cache)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        return self._apply_lm_head(hidden_states)

    def batch_decode(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()
        b, s, _ = input_embeds.shape
        position_ids = self._build_decode_position_ids(s, paged_kv_cache, b)
        hidden_states = self.model(input_embeds, position_ids, paged_kv_cache)
        logits = self._apply_lm_head(hidden_states)
        return logits, paged_kv_cache

    def batch_verify(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        return self.batch_decode(input_embeds, paged_kv_cache)

    def embed(self, input_ids: Tensor):
        if self.config.tensor_parallel_shards > 1:
            input_ids = op.ccl_broadcast_from_worker0(input_ids)
        return self.model.embed_tokens(input_ids)

    def create_paged_kv_cache(  # pylint: disable=too-many-arguments
        self,
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        prefill_chunk_size: tir.Var,
        page_size: tir.Var,
        support_sliding_window: tir.Var,
    ) -> PagedKVCache:
        cfg = self.config
        return PagedKVCache.create_generic(
            attn_kind="mha",
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
            num_hidden_layers=cfg.num_hidden_layers,
            num_attention_heads=cfg.num_attention_heads // cfg.tensor_parallel_shards,
            num_key_value_heads=cfg.num_key_value_heads // cfg.tensor_parallel_shards,
            qk_head_dim=cfg.head_dim,
            v_head_dim=cfg.head_dim,
            rope_mode=RopeMode.NORMAL,
            rope_scaling=cfg.rope_parameters,
            rope_scale=1,
            rope_theta=int(cfg.rope_theta),
            dtype=self.dtype,
        )

    def get_default_spec(self):
        cfg = self.config
        seq_len = "seq_len"
        hidden = cfg.hidden_size
        dtype = self.dtype
        mod_spec = {
            "embed": {
                "input_ids": nn.spec.Tensor([seq_len], "int32"),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "prefill": {
                "input_embed": nn.spec.Tensor([1, seq_len, hidden], dtype),
                "position_ids": nn.spec.Tensor([3, 1, seq_len], "int32"),
                "mrope_deltas": nn.spec.Tensor([1, 1], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "decode": {
                "input_embed": nn.spec.Tensor([1, 1, hidden], dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "batch_prefill": {
                "input_embeds": nn.spec.Tensor([1, seq_len, hidden], dtype),
                "position_ids": nn.spec.Tensor([3, 1, seq_len], "int32"),
                "mrope_deltas": nn.spec.Tensor([1, 1], "int32"),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "batch_decode": {
                "input_embeds": nn.spec.Tensor(["batch_size", 1, hidden], dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "batch_verify": {
                "input_embeds": nn.spec.Tensor(["batch_size", 1, hidden], dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "create_paged_kv_cache": {
                "max_batch_size": int,
                "max_total_seq_len": int,
                "prefill_chunk_size": int,
                "page_size": int,
                "support_sliding_window": int,
                "$": {"param_mode": "none", "effect_mode": "none"},
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
