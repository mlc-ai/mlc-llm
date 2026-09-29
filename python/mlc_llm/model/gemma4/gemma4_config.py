"""Configuration objects for the supported dense Gemma 4 variant."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict  # noqa: UP035

from mlc_llm.support import logging
from mlc_llm.support.config import ConfigBase

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Gemma4AudioConfig(ConfigBase):
    """Configuration of Gemma 4's USM-derived audio encoder."""

    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    hidden_act: str = "silu"
    subsampling_conv_channels: tuple[int, int] = (128, 32)
    conv_kernel_size: int = 5
    residual_weight: float = 0.5
    attention_chunk_size: int = 12
    attention_context_left: int = 13
    attention_context_right: int = 0
    attention_logit_cap: float = 50.0
    attention_invalid_logits_value: float = -1.0e9
    use_clipped_linears: bool = True
    rms_norm_eps: float = 1e-6
    gradient_clipping: float = 1e10
    output_proj_dims: int = 1536
    feature_size: int = 128
    sampling_rate: int = 16_000
    max_samples: int = 480_000
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)  # noqa: UP006

    def __post_init__(self) -> None:
        self.subsampling_conv_channels = tuple(self.subsampling_conv_channels)
        if self.hidden_act != "silu":
            raise ValueError("Gemma 4 audio currently requires SiLU activation")
        if (
            self.attention_chunk_size != 12
            or self.attention_context_left != 13
            or self.attention_context_right != 0
        ):
            raise ValueError("Only Gemma 4 E2B's 12-token causal audio attention is supported")
        if len(self.subsampling_conv_channels) != 2:
            raise ValueError("Gemma 4 audio requires two subsampling convolution layers")


@dataclasses.dataclass
class Gemma4TextConfig(ConfigBase):
    """Configuration of Gemma 4's dense text decoder."""

    vocab_size: int = 262_144
    hidden_size: int = 1536
    intermediate_size: int = 6144
    num_hidden_layers: int = 35
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 256
    global_head_dim: int = 512
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 131_072
    rms_norm_eps: float = 1e-6
    pad_token_id: int = 0
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    attention_dropout: float = 0.0
    sliding_window: int = 512
    layer_types: list[str] | None = None
    final_logit_softcapping: float | None = 30.0
    use_bidirectional_attention: str | None = None
    vocab_size_per_layer_input: int = 262_144
    hidden_size_per_layer_input: int = 256
    attention_k_eq_v: bool = False
    num_global_key_value_heads: int | None = None
    num_kv_shared_layers: int = 20
    enable_moe_block: bool = False
    use_double_wide_mlp: bool = True
    rope_parameters: dict[str, dict[str, Any]] | None = None
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)  # noqa: UP006

    def __post_init__(self) -> None:
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (index + 1) % 5 == 0 else "sliding_attention"
                for index in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must contain one entry per decoder layer")
        if self.rope_parameters is None:
            self.rope_parameters = {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
                "full_attention": {
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1_000_000.0,
                },
            }
        if self.hidden_activation not in ("gelu", "gelu_pytorch_tanh"):
            raise ValueError("Only GeLU is supported for Gemma 4 text")
        if self.attention_bias or self.attention_dropout:
            raise ValueError("Gemma 4 attention bias and dropout are not supported")
        if self.use_bidirectional_attention is not None:
            raise ValueError("The text+audio milestone supports causal attention only")
        if self.attention_k_eq_v or self.enable_moe_block:
            raise ValueError("The first Gemma 4 target is the dense E2B architecture")
        if not self.tie_word_embeddings:
            raise ValueError("Untied Gemma 4 language-model heads are not supported")
        if self.num_key_value_heads != 1 or self.num_attention_heads != 8:
            raise ValueError("The first Gemma 4 target requires 8 query heads and one KV head")
        if not 0 < self.num_kv_shared_layers < self.num_hidden_layers:
            raise ValueError("The first Gemma 4 target requires shared KV layers")
        if self.head_dim != 256 or self.global_head_dim != 512:
            raise ValueError("The first Gemma 4 target requires 256/512 local/global head dims")
        supported_layer_types = {"sliding_attention", "full_attention"}
        if set(self.layer_types) - supported_layer_types:
            raise ValueError("Gemma 4 only supports sliding_attention and full_attention layers")
        physical_layer_types = set(self.layer_types[: self.first_kv_shared_layer])
        shared_layer_types = set(self.layer_types[self.first_kv_shared_layer :])
        if not shared_layer_types.issubset(physical_layer_types):
            raise ValueError("Every shared-KV layer type requires a physical source layer")

    @property
    def first_kv_shared_layer(self) -> int:
        return self.num_hidden_layers - self.num_kv_shared_layers

    def head_dim_for_layer(self, layer_idx: int) -> int:
        return (
            self.global_head_dim
            if self.layer_types[layer_idx] == "full_attention"
            else self.head_dim
        )


@dataclasses.dataclass
class Gemma4Config(ConfigBase):
    """Gemma 4 text+audio configuration; vision is intentionally not instantiated."""

    text_config: Gemma4TextConfig | dict[str, Any] | None = None
    audio_config: Gemma4AudioConfig | dict[str, Any] | None = None
    vision_config: dict[str, Any] | None = None
    boa_token_id: int = 256_000
    audio_token_id: int = 258_881
    eoa_token_index: int = 258_883
    tensor_parallel_shards: int = 1
    max_batch_size: int = 1
    context_window_size: int = -1
    sliding_window_size: int = -1
    prefill_chunk_size: int = -1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)  # noqa: UP006

    def __post_init__(self) -> None:
        text_dict = _as_config_dict(self.text_config, self.kwargs)
        audio_dict = _as_config_dict(self.audio_config, {})
        self.text_config = Gemma4TextConfig.from_dict(text_dict)
        self.audio_config = Gemma4AudioConfig.from_dict(audio_dict)

        if self.tensor_parallel_shards != 1:
            raise ValueError("Gemma 4 text+audio currently supports one tensor-parallel shard")
        self.context_window_size = (
            self.text_config.max_position_embeddings
            if self.context_window_size <= 0
            else self.context_window_size
        )
        # The runtime cache owns Gemma 4's per-layer sliding policy.  Keep the
        # package-level setting disabled so mixed full-attention layers retain
        # the complete context instead of sizing the whole cache to 512 tokens.
        self.sliding_window_size = -1
        if self.prefill_chunk_size <= 0:
            self.prefill_chunk_size = self.text_config.sliding_window
        elif self.prefill_chunk_size > self.text_config.sliding_window:
            logger.info(
                "Clamping Gemma 4 prefill_chunk_size from %d to its strict sliding window %d",
                self.prefill_chunk_size,
                self.text_config.sliding_window,
            )
            self.prefill_chunk_size = self.text_config.sliding_window

    @property
    def vocab_size(self) -> int:
        """Vocabulary size consumed by the existing chat-config generator."""
        return self.text_config.vocab_size


def _as_config_dict(
    value: ConfigBase | dict[str, Any] | None, fallback: dict[str, Any]
) -> dict[str, Any]:
    if value is None:
        result = dict(fallback)
    elif dataclasses.is_dataclass(value):
        result = dataclasses.asdict(value)
    else:
        result = dict(value)
    result.update(result.pop("kwargs", {}))
    return result
