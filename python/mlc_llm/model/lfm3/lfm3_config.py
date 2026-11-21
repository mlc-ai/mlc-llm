"""Configuration for LFM3's convolution and mixed-attention architecture."""

import dataclasses
from typing import Any, Dict  # noqa: UP035

from mlc_llm.model.lfm2.lfm2_model import Lfm2Config


@dataclasses.dataclass
class Lfm3Config(Lfm2Config):
    """LFM3 uses unadjusted FFN widths and separate RoPE settings per attention type."""

    intermediate_size: int = 7680
    block_auto_adjust_ff_dim: bool = False
    conv_L_cache: int = 7
    sliding_window: int = 256
    use_sandwich_norm: bool = True
    rope_parameters: Dict[str, Any] = dataclasses.field(default_factory=dict)  # noqa: UP006

    def __post_init__(self):
        if not self.layer_types or set(self.layer_types) - {
            "conv",
            "full_attention",
            "sliding_attention",
        }:
            raise ValueError("LFM3 requires explicit conv/full_attention/sliding_attention layers")
        if self.block_auto_adjust_ff_dim:
            raise ValueError("LFM3 uses the configured intermediate_size without adjustment")
        if self.sliding_window <= 0:
            raise ValueError("LFM3 sliding_window must be positive")
        if self.head_dim is None:
            self.head_dim = 0
        super().__post_init__()
        # The runtime's ragged prefill kernel applies only a causal mask within
        # each chunk. Keep that span inside the sliding window; the paged cache
        # applies the window mask when attending to tokens from previous chunks.
        if "sliding_attention" in self.layer_types:
            self.prefill_chunk_size = min(self.prefill_chunk_size, self.sliding_window)
        if self.head_dim % 2:
            raise ValueError("LFM3 RoPE requires an even head_dim")
        for layer_type in set(self.layer_types) - {"conv"}:
            rope = self.rope_parameters.get(layer_type)
            if not isinstance(rope, dict) or rope.get("rope_type", "default") != "default":
                raise ValueError(f"LFM3 requires default RoPE parameters for {layer_type}")
            if rope.get("rope_theta", 0) <= 0:
                raise ValueError("LFM3 rope_theta must be positive")
            fraction = rope.get("partial_rotary_factor", 1.0)
            if not 0 < fraction <= 1:
                raise ValueError("LFM3 partial_rotary_factor must be in (0, 1]")
            if int(fraction * self.head_dim) % 2:
                raise ValueError("LFM3 rotated dimensions must be even")

    @property
    def num_attention_layers(self) -> int:
        """Count both full and sliding attention in the shared KV cache."""
        return sum(t in ("full_attention", "sliding_attention") for t in self.layer_types)
