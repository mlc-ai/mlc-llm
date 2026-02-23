"""
Implements the SigLIP Vision Encoder.

Key differences from CLIP (clip_vision.py):
- No CLS token (num_positions = num_patches, not num_patches + 1)
- Uses standard GELU activation (not QuickGELU)
- Returns last hidden state through post_layernorm (not second-to-last)
- No pre_layernorm (only post_layernorm)
"""

import dataclasses
import logging
from typing import Any, Dict, Tuple

from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Module, Tensor
from tvm.relax.frontend.nn import op
from tvm.relax.frontend.nn.op import (
    add,
    broadcast_to,
    permute_dims,
    reshape,
    wrap_nested,
)
from tvm.relax.op import arange
from mlc_llm import op as op_ext
from mlc_llm.support.config import ConfigBase

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SigLIPVisionConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Config for the SigLIP vision encoder."""

    hidden_size: int
    image_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    patch_size: int
    num_channels: int = 3
    layer_norm_eps: float = 1e-06
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)


# pylint: disable=invalid-name,missing-docstring


class SigLIPVisionEmbeddings(Module):
    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        # No CLS token in SigLIP (unlike CLIP)
        self.patch_embedding = nn.modules.Conv2D(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches  # No +1 for CLS
        self.position_embedding = nn.Embedding(num=self.num_positions, dim=self.embed_dim)

    def forward(self, pixel_values: Tensor) -> Tensor:
        batch_size = pixel_values.shape[0]
        # pixel_values: (batch, channels, height, width)
        patch_embeds = self.patch_embedding(pixel_values)  # (batch, embed_dim, grid, grid)
        patch_embeds = reshape(patch_embeds, shape=(batch_size, self.embed_dim, -1))
        patch_embeds = permute_dims(
            patch_embeds, axes=(0, 2, 1)
        )  # (batch, num_patches, embed_dim)

        posi_ids = reshape(
            wrap_nested(arange(0, self.num_positions, dtype="int32"), name="arange"),
            shape=(1, -1),
        )
        batch_position_embedding = broadcast_to(
            self.position_embedding(posi_ids),
            shape=(batch_size, self.num_positions, self.embed_dim),
        )
        embeddings = add(patch_embeds, batch_position_embedding)
        return embeddings


class SigLIPMLP(Module):
    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.activation_fn = nn.GELU()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SigLIPAttention(Module):
    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        d, h = self.head_dim, self.num_heads
        b, s, _ = hidden_states.shape

        q = self.q_proj(hidden_states).reshape(b, s, h, d)
        k = self.k_proj(hidden_states).reshape(b, s, h, d)
        v = self.v_proj(hidden_states).reshape(b, s, h, d)

        attn_output = op_ext.attention(q, k, v, None)
        attn_output = self.out_proj(attn_output)
        return attn_output


class SigLIPEncoderLayer(Module):
    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SigLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SigLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return (hidden_states,)


class SigLIPEncoder(Module):
    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [SigLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, inputs_embeds: Tensor) -> Tensor:
        hidden_states = inputs_embeds
        encoder_states: Tuple[Any, ...] = ()
        for _, encoder_layer in enumerate(self.layers):
            encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(hidden_states)
            hidden_states = layer_outputs[0]
        encoder_states = encoder_states + (hidden_states,)
        return encoder_states


class SigLIPVisionTransformer(Module):
    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        embed_dim = config.hidden_size
        self.embeddings = SigLIPVisionEmbeddings(config)
        self.encoder = SigLIPEncoder(config)
        # SigLIP only has post_layernorm (no pre_layernorm unlike CLIP)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: Tensor) -> Tensor:
        hidden_states = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states)
        # Return last hidden state through post_layernorm
        last_hidden_state = encoder_outputs[-1]
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state


class SigLIPVisionModel(Module):
    no_quantization: bool = False

    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.vision_model = SigLIPVisionTransformer(config)

    def forward(self, pixel_values: Tensor) -> Tensor:
        return self.vision_model(pixel_values)
