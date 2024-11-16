"""
Implements the CLIP Vision Encoder.
"""

import dataclasses
import logging
from typing import Any, Dict, Tuple

from tvm import relax
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Module, Tensor
from tvm.relax.frontend.nn.modules import Conv2D
from tvm.relax.frontend.nn.op import (
    add,
    broadcast_to,
    concat,
    permute_dims,
    reshape,
    wrap_nested,
)
from tvm.relax.op import arange

from mlc_llm import op as op_ext
from mlc_llm.support.config import ConfigBase

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class CLIPVisionConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """
    Config for the vision encoder
    """

    hidden_size: int
    image_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    patch_size: int
    projection_dim: int
    vocab_size: int
    num_channels: int = 3
    layer_norm_eps: float = 1e-06
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)


# pylint: disable=invalid-name,missing-docstring
class CLIPVisionEmbeddings(Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.class_embedding = nn.Parameter((self.embed_dim,))
        self.patch_embedding = Conv2D(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(num=self.num_positions, dim=self.embed_dim)

    def forward(self, pixel_values: Tensor) -> Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = reshape(patch_embeds, shape=(batch_size, self.embed_dim, -1))
        patch_embeds = permute_dims(
            patch_embeds, axes=(0, 2, 1)
        )  # shape = [batch,grid*grid,embed_dim]
        class_embeds = broadcast_to(
            self.class_embedding, shape=(batch_size, 1, self.embed_dim)
        )  # shape of (batch,1,embed_dim)
        embeddings = concat([class_embeds, patch_embeds], dim=1)

        posi_ids = reshape(
            wrap_nested(arange(0, self.num_positions, dtype="int32"), name="arange"), shape=(1, -1)
        )
        batch_position_embedding = broadcast_to(
            self.position_embedding(posi_ids),
            shape=(batch_size, self.num_positions, self.embed_dim),
        )
        embeddings = add(embeddings, batch_position_embedding)
        return embeddings


# pylint: disable=missing-docstring
def sigmoid(x: Tensor, name: str = "sigmoid") -> Tensor:
    """Sigmoid of a Tensor

    Parameters
    ----------
    x : Tensor
        Input tensor to expand.
    name : str
        Name hint for this operator.

    Returns
    -------
    result : Tensor
        Sigmoid result.
    """
    return wrap_nested(relax.op.sigmoid(x._expr), name)  # pylint: disable=protected-access


class QuickGELU(Module):
    def forward(self, input_tensor: Tensor) -> Tensor:
        return input_tensor * sigmoid(input_tensor * 1.702)


class CLIPMLP(Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.activation_fn = QuickGELU()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPAttention(Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tensor:
        d, h = self.head_dim, self.num_heads
        b, s, _ = hidden_states.shape  # batch_size, seq_len, embed_dim

        q = self.q_proj(hidden_states).reshape(b, s, h, d)
        k = self.k_proj(hidden_states).reshape(b, s, h, d)
        v = self.v_proj(hidden_states).reshape(b, s, h, d)

        attn_output = op_ext.attention(q, k, v, None)
        attn_output = self.out_proj(attn_output)
        return attn_output


class CLIPEncoderLayer(Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
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

        outputs = (hidden_states,)
        return outputs


class CLIPEncoder(Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)]
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


class CLIPVisionTransformer(Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        embed_dim = config.hidden_size
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: Tensor) -> Tensor:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states)
        return encoder_outputs


class CLIPVisionModel(Module):
    no_quantization: bool = True

    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.vision_model = CLIPVisionTransformer(config)

    def forward(self, pixel_values: Tensor) -> Tensor:
        return self.vision_model(pixel_values)[-2]
