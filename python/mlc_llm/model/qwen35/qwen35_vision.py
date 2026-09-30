"""Vision encoder for Qwen3.5: a ViT with 2D RoPE and 2 by 2 patch merging."""

import dataclasses
from typing import Any, Dict  # noqa: UP035

import numpy as np
from tvm import relax
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.model.vision.clip_vision import Linear
from mlc_llm.support.config import ConfigBase

# pylint: disable=invalid-name,missing-docstring,too-many-instance-attributes


@dataclasses.dataclass
class Qwen35VisionConfig(ConfigBase):
    """Configuration for Qwen3.5 vision encoder."""

    hidden_size: int = 1024
    num_heads: int = 16
    depth: int = 24
    intermediate_size: int = 4096
    patch_size: int = 16
    spatial_merge_size: int = 2
    out_hidden_size: int = 2560
    in_channels: int = 3
    num_position_embeddings: int = 2304
    hidden_act: str = "gelu_pytorch_tanh"
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)  # noqa: UP006

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


class Qwen35PatchEmbed(nn.Module):
    """Patch embedding as a Conv2D. The loader folds the reference Conv3D's temporal axis."""

    def __init__(self, config: Qwen35VisionConfig):
        self.proj = nn.Conv2D(
            in_channels=config.in_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True,
        )

    def forward(self, pixel_values: Tensor) -> Tensor:
        # Accumulate the patch projection in float32, see Linear in clip_vision.
        x = op.conv2d(
            pixel_values.astype("float32"),
            self.proj.weight.astype("float32"),
            self.proj.bias.astype("float32"),
            stride=self.proj.stride,
        ).astype(pixel_values.dtype)
        b, c, h, w = x.shape
        x = op.permute_dims(x, (0, 2, 3, 1))
        return op.reshape(x, (b, h * w, c))


class Qwen35VisionAttention(nn.Module):
    def __init__(self, config: Qwen35VisionConfig):
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.qkv = Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        b, seq_len, _ = hidden_states.shape
        shape = (b, seq_len, self.num_heads, self.head_dim)
        q, k, v = (op.reshape(x, shape) for x in op.split(self.qkv(hidden_states), 3, axis=-1))
        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)
        output = op_ext.attention(q, k, v, None)
        output = op.reshape(output, (b, seq_len, self.num_heads * self.head_dim))
        return self.proj(output)


class Qwen35VisionMLP(nn.Module):
    def __init__(self, config: Qwen35VisionConfig):
        self.fc1 = Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = Linear(config.intermediate_size, config.hidden_size)
        # The checkpoints use gelu_pytorch_tanh in the blocks; the merger keeps exact GELU.
        self.approximate = "tanh" if config.hidden_act == "gelu_pytorch_tanh" else None

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(op.gelu(self.fc1(x), approximate=self.approximate))


class Qwen35VisionBlock(nn.Module):
    def __init__(self, config: Qwen35VisionConfig):
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen35VisionAttention(config)
        self.mlp = Qwen35VisionMLP(config)

    def forward(self, hidden_states: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cos, sin)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen35PatchMerger(nn.Module):
    """Merge 2 by 2 patches and project to the text hidden size.

    The reference reorders patches into merge order before the blocks. The encoder here
    keeps raster order, which attention does not care about, and groups the patches here.
    """

    def __init__(self, config: Qwen35VisionConfig, grid: int):
        self.hidden_size = config.hidden_size
        self.merge_size = config.spatial_merge_size
        self.merge_dim = config.hidden_size * config.spatial_merge_size**2
        self.grid = grid
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.fc1 = Linear(self.merge_dim, self.merge_dim)
        self.fc2 = Linear(self.merge_dim, config.out_hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        b = x.shape[0]
        m, merged = self.merge_size, self.grid // self.merge_size
        x = self.norm(x)
        x = op.reshape(x, (b, merged, m, merged, m, self.hidden_size))
        x = op.permute_dims(x, (0, 1, 3, 2, 4, 5))
        x = op.reshape(x, (b, merged * merged, self.merge_dim))
        return self.fc2(op.gelu(self.fc1(x)))


class Qwen35VisionModel(nn.Module):
    """Qwen3.5 vision encoder at a fixed resolution, so every shape is static."""

    no_quantization: bool = True

    def __init__(self, config: Qwen35VisionConfig, image_size: int):
        grid = image_size // config.patch_size
        self.patch_embed = Qwen35PatchEmbed(config)
        self.pos_embed = nn.Parameter((grid * grid, config.hidden_size))
        self.blocks = nn.ModuleList([Qwen35VisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen35PatchMerger(config, grid)
        self.rope_cos, self.rope_sin = _precompute_2d_rope(grid, config.head_dim)

    def forward(self, pixel_values: Tensor) -> Tensor:
        hidden_states = self.patch_embed(pixel_values)
        hidden_states = hidden_states + op.reshape(self.pos_embed, (1, *self.pos_embed.shape))
        cos = nn.wrap_nested(relax.const(self.rope_cos, dtype="float32"), "rope_cos")
        sin = nn.wrap_nested(relax.const(self.rope_sin, dtype="float32"), "rope_sin")
        for block in self.blocks:
            hidden_states = block(hidden_states, cos, sin)
        return self.merger(hidden_states)


def _apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate x of shape (batch, seq, heads, dim) with cos and sin of shape (seq, dim)."""
    cos = op.reshape(cos, (1, cos.shape[0], 1, cos.shape[1]))
    sin = op.reshape(sin, (1, sin.shape[0], 1, sin.shape[1]))
    x1, x2 = op.split(x, 2, axis=-1)
    rotated = op.concat([op.negative(x2), x1], dim=-1)
    result = x.astype("float32") * cos + rotated.astype("float32") * sin
    return result.astype(x.dtype)


def _precompute_2d_rope(grid: int, head_dim: int) -> tuple:
    """Cos and sin of the 2D rotary angles for a square patch grid in raster order.

    Half of the rotary dimensions encode the row and half the column, each with the
    frequencies of a rotary embedding of dimension head_dim // 2, as the reference does.
    """
    dim = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    rows = np.repeat(np.arange(grid), grid)
    cols = np.tile(np.arange(grid), grid)
    angles = np.concatenate([np.outer(rows, inv_freq), np.outer(cols, inv_freq)], axis=-1)
    angles = np.concatenate([angles, angles], axis=-1)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
