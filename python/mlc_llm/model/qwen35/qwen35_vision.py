"""Vision encoder for Qwen3.5 VLM: custom ViT with 2D RoPE and patch merging."""

import dataclasses
import math
from typing import Any, Dict

import numpy as np

from tvm import relax
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
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
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    out_hidden_size: int = 2560
    in_channels: int = 3
    num_position_embeddings: int = 2304
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


class Qwen35PatchEmbed(nn.Module):
    """Conv2D patch embedding.

    HF uses nn.Conv3d with temporal_patch_size=2 for video support. For single images,
    HF duplicates the image into identical temporal frames, so summing the Conv3D weight
    over the temporal dimension gives an equivalent Conv2D. We do this sum in the weight
    loader to avoid needing a Conv3D op (no TVM/Metal support). See qwen35v_loader.py.
    """

    def __init__(self, config: Qwen35VisionConfig):
        self.hidden_size = config.hidden_size
        self.proj = nn.Conv2D(
            in_channels=config.in_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True,
        )

    def forward(self, pixel_values: Tensor) -> Tensor:
        # pixel_values: (1, C, H, W) -> conv2d -> (1, hidden, grid_h, grid_w)
        x = self.proj(pixel_values)
        # Reshape to (1, num_patches, hidden_size)
        b, c, h, w = x.shape
        x = op.permute_dims(x, (0, 2, 3, 1))  # (1, grid_h, grid_w, hidden)
        x = op.reshape(x, (b, h * w, c))  # (1, num_patches, hidden)
        return x


class Qwen35VisionAttention(nn.Module):
    def __init__(self, config: Qwen35VisionConfig):
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(
        self, hidden_states: Tensor, cos: Tensor, sin: Tensor
    ) -> Tensor:
        b, seq_len, _ = hidden_states.shape
        # Project QKV
        qkv = self.qkv(hidden_states)  # (1, seq, 3*hidden)
        qkv = op.reshape(qkv, (b, seq_len, 3, self.num_heads, self.head_dim))
        qkv = op.permute_dims(qkv, (2, 0, 1, 3, 4))  # (3, 1, seq, heads, head_dim)
        q, k, v = op.split(qkv, 3, axis=0)  # each (1, 1, seq, heads, head_dim)
        q = op.reshape(q, (b, seq_len, self.num_heads, self.head_dim))
        k = op.reshape(k, (b, seq_len, self.num_heads, self.head_dim))
        v = op.reshape(v, (b, seq_len, self.num_heads, self.head_dim))

        # Apply 2D RoPE
        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)

        # Non-causal attention via op_ext
        output = op_ext.attention(q, k, v, None)  # (1, seq, heads, head_dim)
        output = op.reshape(output, (b, seq_len, self.num_heads * self.head_dim))
        return self.proj(output)


class Qwen35VisionMLP(nn.Module):
    def __init__(self, config: Qwen35VisionConfig):
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(op.gelu(self.fc1(x)))


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
    """Merge 2x2 spatial patches and project to text hidden size.

    HF reorders patches into merge-order before the encoder blocks so the merger can
    simply group consecutive tokens. We instead keep raster order throughout the entire
    encoder (patches, position embeddings, 2D RoPE) and do the 2x2 spatial grouping
    explicitly here via reshape+permute. This is equivalent because vision self-attention
    is permutation-equivariant — as long as each patch gets the correct (row, col)
    position encoding, the attention output is the same regardless of sequence ordering.
    """

    def __init__(self, config: Qwen35VisionConfig, grid_h: int, grid_w: int):
        merge_dim = config.hidden_size * (config.spatial_merge_size**2)
        self.hidden_size = config.hidden_size
        self.merge_dim = merge_dim
        self.spatial_merge_size = config.spatial_merge_size
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.fc1 = nn.Linear(merge_dim, merge_dim, bias=True)
        self.fc2 = nn.Linear(merge_dim, config.out_hidden_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        # x: (1, grid_h * grid_w, hidden_size) in raster order
        b = x.shape[0]
        m = self.spatial_merge_size

        # Per-patch LayerNorm
        x = self.norm(x)

        # Reshape to spatial grid: (1, grid_h, grid_w, hidden)
        x = op.reshape(x, (b, self.grid_h, self.grid_w, self.hidden_size))

        # Group 2x2 blocks: (1, grid_h//2, 2, grid_w//2, 2, hidden)
        x = op.reshape(
            x, (b, self.grid_h // m, m, self.grid_w // m, m, self.hidden_size)
        )
        # Permute to (1, grid_h//2, grid_w//2, 2, 2, hidden)
        x = op.permute_dims(x, (0, 1, 3, 2, 4, 5))
        # Flatten merge dims: (1, merged_tokens, 4*hidden)
        merged_h = self.grid_h // m
        merged_w = self.grid_w // m
        x = op.reshape(x, (b, merged_h * merged_w, self.merge_dim))

        # Project
        x = self.fc2(op.gelu(self.fc1(x)))
        return x


class Qwen35VisionModel(nn.Module):
    """Qwen3.5 vision encoder with 2D RoPE, fixed resolution.

    Resolution is fixed at compile time (default 448x448) so all tensor shapes are
    concrete — no dynamic shapes needed, which simplifies TVM compilation and Metal
    codegen. Different resolutions can be used by changing image_size and recompiling.
    """

    no_quantization: bool = True

    def __init__(self, config: Qwen35VisionConfig, image_size: int):
        self.config = config
        self.image_size = image_size
        self.grid_h = image_size // config.patch_size
        self.grid_w = image_size // config.patch_size
        num_patches = self.grid_h * self.grid_w

        self.patch_embed = Qwen35PatchEmbed(config)
        self.pos_embed = nn.Parameter((num_patches, config.hidden_size))
        self.blocks = nn.ModuleList(
            [Qwen35VisionBlock(config) for _ in range(config.depth)]
        )
        self.merger = Qwen35PatchMerger(config, self.grid_h, self.grid_w)

        # Pre-compute 2D RoPE cos/sin as constants (raster order)
        cos_np, sin_np = _precompute_2d_rope(
            self.grid_h, self.grid_w, config.head_dim, theta=10000.0
        )
        self._rope_cos = cos_np  # stored for export
        self._rope_sin = sin_np

    def forward(self, pixel_values: Tensor) -> Tensor:
        # pixel_values: (1, C, H, W) after preprocessing
        hidden_states = self.patch_embed(pixel_values)  # (1, num_patches, hidden)

        # Add position embeddings (raster order)
        pos = op.reshape(self.pos_embed, (1, self.pos_embed.shape[0], self.pos_embed.shape[1]))
        hidden_states = hidden_states + pos

        # 2D RoPE cos/sin as constants
        cos = relax.const(self._rope_cos, dtype="float32")
        sin = relax.const(self._rope_sin, dtype="float32")
        cos = nn.wrap_nested(cos, "rope_cos")
        sin = nn.wrap_nested(sin, "rope_sin")

        # Vision transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, cos, sin)

        # Merge 2x2 patches and project to text space
        output = self.merger(hidden_states)
        return output


def _apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embedding to tensor x with shape (batch, seq, heads, dim).

    cos/sin shape: (seq, dim) — broadcast over batch and heads.
    Uses rotate_half: split dim in half, rotate, multiply by sin.
    """
    # cos/sin: (seq, dim) -> (1, seq, 1, dim)
    cos = op.reshape(cos, (1, cos.shape[0], 1, cos.shape[1]))
    sin = op.reshape(sin, (1, sin.shape[0], 1, sin.shape[1]))

    # rotate_half: [-x2, x1] where x1, x2 are first/second half of last dim
    x1, x2 = op.split(x, 2, axis=-1)
    rotated = op.concat([op.negative(x2), x1], dim=-1)

    # Cast to float32 for RoPE computation, then back
    orig_dtype = x.dtype
    x = x.astype("float32")
    rotated = rotated.astype("float32")
    result = x * cos + rotated * sin
    return result.astype(orig_dtype)


def _precompute_2d_rope(
    grid_h: int, grid_w: int, head_dim: int, theta: float = 10000.0
) -> tuple:
    """Pre-compute 2D RoPE cos/sin for vision patches in raster order.

    Matches HF's Qwen3_5VisionRotaryEmbedding + rot_pos_emb + apply logic:
    - VisionRotaryEmbedding(dim=head_dim//2) creates inv_freq with head_dim//4 elements
    - For each (row, col): freqs_h = row * inv_freq, freqs_w = col * inv_freq
    - rotary_emb = concat(freqs_h, freqs_w) -> head_dim//2 elements
    - full_emb = concat(rotary_emb, rotary_emb) -> head_dim elements
    - cos/sin of full_emb applied via rotate_half

    Returns:
        cos: (grid_h * grid_w, head_dim) float32 numpy array
        sin: (grid_h * grid_w, head_dim) float32 numpy array
    """
    dim = head_dim // 2  # VisionRotaryEmbedding dim
    inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    # inv_freq shape: (dim // 2,) = (head_dim // 4,)

    # Build position IDs in raster order
    rows = np.arange(grid_h)
    cols = np.arange(grid_w)
    row_ids = np.repeat(rows, grid_w)  # (num_patches,)
    col_ids = np.tile(cols, grid_h)  # (num_patches,)

    # Compute frequencies: outer product of position with inv_freq
    freqs_h = np.outer(row_ids, inv_freq)  # (num_patches, head_dim//4)
    freqs_w = np.outer(col_ids, inv_freq)  # (num_patches, head_dim//4)

    # rotary_emb = concat(freqs_h, freqs_w) -> (num_patches, head_dim//2)
    rotary_emb = np.concatenate([freqs_h, freqs_w], axis=-1)

    # full_emb = concat(rotary_emb, rotary_emb) -> (num_patches, head_dim)
    full_emb = np.concatenate([rotary_emb, rotary_emb], axis=-1)

    cos_all = np.cos(full_emb).astype(np.float32)
    sin_all = np.sin(full_emb).astype(np.float32)

    return cos_all, sin_all
