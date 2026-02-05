"""Utilities for Multimodal Rotary Position Embeddings (MRoPE)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from tvm import relax as rx
from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate the last dimension of ``x`` by swapping pairs."""

    x1, x2 = op.split(x, 2, axis=-1)
    return op.concat([op.negative(x2), x1], dim=-1)


def _repeat_mrope_section(section: Sequence[int]) -> Tuple[int, ...]:
    if not section:
        raise ValueError("mrope_section must not be empty.")
    if any(s <= 0 for s in section):
        raise ValueError(f"All mrope_section entries must be positive, got {section}.")
    return tuple(section) * 2


def _split_indices_from_sizes(sizes: Sequence[int]) -> List[int]:
    indices: List[int] = []
    running = 0
    # Drop the final cumulative sum so split() keeps the last chunk.
    for size in sizes[:-1]:
        running += size
        indices.append(running)
    return indices


def _reorder_cos_sin(
    tensor: Tensor,
    split_sizes: Sequence[int],
) -> Tensor:
    """Reorder cos/sin tensors so the head dimension follows T/H/W repeating sections."""

    if not split_sizes:
        raise ValueError("split_sizes must not be empty.")
    split_points = _split_indices_from_sizes(split_sizes)
    # relax.op.split returns a Python tuple, so we can iterate directly.
    sections = op.split(tensor, indices_or_sections=split_points, axis=-1)
    reordered = []
    for idx, chunk in enumerate(sections):
        axis_selector = nn.Tensor.from_const(np.array([idx % 3], dtype="int32"))
        axis_slice = op.take(chunk, axis_selector, axis=0)
        reordered.append(nn.op.squeeze(axis_slice, 0))
    return op.concat(reordered, dim=-1)


class MultimodalRotaryEmbedding(nn.Module):
    """Generate cosine/sine tables for multimodal rotary embeddings."""

    def __init__(
        self,
        head_dim: int,
        theta: float,
        mrope_section: Sequence[int],
        attention_scaling: float = 1.0,
    ) -> None:
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}.")
        self.head_dim = head_dim
        self.theta = theta
        self.attention_scaling = attention_scaling
        self.mrope_section = tuple(mrope_section)
        self._inv_freq = 1.0 / (
            theta ** (np.arange(0, head_dim, 2, dtype="float32") / np.float32(head_dim))
        )

    def forward(self, reference: Tensor, position_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """Return ``(cos, sin)`` with shape ``(3, batch, seq, head_dim)``."""
        if len(position_ids.shape) != 3:
            raise ValueError(
                "position_ids must be rank-3 with either "
                "(batch, seq, 3) or (3, batch, seq) layout, "
                f"got shape {position_ids.shape}."
            )
        if isinstance(position_ids.shape[0], int) and position_ids.shape[0] == 3:
            batch_size, seq_len = position_ids.shape[1], position_ids.shape[2]
            pos_tensor = op.reshape(position_ids, (3, batch_size, 1, seq_len))
        elif isinstance(position_ids.shape[-1], int) and position_ids.shape[-1] == 3:
            batch_size, seq_len = position_ids.shape[0], position_ids.shape[1]
            permuted_pos = op.permute_dims(position_ids, axes=[2, 0, 1])
            pos_tensor = op.reshape(permuted_pos, (3, batch_size, 1, seq_len))
        else:
            raise ValueError(
                "position_ids must have exactly one static dimension of size 3, "
                f"got shape {position_ids.shape}."
            )

        dtype = reference.dtype
        inv_freq_tensor = nn.Tensor.from_const(self._inv_freq.reshape(1, 1, -1, 1))
        inv_freq_tensor = op.broadcast_to(inv_freq_tensor, (3, batch_size, self._inv_freq.size, 1))

        freqs = op.matmul(inv_freq_tensor.astype("float32"), pos_tensor.astype("float32"))
        freqs = op.permute_dims(freqs, axes=[0, 1, 3, 2])
        emb = op.concat([freqs, freqs], dim=-1)

        def _apply_trig(func_name: str) -> Tensor:
            def compute(x: te.Tensor):
                return te.compute(
                    x.shape,
                    lambda *indices: getattr(tir, func_name)(x[indices]),
                    name=f"mrope_{func_name}",
                )

            return op.tensor_expr_op(compute, f"mrope_{func_name}", [emb])

        cos = _apply_trig("cos") * self.attention_scaling
        sin = _apply_trig("sin") * self.attention_scaling
        return cos.astype(dtype), sin.astype(dtype)


def apply_multimodal_rotary_pos_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    mrope_section: Sequence[int],
    unsqueeze_dim: int = 2,
) -> Tuple[Tensor, Tensor]:
    """Apply multimodal rotary embedding to query and key tensors."""

    split_sizes = _repeat_mrope_section(mrope_section)
    reordered_cos = _reorder_cos_sin(cos, split_sizes)
    reordered_sin = _reorder_cos_sin(sin, split_sizes)
    cos_term = op.unsqueeze(reordered_cos, dim=unsqueeze_dim)
    sin_term = op.unsqueeze(reordered_sin, dim=unsqueeze_dim)
    cos_term = cos_term.astype(q.dtype)
    sin_term = sin_term.astype(q.dtype)
    q_embed = op.add(op.multiply(q, cos_term), op.multiply(_rotate_half(q), sin_term))
    k_embed = op.add(op.multiply(k, cos_term), op.multiply(_rotate_half(k), sin_term))
    return q_embed, k_embed


@dataclass
class VisionPositionMetadata:
    """Metadata required to build multimodal position IDs."""

    vision_start_token_id: int
    image_token_id: int
    video_token_id: int
    spatial_merge_size: int
    tokens_per_second: float

    def merged_hw(self, height: int, width: int) -> Tuple[int, int]:
        if height % self.spatial_merge_size != 0 or width % self.spatial_merge_size != 0:
            raise ValueError(
                "Image or video grid is not divisible by spatial_merge_size "
                f"(got h={height}, w={width}, merge={self.spatial_merge_size})."
            )
        return height // self.spatial_merge_size, width // self.spatial_merge_size


def _text_chunk(length: int, offset: int) -> np.ndarray:
    if length <= 0:
        return np.zeros((3, 0), dtype=np.int64)
    seq = np.arange(length, dtype=np.int64)
    chunk = np.broadcast_to(seq.reshape(1, -1), (3, length))
    return chunk + offset


def _grid_chunk(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    offset: int,
    tokens_per_second: float,
    second_per_grid: float,
) -> np.ndarray:
    if grid_t <= 0 or grid_h <= 0 or grid_w <= 0:
        raise ValueError(
            f"Invalid grid shape t={grid_t}, h={grid_h}, w={grid_w} for multimodal positions."
        )
    grid_size = grid_t * grid_h * grid_w
    time_axis = (np.arange(grid_t, dtype=np.float32) * second_per_grid * tokens_per_second).astype(
        np.int64
    )
    t_index = np.repeat(time_axis, grid_h * grid_w)
    h_index = np.tile(np.repeat(np.arange(grid_h, dtype=np.int64), grid_w), grid_t)
    w_index = np.tile(np.tile(np.arange(grid_w, dtype=np.int64), grid_h), grid_t)
    stacked = np.stack([t_index, h_index, w_index])
    return stacked + offset


def _find_token_index(tokens: Sequence[int], token_id: int, start: int) -> int:
    for idx in range(start, len(tokens)):
        if tokens[idx] == token_id:
            return idx
    return len(tokens)


def get_mrope_position_ids(  # pylint: disable=too-many-arguments,too-many-locals
    input_ids: np.ndarray,
    meta: VisionPositionMetadata,
    attention_mask: Optional[np.ndarray] = None,
    image_grid_thw: Optional[np.ndarray] = None,
    video_grid_thw: Optional[np.ndarray] = None,
    second_per_grid_ts: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate 3D position IDs and deltas following Hugging Face Qwen2.5-VL."""

    input_ids = np.asarray(input_ids, dtype=np.int64)
    batch, seq_len = input_ids.shape
    position_ids = np.ones((3, batch, seq_len), dtype=np.int64)

    attention = None
    if attention_mask is not None:
        attention_mask = np.asarray(attention_mask, dtype=np.int64)
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask shape must match input_ids shape: "
                f"{attention_mask.shape} vs {input_ids.shape}"
            )
        attention = attention_mask.astype(bool)

    image_grid_thw = None if image_grid_thw is None else np.asarray(image_grid_thw, dtype=np.int64)
    video_grid_thw = None if video_grid_thw is None else np.asarray(video_grid_thw, dtype=np.int64)
    if second_per_grid_ts is not None:
        second_per_grid_ts = np.asarray(second_per_grid_ts, dtype=np.float32)

    contains_image_tokens = bool(np.any(input_ids == meta.image_token_id))
    contains_video_tokens = bool(np.any(input_ids == meta.video_token_id))
    if contains_image_tokens and image_grid_thw is None:
        raise ValueError("image_grid_thw must be provided when image tokens exist in input_ids.")
    if contains_video_tokens and video_grid_thw is None:
        raise ValueError("video_grid_thw must be provided when video tokens exist in input_ids.")
    if (
        second_per_grid_ts is not None
        and video_grid_thw is not None
        and second_per_grid_ts.shape[0] != video_grid_thw.shape[0]
    ):
        raise ValueError(
            "second_per_grid_ts length must match number of video grids "
            f"({second_per_grid_ts.shape[0]} vs {video_grid_thw.shape[0]})."
        )

    if not (contains_image_tokens or contains_video_tokens):
        if attention is not None:
            position = attention_mask.cumsum(axis=-1) - 1  # type: ignore[union-attr]
            position = np.where(attention_mask == 0, 1, position)
            position = np.expand_dims(position, axis=0).repeat(3, axis=0)
            max_pos = position.max(axis=0, keepdims=False).max(axis=-1, keepdims=True)
            delta = (max_pos + 1 - seq_len).astype(np.int64)
            return position, delta

        base = np.arange(seq_len, dtype=np.int64).reshape(1, 1, -1)
        tiled = np.broadcast_to(base, (3, batch, seq_len))
        return tiled, np.zeros((batch, 1), dtype=np.int64)

    image_index = 0
    video_index = 0
    deltas: List[int] = []

    for batch_idx in range(batch):
        tokens = input_ids[batch_idx]
        if attention is not None:
            tokens = tokens[attention[batch_idx]]
        input_tokens = tokens.tolist()
        if not input_tokens:
            deltas.append(-tokens.shape[0])
            continue

        token_array = np.array(input_tokens, dtype=np.int64)
        vision_starts = np.where(token_array == meta.vision_start_token_id)[0]
        valid_starts = vision_starts[vision_starts + 1 < token_array.shape[0]]
        following_tokens = token_array[valid_starts + 1]
        image_nums = int(np.sum(following_tokens == meta.image_token_id))
        video_nums = int(np.sum(following_tokens == meta.video_token_id))
        if image_nums > 0 and image_grid_thw is None:
            raise ValueError("Image grids are required for sequences with image tokens.")
        if video_nums > 0 and video_grid_thw is None:
            raise ValueError("Video grids are required for sequences with video tokens.")

        llm_pos_ids_list: List[np.ndarray] = []
        st = 0
        remain_images = image_nums
        remain_videos = video_nums

        for _ in range(image_nums + video_nums):
            if remain_images > 0:
                try:
                    ed_image = input_tokens.index(meta.image_token_id, st)
                except ValueError:
                    ed_image = len(input_tokens) + 1
            else:
                ed_image = len(input_tokens) + 1

            if remain_videos > 0:
                try:
                    ed_video = input_tokens.index(meta.video_token_id, st)
                except ValueError:
                    ed_video = len(input_tokens) + 1
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                grid_t, grid_h, grid_w = image_grid_thw[image_index]  # type: ignore[index]
                second_per_grid = 0.0
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                grid_t, grid_h, grid_w = video_grid_thw[video_index]  # type: ignore[index]
                if second_per_grid_ts is not None:
                    second_per_grid = float(second_per_grid_ts[video_index])
                else:
                    second_per_grid = 1.0
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t = int(grid_t)
            llm_grid_h, llm_grid_w = meta.merged_hw(int(grid_h), int(grid_w))
            text_len = ed - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_range = np.arange(text_len, dtype=np.int64).reshape(1, -1)
            text_chunk = np.broadcast_to(text_range, (3, text_len)) + st_idx
            llm_pos_ids_list.append(text_chunk)

            t_index = (
                (
                    np.broadcast_to(
                        np.arange(llm_grid_t, dtype=np.float32).reshape(-1, 1),
                        (llm_grid_t, llm_grid_h * llm_grid_w),
                    )
                    * second_per_grid
                    * meta.tokens_per_second
                )
                .astype(np.int64)
                .reshape(-1)
            )
            h_index = (
                np.arange(llm_grid_h, dtype=np.int64)
                .reshape(1, -1, 1)
                .repeat(llm_grid_t, axis=0)
                .repeat(llm_grid_w, axis=2)
                .reshape(-1)
            )
            w_index = (
                np.arange(llm_grid_w, dtype=np.int64)
                .reshape(1, 1, -1)
                .repeat(llm_grid_t, axis=0)
                .repeat(llm_grid_h, axis=1)
                .reshape(-1)
            )
            grid_chunk = np.stack([t_index, h_index, w_index]) + text_len + st_idx
            llm_pos_ids_list.append(grid_chunk)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_len = len(input_tokens) - st
            tail_range = np.arange(text_len, dtype=np.int64).reshape(1, -1)
            tail_chunk = np.broadcast_to(tail_range, (3, text_len)) + st_idx
            llm_pos_ids_list.append(tail_chunk)

        llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
        if attention is not None:
            position_ids[:, batch_idx, attention[batch_idx]] = llm_positions
        else:
            position_ids[:, batch_idx, :] = llm_positions
        deltas.append(int(llm_positions.max()) + 1 - len(input_tokens))

    deltas = np.asarray(deltas, dtype=np.int64).reshape(batch, 1)
    return position_ids, deltas
