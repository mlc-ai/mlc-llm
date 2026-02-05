"""Utilities for Multimodal Rotary Position Embeddings (MRoPE)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
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


def apply_multimodal_rotary_pos_emb(  # pylint: disable=too-many-arguments
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
        """Return merged height/width after applying ``spatial_merge_size``."""

        if height % self.spatial_merge_size != 0 or width % self.spatial_merge_size != 0:
            raise ValueError(
                "Image or video grid is not divisible by spatial_merge_size "
                f"(got h={height}, w={width}, merge={self.spatial_merge_size})."
            )
        return height // self.spatial_merge_size, width // self.spatial_merge_size


def _text_chunk(length: int, offset: int) -> np.ndarray:
    """Create a text-position chunk with a shared scalar offset for T/H/W axes."""

    if length <= 0:
        return np.zeros((3, 0), dtype=np.int64)
    seq: np.ndarray = np.arange(length, dtype=np.int64)
    chunk = np.broadcast_to(seq.reshape(1, -1), (3, length))
    return chunk + offset


def _grid_chunk(  # pylint: disable=too-many-arguments
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


def _next_chunk_offset(chunks: Sequence[np.ndarray]) -> int:
    if not chunks:
        return 0
    return int(chunks[-1].max()) + 1


def _count_vision_items(
    token_array: np.ndarray,
    vision_start_token_id: int,
    image_token_id: int,
    video_token_id: int,
) -> Tuple[int, int]:
    vision_starts = np.where(token_array == vision_start_token_id)[0]
    valid_starts = vision_starts[vision_starts + 1 < token_array.shape[0]]
    following_tokens = token_array[valid_starts + 1]
    image_count = int(np.sum(following_tokens == image_token_id))
    video_count = int(np.sum(following_tokens == video_token_id))
    return image_count, video_count


def _next_vision_block(
    tokens: Sequence[int],
    start: int,
    meta: VisionPositionMetadata,
    has_images: bool,
    has_videos: bool,
) -> Tuple[str, int]:
    sentinel = len(tokens) + 1
    image_end = _find_token_index(tokens, meta.image_token_id, start) if has_images else sentinel
    video_end = _find_token_index(tokens, meta.video_token_id, start) if has_videos else sentinel
    if image_end < video_end:
        return "image", image_end
    return "video", video_end


def _load_grid_for_block(  # pylint: disable=too-many-arguments
    block_kind: str,
    image_grid_thw: Optional[np.ndarray],
    video_grid_thw: Optional[np.ndarray],
    second_per_grid_ts: Optional[np.ndarray],
    image_index: int,
    video_index: int,
) -> Tuple[int, int, int, float, int, int]:
    if block_kind == "image":
        if image_grid_thw is None:
            raise ValueError("Image grids are required for sequences with image tokens.")
        grid_t, grid_h, grid_w = image_grid_thw[image_index]
        return int(grid_t), int(grid_h), int(grid_w), 0.0, image_index + 1, video_index

    if video_grid_thw is None:
        raise ValueError("Video grids are required for sequences with video tokens.")
    grid_t, grid_h, grid_w = video_grid_thw[video_index]
    second_per_grid = (
        float(second_per_grid_ts[video_index]) if second_per_grid_ts is not None else 1.0
    )
    return int(grid_t), int(grid_h), int(grid_w), second_per_grid, image_index, video_index + 1


def _build_sequence_position_ids(  # pylint: disable=too-many-arguments,too-many-locals
    input_tokens: Sequence[int],
    meta: VisionPositionMetadata,
    image_grid_thw: Optional[np.ndarray],
    video_grid_thw: Optional[np.ndarray],
    second_per_grid_ts: Optional[np.ndarray],
    image_index: int,
    video_index: int,
) -> Tuple[np.ndarray, int, int, int]:
    token_array = np.asarray(input_tokens, dtype=np.int64)
    image_count, video_count = _count_vision_items(
        token_array,
        vision_start_token_id=meta.vision_start_token_id,
        image_token_id=meta.image_token_id,
        video_token_id=meta.video_token_id,
    )
    if image_count > 0 and image_grid_thw is None:
        raise ValueError("Image grids are required for sequences with image tokens.")
    if video_count > 0 and video_grid_thw is None:
        raise ValueError("Video grids are required for sequences with video tokens.")

    llm_pos_ids_list: List[np.ndarray] = []
    start = 0
    remain_images = image_count
    remain_videos = video_count
    for _ in range(image_count + video_count):
        block_kind, block_end = _next_vision_block(
            tokens=input_tokens,
            start=start,
            meta=meta,
            has_images=remain_images > 0,
            has_videos=remain_videos > 0,
        )
        (
            grid_t,
            grid_h,
            grid_w,
            second_per_grid,
            image_index,
            video_index,
        ) = _load_grid_for_block(
            block_kind=block_kind,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            image_index=image_index,
            video_index=video_index,
        )
        if block_kind == "image":
            remain_images -= 1
        else:
            remain_videos -= 1

        llm_grid_h, llm_grid_w = meta.merged_hw(grid_h, grid_w)
        text_len = block_end - start
        text_offset = _next_chunk_offset(llm_pos_ids_list)
        llm_pos_ids_list.append(_text_chunk(text_len, text_offset))
        grid_offset = text_offset + text_len
        llm_pos_ids_list.append(
            _grid_chunk(
                grid_t=grid_t,
                grid_h=llm_grid_h,
                grid_w=llm_grid_w,
                offset=grid_offset,
                tokens_per_second=meta.tokens_per_second,
                second_per_grid=second_per_grid,
            )
        )
        start = block_end + grid_t * llm_grid_h * llm_grid_w

    if start < len(input_tokens):
        tail_len = len(input_tokens) - start
        tail_offset = _next_chunk_offset(llm_pos_ids_list)
        llm_pos_ids_list.append(_text_chunk(tail_len, tail_offset))

    if not llm_pos_ids_list:
        empty_positions: np.ndarray = np.zeros((3, 0), dtype=np.int64)
        return empty_positions, 0, image_index, video_index
    llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
    delta = int(llm_positions.max()) + 1 - len(input_tokens)
    return llm_positions, delta, image_index, video_index


def _text_only_position_ids(
    input_ids: np.ndarray,
    attention_mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    batch, seq_len = input_ids.shape
    if attention_mask is None:
        base: np.ndarray = np.arange(seq_len, dtype=np.int64).reshape(1, 1, -1)
        tiled = np.broadcast_to(base, (3, batch, seq_len))
        return tiled, np.zeros((batch, 1), dtype=np.int64)

    position = attention_mask.cumsum(axis=-1) - 1
    position = np.where(attention_mask == 0, 1, position)
    position = np.expand_dims(position, axis=0).repeat(3, axis=0)
    max_pos = position.max(axis=0, keepdims=False).max(axis=-1, keepdims=True)
    delta = (max_pos + 1 - seq_len).astype(np.int64)
    return position.astype(np.int64), delta


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
        return _text_only_position_ids(input_ids, attention_mask)

    image_index = 0
    video_index = 0
    deltas: List[int] = []

    for batch_idx in range(batch):
        tokens = input_ids[batch_idx]
        if attention is not None:
            tokens = tokens[attention[batch_idx]]
        token_values = np.asarray(tokens, dtype=np.int64).tolist()
        input_tokens: List[int] = [int(token) for token in token_values]
        if not input_tokens:
            deltas.append(0)
            continue

        llm_positions, delta, image_index, video_index = _build_sequence_position_ids(
            input_tokens=input_tokens,
            meta=meta,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            image_index=image_index,
            video_index=video_index,
        )
        if attention is not None:
            position_ids[:, batch_idx, attention[batch_idx]] = llm_positions
        else:
            position_ids[:, batch_idx, :] = llm_positions
        deltas.append(delta)

    delta_array = np.asarray(deltas, dtype=np.int64).reshape(batch, 1)
    return position_ids, delta_array
