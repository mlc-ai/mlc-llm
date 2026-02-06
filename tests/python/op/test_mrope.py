import numpy as np
import pytest

tvm = pytest.importorskip("tvm")
from tvm import relax
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import spec
from tvm.runtime import tensor as tvm_tensor

from mlc_llm.op import (
    MultimodalRotaryEmbedding,
    VisionPositionMetadata,
    apply_multimodal_rotary_pos_emb,
    get_mrope_position_ids,
)


def _numpy_rotate_half(x: np.ndarray) -> np.ndarray:
    x1, x2 = np.split(x, 2, axis=-1)
    return np.concatenate([-x2, x1], axis=-1)


def _numpy_apply_mrope(
    q: np.ndarray,
    k: np.ndarray,
    position_ids: np.ndarray,
    theta: float,
    mrope_section: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if position_ids.ndim != 3:
        raise ValueError(f"position_ids must be rank-3, got shape {position_ids.shape}")
    if position_ids.shape[0] == 3:
        position_ids = np.transpose(position_ids, (1, 2, 0))
    elif position_ids.shape[-1] != 3:
        raise ValueError(
            "position_ids must have shape (batch, seq, 3) or (3, batch, seq), "
            f"got {position_ids.shape}"
        )

    head_dim = q.shape[-1]
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / float(head_dim)))
    pos = np.transpose(position_ids, (2, 0, 1))
    inv = inv_freq.reshape(1, 1, -1, 1).astype(np.float32)
    inv = np.broadcast_to(inv, (3, pos.shape[1], inv_freq.size, 1))
    pos = pos.reshape(3, pos.shape[1], 1, pos.shape[2]).astype(np.float32)
    freqs = np.matmul(inv, pos)
    freqs = np.transpose(freqs, (0, 1, 3, 2))
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos = np.cos(emb)
    sin = np.sin(emb)
    split_sizes = list(mrope_section) * 2
    split_points = np.cumsum(split_sizes)[:-1]
    cos_chunks = np.split(cos, split_points, axis=-1)
    sin_chunks = np.split(sin, split_points, axis=-1)
    cos = np.concatenate([chunk[idx % 3] for idx, chunk in enumerate(cos_chunks)], axis=-1)
    sin = np.concatenate([chunk[idx % 3] for idx, chunk in enumerate(sin_chunks)], axis=-1)
    cos = np.expand_dims(cos, axis=2)
    sin = np.expand_dims(sin, axis=2)
    q_out = q * cos + _numpy_rotate_half(q) * sin
    k_out = k * cos + _numpy_rotate_half(k) * sin
    return q_out, k_out


def _evaluate_tensor(expr):
    mod = tvm.IRModule.from_expr(expr)
    target = tvm.target.Target("llvm")
    ex = tvm.relax.build(mod, target)
    vm = tvm.relax.VirtualMachine(ex, tvm.cpu())
    return vm["main"]().numpy()


def _run_mlc_mrope(
    q_np: np.ndarray,
    k_np: np.ndarray,
    position_ids_np: np.ndarray,
    theta: float,
    mrope_section: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    class RopeModule(nn.Module):  # pylint: disable=too-few-public-methods
        def __init__(self):
            super().__init__()
            self.rotary = MultimodalRotaryEmbedding(q_np.shape[-1], theta, mrope_section)

        def forward(
            self,
            q: nn.Tensor,
            k: nn.Tensor,
            pos: nn.Tensor,
        ):
            """Run MRoPE on test tensors and return rotated query/key outputs."""
            cos, sin = self.rotary(q, pos)
            return apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)

    module = RopeModule()
    mod, _, _ = module.export_tvm(
        spec={
            "forward": {
                "q": spec.Tensor(q_np.shape, "float32"),
                "k": spec.Tensor(k_np.shape, "float32"),
                "pos": spec.Tensor(position_ids_np.shape, "int64"),
            }
        },
        allow_extern=True,
    )
    target = tvm.target.Target("llvm")
    exec_mod = relax.build(mod, target=target)
    vm = relax.VirtualMachine(exec_mod, tvm.cpu())
    device = tvm.cpu()
    q_nd = tvm_tensor(q_np.astype("float32"), device=device)
    k_nd = tvm_tensor(k_np.astype("float32"), device=device)
    pos_nd = tvm_tensor(position_ids_np.astype("int64"), device=device)
    out_q, out_k = vm["forward"](q_nd, k_nd, pos_nd)
    return out_q.numpy(), out_k.numpy()


def test_apply_mrope_matches_numpy_reference():
    theta = 10000.0
    mrope_section = (2, 2, 2)
    batch, seq_len, heads, head_dim = 1, 4, 2, 12
    rng = np.random.default_rng(0)
    q_np = rng.standard_normal((batch, seq_len, heads, head_dim), dtype=np.float32)
    k_np = rng.standard_normal((batch, seq_len, heads, head_dim), dtype=np.float32)
    position_ids = np.zeros((batch, seq_len, 3), dtype=np.int64)
    position_ids[0, :, 0] = np.arange(seq_len)
    position_ids[0, :, 1] = np.arange(seq_len) * 2
    position_ids[0, :, 2] = np.arange(seq_len) * 3

    mlc_q, mlc_k = _run_mlc_mrope(q_np, k_np, position_ids, theta, mrope_section)
    ref_q, ref_k = _numpy_apply_mrope(q_np, k_np, position_ids, theta, mrope_section)

    np.testing.assert_allclose(mlc_q, ref_q, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(mlc_k, ref_k, rtol=1e-5, atol=1e-5)


def test_get_mrope_position_ids_text_only():
    input_ids = np.array([[1, 2, 3, 0, 0]], dtype=np.int64)
    attention_mask = np.array([[1, 1, 1, 0, 0]], dtype=np.int64)
    meta = VisionPositionMetadata(
        vision_start_token_id=1000,
        image_token_id=1001,
        video_token_id=1002,
        spatial_merge_size=2,
        tokens_per_second=4.0,
    )
    position_ids, deltas = get_mrope_position_ids(
        input_ids,
        meta,
        attention_mask=attention_mask,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
    )
    expected = attention_mask.cumsum(axis=-1) - 1
    expected = np.where(attention_mask == 0, 1, expected)
    expected = np.expand_dims(expected, axis=0).repeat(3, axis=0)
    np.testing.assert_array_equal(position_ids, expected)
    np.testing.assert_array_equal(deltas, np.array([[-2]], dtype=np.int64))


def test_get_mrope_position_ids_single_image_block():
    meta = VisionPositionMetadata(
        vision_start_token_id=5000,
        image_token_id=5001,
        video_token_id=6000,
        spatial_merge_size=2,
        tokens_per_second=4.0,
    )
    input_ids = np.array(
        [[11, 12, 5000, 5001, 21, 22, 23, 24, 31, 32]],
        dtype=np.int64,
    )
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    image_grid_thw = np.array([[1, 4, 4]], dtype=np.int64)
    position_ids, deltas = get_mrope_position_ids(
        input_ids,
        meta,
        attention_mask=attention_mask,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        second_per_grid_ts=None,
    )
    expected = np.array(
        [
            [0, 1, 2, 3, 3, 3, 3, 5, 6, 7],
            [0, 1, 2, 3, 3, 4, 4, 5, 6, 7],
            [0, 1, 2, 3, 4, 3, 4, 5, 6, 7],
        ],
        dtype=np.int64,
    ).reshape(3, 1, -1)
    np.testing.assert_array_equal(position_ids, expected)
    np.testing.assert_array_equal(deltas, np.array([[-2]], dtype=np.int64))


def test_apply_mrope_accepts_3_batch_seq_layout():
    theta = 10000.0
    mrope_section = (2, 2, 2)
    batch, seq_len, heads, head_dim = 1, 4, 2, 12
    rng = np.random.default_rng(1)
    q_np = rng.standard_normal((batch, seq_len, heads, head_dim), dtype=np.float32)
    k_np = rng.standard_normal((batch, seq_len, heads, head_dim), dtype=np.float32)

    position_ids_bsc = np.zeros((batch, seq_len, 3), dtype=np.int64)
    position_ids_bsc[0, :, 0] = np.arange(seq_len)
    position_ids_bsc[0, :, 1] = np.arange(seq_len) * 2
    position_ids_bsc[0, :, 2] = np.arange(seq_len) * 3
    position_ids_3bs = np.transpose(position_ids_bsc, (2, 0, 1))

    mlc_q_bsc, mlc_k_bsc = _run_mlc_mrope(q_np, k_np, position_ids_bsc, theta, mrope_section)
    mlc_q_3bs, mlc_k_3bs = _run_mlc_mrope(q_np, k_np, position_ids_3bs, theta, mrope_section)
    ref_q, ref_k = _numpy_apply_mrope(q_np, k_np, position_ids_bsc, theta, mrope_section)

    np.testing.assert_allclose(mlc_q_bsc, ref_q, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(mlc_k_bsc, ref_k, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(mlc_q_3bs, ref_q, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(mlc_k_3bs, ref_k, rtol=1e-5, atol=1e-5)


def test_get_mrope_position_ids_output_is_directly_usable():
    theta = 10000.0
    mrope_section = (2, 2, 2)
    meta = VisionPositionMetadata(
        vision_start_token_id=7000,
        image_token_id=7001,
        video_token_id=7002,
        spatial_merge_size=2,
        tokens_per_second=4.0,
    )
    input_ids = np.array([[11, 12, 7000, 7001, 21, 22, 23, 24, 31, 32]], dtype=np.int64)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    image_grid_thw = np.array([[1, 4, 4]], dtype=np.int64)
    position_ids_3bs, _ = get_mrope_position_ids(
        input_ids,
        meta,
        attention_mask=attention_mask,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        second_per_grid_ts=None,
    )
    position_ids_bsc = np.transpose(position_ids_3bs, (1, 2, 0))

    batch, seq_len = input_ids.shape
    heads, head_dim = 2, 12
    rng = np.random.default_rng(2)
    q_np = rng.standard_normal((batch, seq_len, heads, head_dim), dtype=np.float32)
    k_np = rng.standard_normal((batch, seq_len, heads, head_dim), dtype=np.float32)

    mlc_q_3bs, mlc_k_3bs = _run_mlc_mrope(q_np, k_np, position_ids_3bs, theta, mrope_section)
    mlc_q_bsc, mlc_k_bsc = _run_mlc_mrope(q_np, k_np, position_ids_bsc, theta, mrope_section)

    np.testing.assert_allclose(mlc_q_3bs, mlc_q_bsc, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(mlc_k_3bs, mlc_k_bsc, rtol=1e-5, atol=1e-5)
