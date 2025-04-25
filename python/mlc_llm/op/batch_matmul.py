"""Batch matmul operators"""

from typing import Tuple

from tvm.relax.frontend import nn

from mlc_llm.op import cutlass
from mlc_llm.quantization.block_scale_quantization import rowwise_group_quant_fp8


def quantized_bmm(
    x: nn.Tensor,
    w: nn.Tensor,
    w_scale: nn.Tensor,
    block_size: Tuple[int, int],
) -> nn.Tensor:
    """Quantized batch matmul.
    Currently only support CUDA backend (by using CUTLASS).

    Parameters
    ----------
    x : nn.Tensor
        The input tensor, with shape of [b, m, k].

    w : nn.Tensor
        The weight tensor, with shape of [b, n, k] (column major).

    w_scale : nn.Tensor
        The scale tensor, with shape of [b, n // block_size[0], k // block_size[1]].

    block_size : Tuple[int, int]
        The block size.

    Returns
    -------
    ret : nn.Tensor
        The output tensor, with shape of [b, m, n].
    """
    x_fp8, x_scale = rowwise_group_quant_fp8(
        x, block_size[1], w.dtype, transpose_scale=True, keep_first_batch_dim=True
    )
    return cutlass.fp8_block_scale_bmm(x_fp8, x_scale, w, w_scale, block_size, out_dtype=x.dtype)
