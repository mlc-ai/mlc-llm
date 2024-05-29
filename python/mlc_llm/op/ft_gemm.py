"""Operators enabled by external modules."""

import operator
from functools import reduce
from typing import Optional

from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import op


def faster_transformer_dequantize_gemm(  # pylint: disable=too-many-arguments
    x: nn.Tensor,
    weight: nn.Tensor,
    scale: nn.Tensor,
    bias: Optional[nn.Tensor] = None,
    activation: Optional[str] = None,
    group_size: Optional[int] = None,
):
    """
    Faster Transformer dequantize gemm inference with CutlassFpAIntB

    Parameters
    ----------
    x : nn.Tensor
        The input tensor, with shape of [*m, k].

    weight : nn.Tensor
        The quantized weight data tensor, with shape of [k, n // num_elem_per_storage].

    scale : nn.Tensor
        The quantized weight scale tensor, with shape of [k // group_size, n].

    bias : Optional[nn.Tensor]
        The optional bias for matmul, with shape broadcastable to [*m, n].

    group_size : Optional[int]
        The optional group size. If not set, then using k as group size.

    Returns
    ------
    ret: nn.Tensor
        The output tensor of deocde matmul, with shape of [*m, n].
    """
    assert x.dtype == "float16" and x.ndim >= 1
    assert weight.ndim == 2
    assert scale.dtype == "float16" and scale.ndim == 2
    assert x.shape[-1] == weight.shape[0], (
        "Reduction dimension mismatched between x and weight, "
        f"{x.shape[-1]} vs {weight.shape[0]}."
    )
    assert activation in [
        None,
        "relu",
        "gelu",
        "silu",
        "identity",
    ], "Supported activations are [None, 'identity', 'gelu', 'silu', 'relu']."
    activation = activation if activation else "identity"
    m = reduce(operator.mul, x.shape[:-1], 1)
    k = x.shape[-1]
    n = scale.shape[1]

    if not group_size:
        group_size = k

    if bias:
        assert bias.dtype == "float16" and bias.ndim >= 1
        bias_stride = (
            bias.shape[-1]
            if bias and not reduce(operator.mul, bias.shape, 1) == bias.shape[-1]
            else 0
        )
        return op.extern(
            name="fastertransformer.gemm_fp16_int_bias",
            args=[
                x,
                weight,
                scale,
                bias,
                activation,
                m,
                n,
                k,
                group_size,
                bias_stride,
            ],
            out=nn.Tensor.placeholder((*x.shape[:-1], scale.shape[1]), dtype="float16"),
        )
    return op.extern(
        name="fastertransformer.gemm_fp16_int",
        args=[x, weight, scale, activation, m, n, k, group_size],
        out=nn.Tensor.placeholder((*x.shape[:-1], scale.shape[1]), dtype="float16"),
    )


def faster_transformer_moe_gemm(  # pylint: disable=too-many-arguments
    x: nn.Tensor,
    weight: nn.Tensor,
    total_rows_before: nn.Tensor,
):
    """
    Faster Transformer moe gemm inference with CutlassFpAIntB

    Parameters
    ----------
    x : nn.Tensor
        The input tensor, with shape of [*m, k].

    weight : nn.Tensor
        The weight data tensor, with shape of [num_experts, n, k].

    total_rows_before : nn.Tensor
        The total rows before tensor the current expert, with shape of [num_experts]. This is the
        same as the indptr excluding the first zero element.

    Returns
    ------
    ret: nn.Tensor
        The output tensor of deocde matmul, with shape of [*m, n].
    """
    assert x.dtype == "float16" and x.ndim >= 1
    assert weight.dtype == "float16" and weight.ndim == 3
    assert x.shape[-1] == weight.shape[-1], (
        "Reduction dimension mismatched between x and weight, "
        f"{x.shape[-1]} vs {weight.shape[-1]}."
    )
    m = reduce(operator.mul, x.shape[:-1], 1)
    num_experts = weight.shape[0]
    n = weight.shape[1]
    k = x.shape[-1]

    return op.extern(
        name="fastertransformer.moe_gemm_fp16_fp16",
        args=[x, weight, total_rows_before, m, n, k, num_experts],
        out=nn.Tensor.placeholder((*x.shape[:-1], n), dtype="float16"),
    )
