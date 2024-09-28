"""Operators enabled by external modules."""

from typing import Optional

from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import op


def group_gemm(
    x: nn.Tensor,
    weight: nn.Tensor,
    indptr: nn.Tensor,
    scale: Optional[nn.Tensor] = None,
    weight_dtype: Optional[str] = None,
    out_dtype: Optional[str] = None,
):  # pylint: disable=too-many-arguments
    """
    Cutlass group gemm operator.

    Parameters
    ----------
    x : nn.Tensor
        The input tensor, with shape of [m, k].

    weight : nn.Tensor
        The weight tensor, with shape of [num_groups, n, k].

    indptr : nn.Tensor
        The indptr tensor, with shape of [num_groups].

    scale : Optional[nn.Tensor]
        The scale tensor, with shape of [1].

    weight_dtype: Optional[str]
        The data type of the weight tensor.

    out_dtype: Optional[str]
        The data type of the output tensor.

    Returns
    -------
    nn.Tensor
        The output tensor, with shape of [m, n].
    """
    assert x.ndim == 2
    assert weight.ndim == 3
    assert indptr.ndim == 1
    assert weight.shape[0] == indptr.shape[0]
    assert indptr.dtype == "int64"
    out_dtype = out_dtype if out_dtype else x.dtype
    weight_dtype = weight_dtype if weight_dtype else weight.dtype

    if x.dtype == "e5m2_float8" and weight_dtype == "e5m2_float8" and out_dtype == "float16":
        func_name = "cutlass.group_gemm_e5m2_e5m2_fp16"
    elif x.dtype == "e4m3_float8" and weight_dtype == "e5m2_float8" and out_dtype == "float16":
        func_name = "cutlass.group_gemm_e4m3_e5m2_fp16"
    elif x.dtype == "e4m3_float8" and weight_dtype == "e4m3_float8" and out_dtype == "float16":
        func_name = "cutlass.group_gemm_e4m3_e4m3_fp16"
    elif x.dtype == "float16" and weight_dtype == "float16" and out_dtype == "float16":
        func_name = "cutlass.group_gemm_fp16_sm90"
    else:
        raise NotImplementedError(
            f"Unsupported data type: x={x.dtype}, weight={weight_dtype}, out={out_dtype}"
        )

    if "float8" in x.dtype:
        assert scale is not None, "scale is required for float8 input"

    workspace = op.empty((4096 * 1024,), dtype="uint8", name="workspace")

    return op.extern(
        func_name,
        args=[x, weight, indptr, workspace] + ([scale] if scale is not None else []),
        out=nn.Tensor.placeholder((x.shape[0], weight.shape[1]), dtype=out_dtype),
    )


def fp8_gemm(
    x: nn.Tensor,
    weight: nn.Tensor,
    scale: nn.Tensor,
    weight_dtype: Optional[str] = None,
    out_dtype: Optional[str] = None,
):  # pylint: disable=too-many-arguments
    """
    Cutlass fp8 gemm operator.

    Parameters
    ----------
    x : nn.Tensor
        The input tensor, with shape of [m, k].

    weight : nn.Tensor
        The weight tensor, with shape of [num_groups, n, k].

    scale : Optional[nn.Tensor]
        The scale tensor, with shape of [1].

    weight_dtype: Optional[str]
        The data type of the weight tensor.

    out_dtype: Optional[str]
        The data type of the output tensor.

    Returns
    -------
    nn.Tensor
        The output tensor, with shape of [m, n].
    """
    assert x.ndim >= 2
    assert weight.ndim == 2
    assert scale.ndim == 1 and scale.shape[0] == 1
    out_dtype = out_dtype if out_dtype else x.dtype
    weight_dtype = weight_dtype if weight_dtype else weight.dtype

    if x.dtype == "e5m2_float8" and weight_dtype == "e5m2_float8" and out_dtype == "float16":
        func_name = "cutlass.gemm_e5m2_e5m2_fp16"
    elif x.dtype == "e4m3_float8" and weight_dtype == "e5m2_float8" and out_dtype == "float16":
        func_name = "cutlass.gemm_e5m2_e4m3_fp16"
    elif x.dtype == "e4m3_float8" and weight_dtype == "e4m3_float8" and out_dtype == "float16":
        func_name = "cutlass.gemm_e4m3_e4m3_fp16"
    else:
        raise NotImplementedError(
            f"Unsupported data type: x={x.dtype}, weight={weight_dtype}, out={out_dtype}"
        )

    workspace = op.empty((4096 * 1024,), dtype="uint8", name="workspace")

    return op.extern(
        func_name,
        args=[x, weight, workspace, scale],
        out=nn.Tensor.placeholder((*x.shape[:-1], weight.shape[0]), dtype=out_dtype),
    )
