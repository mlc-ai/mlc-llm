"""Operators enabled by external modules."""

from typing import Optional, Tuple

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

    # pylint: disable=too-many-boolean-expressions
    if x.dtype == "float8_e5m2" and weight_dtype == "float8_e5m2" and out_dtype == "float16":
        func_name = "cutlass.group_gemm_e5m2_e5m2_fp16"
    elif x.dtype == "float8_e4m3fn" and weight_dtype == "float8_e5m2" and out_dtype == "float16":
        func_name = "cutlass.group_gemm_e4m3_e5m2_fp16"
    elif x.dtype == "float8_e4m3fn" and weight_dtype == "float8_e4m3fn" and out_dtype == "float16":
        func_name = "cutlass.group_gemm_e4m3_e4m3_fp16"
    elif (x.dtype == "float16" and weight_dtype == "float16" and out_dtype == "float16") or (
        x.dtype == "bfloat16" and weight_dtype == "bfloat16" and out_dtype == "bfloat16"
    ):
        func_name = "cutlass.group_gemm"
    else:
        raise NotImplementedError(
            f"Unsupported data type: x={x.dtype}, weight={weight_dtype}, out={out_dtype}"
        )
    # pylint: enable=too-many-boolean-expressions

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

    if x.dtype == "float8_e5m2" and weight_dtype == "float8_e5m2" and out_dtype == "float16":
        func_name = "cutlass.gemm_e5m2_e5m2_fp16"
    elif x.dtype == "float8_e4m3fn" and weight_dtype == "float8_e5m2" and out_dtype == "float16":
        func_name = "cutlass.gemm_e5m2_e4m3_fp16"
    elif x.dtype == "float8_e4m3fn" and weight_dtype == "float8_e4m3fn" and out_dtype == "float16":
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


def fp8_groupwise_scaled_gemm(  # pylint: disable=too-many-arguments
    x: nn.Tensor,
    x_scale: nn.Tensor,
    weight: nn.Tensor,
    weight_scale: nn.Tensor,
    block_size: Tuple[int, int],
    out_dtype: str,
):
    """Cutlass block-scale fp8 gemm operator.

    Parameters
    ----------
    x : nn.Tensor
        The input tensor, with shape of [m, k].

    x_scale : nn.Tensor
        The scale tensor, with shape of [k // block_size, m].

    weight : nn.Tensor
        The weight tensor, with shape of [n, k].

    weight_scale : nn.Tensor
        The scale tensor, with shape of [n // block_size, k // block_size].

    block_size : Tuple[int, int]
        The block size.

    out_dtype : str
        The data type of the output tensor.

    Returns
    -------
    out : nn.Tensor
        The output tensor, with shape of [m, n] and dtype of `out_dtype`.
    """
    assert x.ndim >= 2
    assert weight.ndim == 2
    assert x_scale.ndim == x.ndim
    assert weight_scale.ndim == weight.ndim

    if block_size[0] != 128 or block_size[1] != 128:
        raise ValueError(f"block_size must be (128, 128), but got {block_size}")
    if x.dtype != "float8_e4m3fn" or weight.dtype != "float8_e4m3fn":
        raise ValueError(
            f"x and weight must be float8_e4m3fn, but got x={x.dtype}, weight={weight.dtype}"
        )
    if x_scale.dtype != "float32" or weight_scale.dtype != "float32":
        raise ValueError(
            "x_scale and weight_scale must be float32, but got "
            f"x_scale={x_scale.dtype}, weight_scale={weight_scale.dtype}"
        )
    if out_dtype not in ["float16", "bfloat16"]:
        raise ValueError(f"out_dtype must be float16 or bfloat16, but got {out_dtype}")

    func_name = "cutlass.groupwise_scaled_gemm_e4m3fn_e4m3fn"
    workspace = op.empty((4096 * 1024,), dtype="uint8", name="workspace")
    return op.extern(
        func_name,
        args=[x, weight, x_scale, weight_scale, workspace, block_size[0], block_size[1]],
        out=nn.Tensor.placeholder((*x.shape[:-1], weight.shape[0]), dtype=out_dtype),
    )


def fp8_groupwise_scaled_bmm(  # pylint: disable=too-many-arguments
    x: nn.Tensor,
    x_scale: nn.Tensor,
    weight: nn.Tensor,
    weight_scale: nn.Tensor,
    block_size: Tuple[int, int],
    out_dtype: str,
):
    """Cutlass block-scale fp8 gemm operator.

    Parameters
    ----------
    x : nn.Tensor
        The input tensor, with shape of [b, m, k].

    x_scale : nn.Tensor
        The scale tensor, with shape of [b, k // block_size, m].

    weight : nn.Tensor
        The weight tensor, with shape of [b, n, k].

    weight_scale : nn.Tensor
        The scale tensor, with shape of [b, n // block_size, k // block_size].

    block_size : Tuple[int, int]
        The block size.

    out_dtype : str
        The data type of the output tensor.

    Returns
    -------
    out : nn.Tensor
        The output tensor, with shape of [m, n] and dtype of `out_dtype`.
    """
    assert x.ndim == 3
    assert weight.ndim == 3
    assert x_scale.ndim == x.ndim
    assert weight_scale.ndim == weight.ndim
    assert x.shape[0] == x_scale.shape[0] == weight.shape[0] == weight_scale.shape[0]

    if block_size[0] != 128 or block_size[1] != 128:
        raise ValueError(f"block_size must be (128, 128), but got {block_size}")
    if x.dtype != "float8_e4m3fn" or weight.dtype != "float8_e4m3fn":
        raise ValueError(
            f"x and weight must be float8_e4m3fn, but got x={x.dtype}, weight={weight.dtype}"
        )
    if x_scale.dtype != "float32" or weight_scale.dtype != "float32":
        raise ValueError(
            "x_scale and weight_scale must be float32, but got "
            f"x_scale={x_scale.dtype}, weight_scale={weight_scale.dtype}"
        )
    if out_dtype not in ["float16", "bfloat16"]:
        raise ValueError(f"out_dtype must be float16 or bfloat16, but got {out_dtype}")

    func_name = "cutlass.groupwise_scaled_bmm_e4m3fn_e4m3fn"
    workspace = op.empty((4096 * 1024,), dtype="uint8", name="workspace")
    return op.extern(
        func_name,
        args=[x, weight, x_scale, weight_scale, workspace, block_size[0], block_size[1]],
        out=nn.Tensor.placeholder((x.shape[0], x.shape[1], weight.shape[1]), dtype=out_dtype),
    )


def fp8_groupwise_scaled_group_gemm(  # pylint: disable=too-many-arguments,too-many-locals
    x: nn.Tensor,
    x_scale: nn.Tensor,
    weight: nn.Tensor,
    weight_scale: nn.Tensor,
    indptr: nn.Tensor,
    block_size: Tuple[int, int],
    out_dtype: str,
):
    """Triton block-scale fp8 group gemm operator.

    Parameters
    ----------
    x : nn.Tensor
        The input tensor, with shape of [m, k].

    x_scale : nn.Tensor
        The scale tensor, with shape of [m, k // block_size].

    weight : nn.Tensor
        The weight tensor, with shape of [num_experts, n, k].

    weight_scale : nn.Tensor
        The scale tensor, with shape of [num_experts, n // block_size, k // block_size].

    indptr : nn.Tensor
        The indptr tensor of group gemm, with shape of [num_experts + 1,].

    block_size : Tuple[int, int]
        The block size.

    out_dtype : str
        The data type of the output tensor.

    Returns
    -------
    out : nn.Tensor
        The output tensor, with shape of [m, n] and dtype of `out_dtype`.
    """
    assert x.ndim >= 2
    assert weight.ndim == 3
    assert x_scale.ndim == x.ndim
    assert weight_scale.ndim == weight.ndim
    assert x.shape[-1] == weight.shape[2]
    assert (x.shape[-1] + block_size[1] - 1) // block_size[1] == x_scale.shape[-1]
    assert (weight.shape[2] + block_size[1] - 1) // block_size[1] == weight_scale.shape[2]
    assert (weight.shape[1] + block_size[0] - 1) // block_size[0] == weight_scale.shape[1]

    if block_size[0] != 128 or block_size[1] != 128:
        raise ValueError(f"block_size must be (128, 128), but got {block_size}")
    if x.dtype != "float8_e4m3fn" or weight.dtype != "float8_e4m3fn":
        raise ValueError(
            f"x and weight must be float8_e4m3fn, but got x={x.dtype}, weight={weight.dtype}"
        )
    if x_scale.dtype != "float32" or weight_scale.dtype != "float32":
        raise ValueError(
            "x_scale and weight_scale must be float32, but got "
            f"x_scale={x_scale.dtype}, weight_scale={weight_scale.dtype}"
        )
    if out_dtype not in ["float16", "bfloat16"]:
        raise ValueError(f"out_dtype must be float16 or bfloat16, but got {out_dtype}")

    num_experts = weight.shape[0]
    m = x.shape[0]
    for i in range(1, x.ndim - 1):
        m *= x.shape[i]
    n = weight.shape[1]
    k = x.shape[-1]
    assert weight_scale.shape[0] == num_experts
    assert indptr.ndim == 1
    assert indptr.shape[0] == num_experts
    assert indptr.dtype == "int64"

    x_shape = x.shape
    if x.ndim > 2:
        x = x.reshape(m, k)
        x_scale = x_scale.reshape(m, x_scale.shape[-1])

    func_name = "cutlass.groupwise_scaled_group_gemm_e4m3fn_e4m3fn"
    workspace = op.empty((4096 * 1024,), dtype="uint8", name="workspace")
    out = op.extern(
        func_name,
        args=[x, weight, x_scale, weight_scale, indptr, workspace, block_size[0], block_size[1]],
        out=nn.Tensor.placeholder((m, n), dtype=out_dtype),
    )
    return out.reshape(*x_shape[:-1], n) if len(x_shape) > 2 else out
