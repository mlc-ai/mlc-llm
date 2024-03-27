"""Common utilities for quantization"""

from typing import Callable, List, Optional

from tvm import te, tir, relax
from tvm import DataType
from tvm.ir import IRModule
from tvm.target import Target
from tvm import dlight as dl
from tvm.relax.frontend import nn
from mlc_chat.support import tensor_parallel as tp


def convert_uint_to_float(  # pylint: disable=too-many-arguments
    weight: te.Tensor,
    bits: int,
    num_elem_per_storage: int,
    storage_dtype: str,
    model_dtype: str,
    axis: int = -1,
    out_shape: Optional[List[tir.PrimExpr]] = None,
    ft_reorder: Optional[bool] = False,
) -> te.Tensor:
    """Convert a quantized uint weight to an unquantized float weight."""
    tir_bin_mask = tir.const((1 << bits) - 1, storage_dtype)
    if out_shape is None:
        out_shape = weight.shape
        out_shape[axis] *= num_elem_per_storage
    axis = axis if axis >= 0 else len(out_shape) + axis
    return te.compute(
        shape=out_shape,
        fcompute=lambda *idx: tir.bitwise_and(
            tir.shift_right(
                weight(*idx[:axis], idx[axis] // num_elem_per_storage, *idx[axis + 1 :]),
                (
                    (
                        (idx[axis] % num_elem_per_storage) % 2 * 4
                        + (idx[axis] % num_elem_per_storage) // 2
                    )
                    * bits
                    if ft_reorder
                    else (idx[axis] % num_elem_per_storage) * bits
                ).astype(storage_dtype),
            ),
            tir_bin_mask,
        ).astype(model_dtype),
    )


def convert_uint_packed_fp8_to_float(  # pylint: disable=too-many-arguments
    weight: te.Tensor,
    bits: int,
    num_elem_per_storage: int,
    storage_dtype: str,
    model_dtype: str,
    quant_dtype: str,
    axis: int = -1,
    out_shape: Optional[List[tir.PrimExpr]] = None,
    ft_reorder: Optional[bool] = False,
) -> te.Tensor:
    """Convert a quantized uint weight tensor to an unquantized e4m3_float8 weight tensor."""
    # Does *not* have FT reoder support right now, can add back in (need to verify bit-match for fp8)
    if ft_reorder:
        raise NotImplementedError()
    assert quant_dtype in ["e4m3_float8", "e5m2_float8"]
    elem_storage_dtype = f"uint{bits}"
    tir_bin_mask = tir.const((1 << bits) - 1, elem_storage_dtype)
    if out_shape is None:
        out_shape = weight.shape
        out_shape[axis] *= num_elem_per_storage
    axis = axis if axis >= 0 else len(out_shape) + axis
    return te.compute(
        shape=out_shape,
        fcompute=lambda *idx: tir.reinterpret(
            DataType(quant_dtype),
            tir.bitwise_and(
                tir.shift_right(
                    weight(*idx[:axis], idx[axis] // num_elem_per_storage, *idx[axis + 1 :]),
                    ((idx[axis] % num_elem_per_storage) * bits).astype(storage_dtype),
                ).astype(elem_storage_dtype),
                tir_bin_mask,
            ),
        ).astype(model_dtype),
    )


def is_final_fc(name: str) -> bool:
    """Determines whether the parameter is the last layer based on its name."""
    # TODO: use more specious condition to determine final fc  # pylint: disable=fixme
    return name in ["head", "lm_head"]


def compile_quantize_func(mod: IRModule, device) -> Callable:
    device_type = device.MASK2STR[device.device_type]
    if device_type in ["cuda", "rocm", "metal", "vulkan"]:
        target = Target.current()
        if target is None:
            target = Target.from_device(device)
        with target:
            mod = dl.ApplyDefaultSchedule(  # type: ignore   # pylint: disable=not-callable
                dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(),
                dl.gpu.Fallback(),
            )(mod)
    elif device_type == "cpu":
        target = "llvm"
        mod = relax.transform.LegalizeOps()(mod)
    else:
        raise NotImplementedError(f"Device type {device_type} is not supported")
    ex = relax.build(mod, target=target)
    vm = relax.VirtualMachine(ex, device)  # pylint: disable=invalid-name
    return vm["main"]


def apply_sharding(shard, name: str, weight: nn.Parameter):
    if isinstance(shard, tp.ShardSingleDim):
        weight.attrs["shard_strategy"] = tp.ShardSingleDim(
            name=name,
            dim=shard.dim,
            segs=shard.segs,
        )
    elif isinstance(shard, tp.ShardScalar):
        weight.attrs["shard_strategy"] = tp.ShardScalar(
            name=name,
        )
    else:
        raise NotImplementedError(f"Unknowing sharding strategy: {shard}")
