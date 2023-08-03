from dataclasses import dataclass
from typing import List, Literal, Optional

import tvm
from tvm import relax, te, tir, topi
from tvm.script import tir as T
from tvm.relax.expr_functor import visitor

from . import tir_utils
from .quantization import QuantizationSpec, QuantSpecUpdater
from .quantization import NoQuantizationSpec
from .quantization import FQuantize, FTEQuantize, FTEDequantize, convert_TE_func


@dataclass
class GroupQuantizationSpec(QuantizationSpec):
    """The quantization specification for group quantization algorithm."""

    mode: Literal["int3", "int4"]
    sym: bool
    storage_nbit: int
    group_size: int
    transpose: bool

    def get_quantize_func(self, param_info: relax.TensorStructInfo) -> Optional[FQuantize]:
        return convert_TE_func(
            encoding_func(
                sym=self.sym,
                group_size=self.group_size,
                nbit=int(self.mode[-1]),
                mode=self.mode,
                storage_nbit=self.storage_nbit,
                transpose=self.transpose,
                dtype=self.dtype,
            ),
            func_name="encode",
        )

    def get_dequantize_func(
        self,
        param_info: relax.TensorStructInfo,
        qparam_info: List[relax.TensorStructInfo],
    ) -> Optional[FQuantize]:
        return convert_TE_func(
            decoding_func(
                sym=self.sym,
                group_size=self.group_size,
                nbit=int(self.mode[-1]),
                mode=self.mode,
                storage_nbit=self.storage_nbit,
                dim_length=param_info.shape.values[-1],
                data_transposed=self.transpose,
                transpose_output=self.transpose,
                dtype=self.dtype,
            ),
            func_name="decode",
        )


# fmt: off
def encoding_func(sym: bool, group_size: int, nbit: int, mode: str, storage_nbit: int, transpose: bool=True, dtype: str = "float32") -> FTEQuantize:
    def te_encode_asym(weight: te.Tensor):
        assert weight.shape[1] % group_size == 0
        n_group = weight.shape[1] // group_size
        n_float_per_u32 = 32 // nbit

        scale_min_shape = (weight.shape[0], n_group)
        k = te.reduce_axis((0, group_size), name="k")
        min_value = te.compute(shape=scale_min_shape, fcompute=lambda i, j: te.min(weight[i, j * group_size + k], axis=k), name="min_value")
        max_value = te.compute(shape=scale_min_shape, fcompute=lambda i, j: te.max(weight[i, j * group_size + k], axis=k), name="max_value")
        scale = te.compute(shape=scale_min_shape, fcompute=lambda i, j: (max_value[i, j] - min_value[i, j]) / tir.const((1 << nbit) - 1, dtype), name="scale")

        def f_scale_weight(i, j):
            group_idx = j // group_size
            w_scaled = tir.round((weight[i, j] - min_value[i, group_idx]) / scale[i, group_idx]).astype("int32")
            w_scaled = T.min(T.max(w_scaled, tir.const(0, "int32")), tir.const((1 << nbit) - 1, "int32"))
            w_scaled = w_scaled.astype("uint32")
            return w_scaled

        k = te.reduce_axis((0, n_float_per_u32), name="k")
        reducer = te.comm_reducer(fcombine=lambda x, y: tir.bitwise_or(x, y), fidentity=lambda dtype: tir.const(0, dtype), name="bitwise_or")
        if dtype == "float32":
            if transpose:
                w_gathered = te.compute(shape=(weight.shape[1] // n_float_per_u32, weight.shape[0]), fcompute=lambda j, i: reducer(f_scale_weight(i, j * n_float_per_u32 + k) << (k * nbit).astype("uint32"), axis=k), name="w_gathered")
                scale_bias = te.compute(shape=(n_group, weight.shape[0]), fcompute=lambda j, i: tir_utils._tir_f32x2_to_bf16x2_to_u32(scale[i, j], min_value[i, j], round_to_even=True), name="scale_min")
            else:
                w_gathered = te.compute(shape=(weight.shape[0], weight.shape[1] // n_float_per_u32), fcompute=lambda i, j: reducer(f_scale_weight(i, j * n_float_per_u32 + k) << (k * nbit).astype("uint32"), axis=k), name="w_gathered")
                scale_bias = te.compute(shape=(weight.shape[0], n_group), fcompute=lambda i, j: tir_utils._tir_f32x2_to_bf16x2_to_u32(scale[i, j], min_value[i, j], round_to_even=True), name="scale_min")
            return w_gathered, scale_bias
        else:
            if transpose:
                w_gathered = te.compute(shape=(weight.shape[1] // n_float_per_u32, weight.shape[0]), fcompute=lambda j, i: reducer(f_scale_weight(i, j * n_float_per_u32 + k) << (k * nbit).astype("uint32"), axis=k), name="w_gathered")
                scale = te.compute(shape=(n_group, weight.shape[0]), fcompute=lambda j, i: scale[i, j], name="scale_transpose")
                min_value = te.compute(shape=(n_group, weight.shape[0]), fcompute=lambda j, i: min_value[i, j], name="min_transpose")
            else:
                w_gathered = te.compute(shape=(weight.shape[0], weight.shape[1] // n_float_per_u32), fcompute=lambda i, j: reducer(f_scale_weight(i, j * n_float_per_u32 + k) << (k * nbit).astype("uint32"), axis=k), name="w_gathered")
            return w_gathered, scale, min_value

    def te_encode_sym(weight: te.Tensor):
        n_group = tir.ceildiv(weight.shape[1], group_size)
        n_float_per_int = storage_nbit // nbit
        max_int_value = (1 << (nbit - 1)) - 1
        assert group_size % n_float_per_int == 0

        scale_min_shape = (weight.shape[0], n_group)
        k = te.reduce_axis((0, group_size), name="k")
        max_abs_value = te.compute(shape=scale_min_shape, fcompute=lambda i, j: te.max(tir.if_then_else(j * group_size + k < weight.shape[1], te.abs(weight[i, j * group_size + k]), tir.min_value(dtype)), axis=k), name="max_abs_value")

        def f_compute_scale(i, j):
            max_value = tir.max(max_abs_value[i, j], tir.const(1e-4, dtype))
            return (max_value / tir.const(max_int_value, dtype)) if mode.startswith("int") else max_value

        scale = te.compute(shape=scale_min_shape, fcompute=f_compute_scale, name="scale")
        storage_dtype = ("uint" + str(storage_nbit)) if mode.startswith("int") else "uint32"

        def f_scale_weight(i, j):
            group_idx = j // group_size
            if mode.startswith("int"):
                w_scaled = tir.round(weight[i, j] / scale[i, group_idx] + tir.const(max_int_value, dtype))
                w_scaled = T.min(T.max(w_scaled, tir.const(0, dtype)), tir.const(max_int_value * 2, dtype)).astype(storage_dtype)
                return w_scaled
            else:
                f_convert = tir_utils._tir_f32_to_uint_to_f4 if dtype == "float32" else tir_utils._tir_f16_to_uint_to_f4
                return f_convert(weight[i, j] / scale[i, group_idx])

        k = te.reduce_axis((0, n_float_per_int), name="k")
        reducer = te.comm_reducer(fcombine=lambda x, y: tir.bitwise_or(x, y), fidentity=lambda dtype: tir.const(0, dtype), name="bitwise_or")
        n_i32 = tir.ceildiv(group_size, n_float_per_int) * n_group
        if transpose:
            w_gathered = te.compute(shape=(n_i32, weight.shape[0]), fcompute=lambda j, i: reducer(tir.if_then_else(j * n_float_per_int + k < weight.shape[1], f_scale_weight(i, j * n_float_per_int + k) << (k.astype(storage_dtype) * tir.const(nbit, storage_dtype)), tir.const(0, storage_dtype)), axis=k), name="w_gathered")
            scale = te.compute(shape=(n_group, weight.shape[0]), fcompute=lambda j, i: scale[i, j])
        else:
            w_gathered = te.compute(shape=(weight.shape[0], n_i32), fcompute=lambda i, j: reducer(tir.if_then_else(j * n_float_per_int + k < weight.shape[1], f_scale_weight(i, j * n_float_per_int + k) << (k.astype(storage_dtype) * tir.const(nbit, storage_dtype)), tir.const(0, storage_dtype)), axis=k), name="w_gathered")
        return w_gathered, scale

    return te_encode_sym if sym else te_encode_asym


def decoding_func(sym: bool, group_size: int, nbit: int, mode: str, storage_nbit: int, dim_length: tir.PrimExpr, data_transposed: bool=True, transpose_output: bool=False, dtype: str = "float32") -> FTEDequantize:
    def te_decode_asym(*args):
        n_float_per_u32 = 32 // nbit
        data = args[0]
        if dtype == "float32":
            scale_bias_bf16x2 = args[1]
        else:
            scale, min_value = args[1], args[2]

        def f_decode_asym(i, j):
            if data_transposed:
                data_float = tir_utils._tir_u32_to_int_to_float(nbit, data[i // n_float_per_u32, j], i % n_float_per_u32, dtype=dtype)
                if dtype == "float32":
                    scale_float, bias_float = tir_utils._tir_u32_to_bf16x2_to_f32x2(scale_bias_bf16x2[i // group_size, j])
                else:
                    scale_float, bias_float = scale[i // group_size, j], min_value[i // group_size, j]
            else:
                data_float = tir_utils._tir_u32_to_int_to_float(nbit, data[i, j // n_float_per_u32], j % n_float_per_u32, dtype=dtype)
                if dtype == "float32":
                    scale_float, bias_float = tir_utils._tir_u32_to_bf16x2_to_f32x2(scale_bias_bf16x2[i, j // group_size])
                else:
                    scale_float, bias_float = scale[i, j // group_size], min_value[i, j // group_size]
            w = data_float * scale_float + bias_float
            return w

        shape = (dim_length, data.shape[1]) if data_transposed else (data.shape[0], dim_length)
        w = te.compute(shape=shape, fcompute=f_decode_asym, name="decode")
        if transpose_output:
            w = topi.transpose(w)
        return w

    def te_decode_sym(data, scale):
        n_float_per_int = storage_nbit // nbit

        def f_decode_sym(i, j):
            f_convert = tir_utils._tir_packed_uint_to_uint_to_float(storage_nbit) if mode.startswith("int") else (tir_utils._tir_u32_to_f4_to_f32 if dtype == "float32" else tir_utils._tir_u32_to_f4_to_f16)
            if data_transposed:
                data_float = f_convert(nbit, data[i // n_float_per_int, j], i % n_float_per_int, dtype=dtype)
                scale_float = scale[i // group_size, j]
            else:
                data_float = f_convert(nbit, data[i, j // n_float_per_int], j % n_float_per_int, dtype=dtype)
                scale_float = scale[i, j // group_size]
            return data_float * scale_float

        shape = (dim_length, data.shape[1]) if data_transposed else (data.shape[0], dim_length)
        w = te.compute(shape=shape, fcompute=f_decode_sym, name="decode")
        if transpose_output:
            w = topi.transpose(w)
        return w

    return te_decode_sym if sym else te_decode_asym
# fmt: on


# A simple example demo showing how QuantSpecUpdater is used.
# NOTE: This visitor is only for demo purpose and should not be put into real use.
@visitor
class GroupQuantDemoUpdater(QuantSpecUpdater._cls):
    def visit_call_(self, call: relax.Call):
        if call.op != tvm.ir.Op.get("relax.matmul"):
            return
        rhs = self.lookup_binding(call.args[1])
        assert rhs is not None
        if (
            rhs.op != tvm.ir.Op.get("relax.permute_dims")
            or rhs.attrs.axes is not None
            or rhs.args[0].struct_info.ndim != 2
        ):
            return

        if rhs.args[0] not in self.param_map:
            return
        param = self.param_map[rhs.args[0]]
        # Update to no quantization for matmul with float32 output dtype.
        if call.struct_info.dtype == "float32":
            param.quant_spec = NoQuantizationSpec(param.param_info.dtype)
