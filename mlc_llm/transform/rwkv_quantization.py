"""Relax quantization passes."""

# pylint: disable=missing-docstring, abstract-method

from typing import Tuple

import tvm
from tvm import relax, te, tir, topi
from tvm.ir.module import IRModule
from tvm.relax.analysis import remove_all_unused
from tvm.relax.expr_functor import PyExprMutator, mutator
from tvm.relax.op.builtin import stop_lift_params


def encoding_func(input_dtype: str, quant_dtype: str):
    assert (input_dtype, quant_dtype) == ("float16", "uint8")
    nbit = 8
    max_value = 1 << nbit
    assert nbit % 2 == 0
    scale = 1 << (nbit // 2)

    def encode(
        weight: te.Tensor,
    ) -> Tuple[te.Tensor, te.Tensor, te.Tensor, te.Tensor, te.Tensor]:
        weight = weight.astype("float32")
        if weight.shape[0] > weight.shape[1]:
            min_y = topi.min(weight, axis=1, keepdims=True)
            weight = weight - min_y
            min_x = topi.min(weight, axis=0, keepdims=True)
            weight = weight - min_x
        else:
            min_x = topi.min(weight, axis=0, keepdims=True)
            weight = weight - min_x
            min_y = topi.min(weight, axis=1, keepdims=True)
            weight = weight - min_y
        max_x = topi.max(weight, axis=0, keepdims=True)
        weight = weight / max_x
        max_y = topi.max(weight, axis=1, keepdims=True)
        weight = weight / max_y
        weight = topi.clip(topi.floor(weight * max_value), 0, max_value - 1).astype(
            quant_dtype
        )
        min_x = min_x.astype(input_dtype)
        min_y = min_y.astype(input_dtype)
        max_x = topi.divide(max_x, scale).astype(input_dtype)
        max_y = topi.divide(max_y, scale).astype(input_dtype)
        return weight, min_x, max_x, min_y, max_y

    return encode


def emit_encoding(builder: relax.BlockBuilder, weight: relax.Expr, quant_dtype: str):
    input_dtype = weight.struct_info.dtype
    data = builder.emit_te(
        encoding_func(input_dtype, quant_dtype),
        weight,
        primfunc_name_hint="quant_encoding",
    )
    encoded_data = []
    for i in range(5):
        encoded_data.append(builder.emit(stop_lift_params(relax.TupleGetItem(data, i))))
    return encoded_data


def decoding_func(input_dtype, quant_dtype: str):
    assert (input_dtype, quant_dtype) == ("float16", "uint8")

    def decode(
        weight: te.Tensor,
        min_x: te.Tensor,
        max_x: te.Tensor,
        min_y: te.Tensor,
        max_y: te.Tensor,
    ) -> te.Tensor:
        x = weight.astype(input_dtype) + tir.const(0.5, input_dtype)
        return x * max_y * max_x + min_y + min_x

    return decode


@tvm.transform.module_pass(opt_level=0, name="RWKVQuantize")
class RWKVQuantize:
    def __init__(
        self,
        mode: str = "uint8",
        dtype: str = "float16",
    ) -> None:
        self.mode = mode
        self.dtype = dtype

    def transform_module(
        self,
        mod: IRModule,
        ctx: tvm.transform.PassContext,  # pylint: disable=unused-argument
    ) -> IRModule:
        @mutator
        class QuantizeMutator(PyExprMutator):
            def __init__(
                self,
                mod: IRModule,
                mode: str,
                dtype: str,
            ):
                super().__init__(mod)
                self.mod = mod
                self._params = set()
                self.nbit = int(mode[-1])
                self.mode = mode
                self.dtype = dtype

            def transform(self) -> IRModule:
                for global_var, func in self.mod.functions.items():
                    if not isinstance(func, relax.Function):
                        continue
                    if func.attrs is None or not "num_input" in func.attrs:
                        continue
                    num_inputs = func.attrs["num_input"]
                    for i in range(int(num_inputs), len(func.params)):
                        self._params.add(func.params[i])
                    updated_func = self.visit_expr(func)
                    updated_func = remove_all_unused(updated_func)
                    self.builder_.update_func(global_var, updated_func)
                return self.builder_.get()

            def quantize_matmul(self, call):
                weight = call.args[1]
                conditions = [weight.struct_info.ndim == 2, weight in self._params]
                if not all(conditions):
                    return call
                encoded_weight = emit_encoding(self.builder_, weight, self.mode)
                decoded_weight = self.builder_.emit_te(
                    decoding_func(self.dtype, self.mode), *encoded_weight
                )
                return relax.op.matmul(call.args[0], decoded_weight)

            def visit_call_(self, op):
                call = self.visit_expr_post_order(op)

                if call.op == tvm.ir.Op.get("relax.matmul"):
                    return self.quantize_matmul(call)
                else:
                    return call

        return QuantizeMutator(mod, self.mode, self.dtype).transform()
