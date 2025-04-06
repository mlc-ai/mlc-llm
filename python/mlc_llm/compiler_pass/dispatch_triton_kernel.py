"""A pass that dispatch generic calls of triton kernels to specific kernel implementations."""

# pylint: disable=invalid-name

from typing import List

import tvm
from tvm import IRModule, relax
from tvm.relax.expr_functor import PyExprMutator, mutator

from mlc_llm.op.triton import (
    get_tir_w8a8_block_fp8_group_matmul,
    get_tir_w8a8_block_fp8_matmul,
)
from mlc_llm.support import logging

logger = logging.getLogger(__name__)


@mutator
class _Rewriter(PyExprMutator):  # pylint: disable=abstract-method
    def __init__(self, mod: IRModule, target: tvm.target.Target) -> None:
        super().__init__(mod)
        self.mod = mod
        self.target = target
        self.extern_mods: List[tvm.runtime.Module] = []

    def transform(self) -> tvm.IRModule:  # pylint: disable=too-many-locals
        """Entry point of the transformation"""
        for g_var, func in self.mod.functions_items():
            if not isinstance(func, relax.Function):
                continue
            new_func = self.visit_expr(func)
            # new_func = remove_all_unused(new_func)
            self.builder_.update_func(g_var, new_func)

        mod = self.builder_.finalize()
        mod_attrs = dict(mod.attrs) if mod.attrs else {}
        mod = mod.with_attr(
            "external_mods", list(mod_attrs.get("external_mods", [])) + self.extern_mods
        )
        return mod

    def visit_call_(self, call: relax.Call) -> relax.Expr:  # pylint: disable=arguments-renamed
        call = super().visit_call_(call)

        if (
            call.op != tvm.ir.Op.get("relax.call_dps_packed")
            or not isinstance(call.args[0], relax.ExternFunc)
            or not str(call.args[0].global_symbol).startswith("mlc.triton.")
        ):
            return call

        global_symbol = str(call.args[0].global_symbol)
        assert isinstance(call.args[1], relax.Tuple)
        if global_symbol == "mlc.triton.w8a8_block_fp8_matmul":
            return self.w8a8_block_fp8_matmul(call.args[1].fields, call.struct_info)
        if global_symbol == "mlc.triton.w8a8_block_fp8_group_matmul":
            return self.w8a8_block_fp8_group_matmul(call.args[1].fields, call.struct_info)
        raise ValueError(f"Unknown mlc.triton kernel identifier: {global_symbol}")

    def w8a8_block_fp8_matmul(  # pylint: disable=too-many-locals
        self, args: List[relax.Expr], out_sinfo: relax.StructInfo
    ) -> relax.Expr:
        """Emit the w8a8_block_fp8_matmul triton kernel."""
        assert len(args) == 16
        x, weight, x_scale, weight_scale = args[:4]
        (
            N,
            K,
            block_n,
            block_k,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE_M,
            num_warps,
            num_stages,
        ) = [arg.value.value for arg in args[4:14]]
        in_dtype, out_dtype = str(args[14].value), str(args[15].value)

        prim_func, func_name = get_tir_w8a8_block_fp8_matmul(
            N,
            K,
            block_n,
            block_k,
            in_dtype,  # type: ignore
            out_dtype,  # type: ignore
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE_M,
            num_warps,
            num_stages,
            self.extern_mods,
        )
        if prim_func is None:
            # The TIR function is already in the IRModule
            gv = self.builder_.get().get_global_var(func_name)
        else:
            # Add the TIR function to the IRModule
            gv = self.builder_.add_func(prim_func, func_name)

        return relax.call_tir(gv, [x, weight, x_scale, weight_scale], out_sinfo=out_sinfo)

    def w8a8_block_fp8_group_matmul(  # pylint: disable=too-many-locals
        self, args: List[relax.Expr], out_sinfo: relax.StructInfo
    ) -> relax.Expr:
        """Emit the w8a8_block_fp8_group_matmul triton kernel."""
        assert len(args) == 19
        x, weight, x_scale, weight_scale, expert_ids, indptr = args[:6]
        (
            N,
            K,
            num_experts,
            block_n,
            block_k,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE_M,
            num_warps,
            num_stages,
        ) = [arg.value.value for arg in args[6:17]]
        in_dtype, out_dtype = str(args[17].value), str(args[18].value)

        prim_func, func_name = get_tir_w8a8_block_fp8_group_matmul(
            N,
            K,
            num_experts,
            block_n,
            block_k,
            in_dtype,  # type: ignore
            out_dtype,  # type: ignore
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE_M,
            num_warps,
            num_stages,
            self.extern_mods,
        )
        if prim_func is None:
            # The TIR function is already in the IRModule
            gv = self.builder_.get().get_global_var(func_name)
        else:
            # Add the TIR function to the IRModule
            gv = self.builder_.add_func(prim_func, func_name)

        return relax.call_tir(
            gv,
            [x, weight, x_scale, weight_scale, expert_ids, indptr],
            out_sinfo=out_sinfo,
        )


@tvm.transform.module_pass(opt_level=0, name="DispatchTritonKernel")
class DispatchTritonKernel:  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """Rewrite KV cache creation functions to IRModule."""

    def __init__(self, target: tvm.target.Target) -> None:
        """Initializer.

        Parameters
        ----------
        """
        self.target = target

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """Entrypoint"""
        if self.target.kind.name != "cuda":
            return mod

        return _Rewriter(mod, self.target).transform()
