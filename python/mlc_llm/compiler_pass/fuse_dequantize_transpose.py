"""A compiler pass that fuses transpose + dequantize."""

import tvm
from tvm import relax, tir
from tvm.ir.module import IRModule
from tvm.relax.analysis import remove_all_unused
from tvm.relax.expr_functor import PyExprMutator, mutator


@tvm.transform.module_pass(opt_level=0, name="FuseDequantizeTranspose")
class FuseDequantizeTranspose:  # pylint: disable=too-few-public-methods
    """A compiler pass that fuses transpose + dequantize."""

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """IRModule-level transformation"""
        return _DequantizeTransposeFuser(mod).transform()


@mutator
class _DequantizeTransposeFuser(PyExprMutator):  # pylint: disable=abstract-method
    def __init__(
        self,
        mod: IRModule,
    ):
        super().__init__(mod)
        self.mod = mod

    def transform(self) -> IRModule:
        """Entry point"""
        for g_var, func in self.mod.functions_items():
            if isinstance(func, relax.Function):
                updated_func = self.visit_expr(func)
                updated_func = remove_all_unused(updated_func)
                self.builder_.update_func(g_var, updated_func)
        return self.builder_.get()

    def visit_call_(  # pylint: disable=arguments-renamed
        self,
        call: relax.Call,
    ) -> relax.Expr:
        call = self.visit_expr_post_order(call)
        if call.op != tvm.ir.Op.get("relax.matmul"):
            return call
        # Do not fuse dequantize-transpose for GeMM
        if (
            call.args[0].struct_info.ndim < 2
            or not isinstance(call.args[0].struct_info.shape[-2], tir.IntImm)
            or call.args[0].struct_info.shape[-2].value != 1
        ):
            return call

        matmul_rhs = self.lookup_binding(call.args[1])
        if (
            not isinstance(matmul_rhs, relax.Call)
            or matmul_rhs.op != tvm.ir.Op.get("relax.permute_dims")
            or matmul_rhs.args[0].struct_info.ndim != 2
            or matmul_rhs.attrs.axes is not None
        ):
            return call

        transpose_input = self.lookup_binding(matmul_rhs.args[0])
        if (
            not isinstance(transpose_input, relax.Call)
            or transpose_input.op != tvm.ir.Op.get("relax.call_tir")
            or not transpose_input.args[0].name_hint.startswith("dequantize")
            or not isinstance(transpose_input.struct_info, relax.TensorStructInfo)
        ):
            return call

        dequantize_tir_func = self.mod[transpose_input.args[0]]
        assert isinstance(dequantize_tir_func, tir.PrimFunc)
        if (  # pylint: disable=too-many-boolean-expressions
            len(dequantize_tir_func.body.block.alloc_buffers) != 1
            or not isinstance(dequantize_tir_func.body.block.body, tir.SeqStmt)
            or len(dequantize_tir_func.body.block.body) != 2
            or not isinstance(dequantize_tir_func.body.block.body[1], tir.For)
            or not isinstance(dequantize_tir_func.body.block.body[1].body.body, tir.BlockRealize)
            or dequantize_tir_func.body.block.body[1].body.body.block.name_hint != "T_transpose"
        ):
            return call

        new_func_buffers = [
            dequantize_tir_func.buffer_map[var] for var in dequantize_tir_func.params
        ]
        new_func_buffers[-1] = dequantize_tir_func.body.block.alloc_buffers[0]
        new_func = tir.PrimFunc(
            params=new_func_buffers,
            body=tir.BlockRealize(
                iter_values=[],
                predicate=True,
                block=tir.Block(
                    iter_vars=[],
                    reads=[],
                    writes=[],
                    name_hint="root",
                    body=dequantize_tir_func.body.block.body[0],
                ),
            ),
        )
        # Call `renew_defs` for deep-copy to avoid IR node duplication in
        # different PrimFuncs of an IRModule.
        new_func = tir.stmt_functor.renew_defs(new_func)
        g_var = self.builder_.add_func(new_func, func_name="dequantize")
        dequantize_matmul_rhs = self.builder_.emit(
            relax.call_tir(g_var, transpose_input.args[1], out_sinfo=matmul_rhs.struct_info)
        )
        return relax.op.matmul(call.args[0], dequantize_matmul_rhs, out_dtype=call.attrs.out_dtype)
