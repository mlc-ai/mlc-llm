"""Fusing and inlining transpose function into decode function."""
import tvm
from tvm import relax, tir
from tvm.ir.module import IRModule
from tvm.relax.analysis import remove_all_unused
from tvm.relax.expr_functor import PyExprMutator, mutator


@tvm.transform.module_pass(opt_level=0, name="FuseDecodeTranspose")
class FuseDecodeTranspose:
    def transform_module(
        self, mod: IRModule, ctx: tvm.transform.PassContext
    ) -> IRModule:
        @mutator
        class DecodeTransposeFusor(PyExprMutator):
            def __init__(self, mod: IRModule):
                super().__init__(mod)
                self.mod = mod

            def transform(self) -> IRModule:
                for gv, func in self.mod.functions.items():
                    if not isinstance(func, relax.Function):
                        continue

                    updated_func = self.visit_expr(func)
                    updated_func = remove_all_unused(updated_func)
                    self.builder_.update_func(gv, updated_func)

                return self.builder_.get()

            def visit_call_(self, call: relax.Call) -> relax.Expr:
                call = self.visit_expr_post_order(call)

                if call.op != tvm.ir.Op.get("relax.matmul"):
                    return call

                # Do not fuse decode-transpose for GeMM
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
                    or not transpose_input.args[0].name_hint.startswith("decode")
                    or not isinstance(
                        transpose_input.struct_info, relax.TensorStructInfo
                    )
                ):
                    return call

                decode_tir_func = self.mod[transpose_input.args[0]]
                assert isinstance(decode_tir_func, tir.PrimFunc)
                if (
                    len(decode_tir_func.body.block.alloc_buffers) != 1
                    or not isinstance(decode_tir_func.body.block.body, tir.SeqStmt)
                    or len(decode_tir_func.body.block.body) != 2
                    or not isinstance(decode_tir_func.body.block.body[1], tir.For)
                    or not isinstance(
                        decode_tir_func.body.block.body[1].body.body, tir.BlockRealize
                    )
                    or decode_tir_func.body.block.body[1].body.body.block.name_hint
                    != "T_transpose"
                ):
                    return call
                print(
                    "====================== debug print for decode ======================"
                )
                sch = tvm.tir.Schedule(decode_tir_func)
                br = sch.get_block("root")
                bt = sch.get_child_blocks(br)[-1]
                bd = sch.get_producers(bt)
                new_func_buffers = [
                    decode_tir_func.buffer_map[var] for var in decode_tir_func.params
                ]
                new_func_buffers[-1] = decode_tir_func.body.block.alloc_buffers[0]
                old_body = decode_tir_func.body.block.body[0]
                # deep copy old_body to new_body
                # new_body = 
                
                new_body = tir.For(
                    loop_var=old_body.loop_var,
                    min_val=old_body.min,
                    extent=old_body.extent,
                    kind=old_body.kind,
                    thread_binding=old_body.thread_binding,
                    annotations=old_body.annotations,
                    body=tir.For(
                        loop_var=old_body.body.loop_var,
                        min_val=old_body.body.min,
                        extent=old_body.body.extent,
                        kind=old_body.body.kind,
                        thread_binding=old_body.body.thread_binding,
                        annotations=old_body.body.annotations,
                        body=tir.BlockRealize(
                            iter_values=old_body.body.body.iter_values,
                            predicate=old_body.body.body.predicate,
                            block=tir.Block(
                                iter_vars=old_body.body.body.block.iter_vars,
                                reads=old_body.body.body.block.reads,
                                writes=old_body.body.body.block.writes,
                                name_hint=old_body.body.body.block.name_hint,
                                body=old_body.body.body.block.body,
                            ),
                        ),
                    ),
                )

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
                            body=new_body,
                        ),
                    ),
                )
                # new_func = tir.PrimFunc(
                #     params=new_func_buffers,
                #     body=tir.BlockRealize(
                #         iter_values=[],
                #         predicate=True,
                #         block=tir.Block(
                #             iter_vars=[],
                #             reads=[],
                #             writes=[],
                #             name_hint="root",
                #             body=decode_tir_func.body.block.body[0],
                #         ),
                #     ),
                # )
                gv = self.builder_.add_func(new_func, func_name="decode")
                decoded_matmul_rhs = self.builder_.emit(
                    relax.call_tir(
                        gv, transpose_input.args[1], out_sinfo=matmul_rhs.struct_info
                    )
                )
                print(sch.get_sref(bd[0]).parent.parent.parent.stmt)
                print("parent scope", sch.get_sref(bd[0]).parent.parent.parent)
                print("new func scope", new_func)

                return relax.op.matmul(
                    call.args[0], decoded_matmul_rhs, out_dtype=call.attrs.out_dtype
                )

        return DecodeTransposeFusor(mod).transform()
