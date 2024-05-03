"""A compiler pass that rewrites one-shot softmax into two-stage softmax."""

import tvm
from tvm import relax
from tvm.ir.module import IRModule
from tvm.relax.expr import Expr
from tvm.relax.expr_functor import PyExprMutator, mutator
from tvm.script import tir as T

from ..support.max_thread_check import get_max_num_threads_per_block


@tvm.transform.module_pass(opt_level=0, name="RewriteTwoStageSoftmax")
class RewriteTwoStageSoftmax:  # pylint: disable=too-few-public-methods
    """Rewrites one-shot softmax into two-stage softmax."""

    def __init__(self, target: tvm.target.Target) -> None:
        self.target = target

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """IRModule-level transformation"""
        return _Rewriter(mod, self.target).transform()


@mutator
class _Rewriter(PyExprMutator):  # pylint: disable=abstract-method
    def __init__(self, mod: IRModule, target: tvm.target.Target) -> None:
        super().__init__(mod)
        self.mod = mod
        self.target = target
        self.chunk_size = 4096

    def transform(self) -> IRModule:
        """Entry point"""
        func_name = "softmax_with_temperature"
        if func_name not in self.mod:
            return self.mod
        gv = self.mod.get_global_var(func_name)
        updated_func = self.visit_expr(self.mod[gv])
        self.builder_.update_func(gv, updated_func)
        return self.builder_.get()

    def visit_call_(self, call: relax.Call) -> Expr:  # pylint: disable=arguments-renamed
        if call.op != tvm.ir.Op.get("relax.nn.softmax"):
            return call
        x = call.args[0]
        if call.attrs.axis not in [-1, x.struct_info.ndim - 1]:
            return call
        # Currently the softmax input is 3-dim, and dtype is float32.
        assert x.struct_info.ndim == 3
        assert x.struct_info.dtype == "float32"
        x_shape = x.struct_info.shape
        new_shape = relax.ShapeExpr([x_shape[0] * x_shape[1], x_shape[2]])
        x_reshaped = relax.call_pure_packed(
            "vm.builtin.reshape",
            x,
            new_shape,
            sinfo_args=relax.TensorStructInfo(new_shape, x.struct_info.dtype),
        )
        f_chunk_lse, f_softmax_with_lse = _get_lse_and_softmax_func(self.target, self.chunk_size)
        chunked_lse = relax.call_tir(
            self.builder_.add_func(f_chunk_lse, "chunk_lse"),
            args=[x_reshaped],
            out_sinfo=relax.TensorStructInfo(
                (new_shape[0], (new_shape[1] + self.chunk_size - 1) // self.chunk_size),
                x.struct_info.dtype,
            ),
        )
        softmax = relax.call_tir(
            self.builder_.add_func(f_softmax_with_lse, "softmax_with_chunked_lse"),
            args=[x_reshaped, chunked_lse],
            out_sinfo=relax.TensorStructInfo(new_shape, x.struct_info.dtype),
        )
        return relax.call_pure_packed(
            "vm.builtin.reshape", softmax, x_shape, sinfo_args=x.struct_info
        )


def _get_lse_and_softmax_func(  # pylint: disable=too-many-locals,too-many-statements
    target: tvm.target.Target, chunk_size: int
):
    # pylint: disable=invalid-name
    @T.prim_func
    def chunk_lse(var_A: T.handle, var_chunked_lse: T.handle):  # pylint: disable=too-many-locals
        T.func_attr({"tir.noalias": T.bool(True)})
        batch_size = T.int64(is_size_var=True)
        vocab_size = T.int64(is_size_var=True)
        num_chunks = T.int64(is_size_var=True)
        A = T.match_buffer(var_A, (batch_size, vocab_size), dtype="float32")
        chunked_lse = T.match_buffer(var_chunked_lse, (batch_size, num_chunks), dtype="float32")
        A_pad = T.alloc_buffer((batch_size, num_chunks, T.int64(chunk_size)), dtype="float32")
        temp_max = T.alloc_buffer((batch_size, num_chunks), dtype="float32")
        temp_sum = T.alloc_buffer((batch_size, num_chunks), dtype="float32")

        for l0, l1, l2 in T.grid(batch_size, num_chunks, T.int64(chunk_size)):
            with T.block("pad"):
                v0, v1, v2 = T.axis.remap("SSS", [l0, l1, l2])
                A_pad[v0, v1, v2] = T.if_then_else(
                    v1 * T.int64(chunk_size) + v2 < vocab_size,
                    A[v0, v1 * T.int64(chunk_size) + v2],
                    T.min_value("float32"),
                )
        for l0, l1, l2 in T.grid(batch_size, num_chunks, T.int64(chunk_size)):
            with T.block("max"):
                v0, v1, v2 = T.axis.remap("SSR", [l0, l1, l2])
                with T.init():
                    temp_max[v0, v1] = T.min_value("float32")
                temp_max[v0, v1] = T.max(temp_max[v0, v1], A_pad[v0, v1, v2])
        for l0, l1, l2 in T.grid(batch_size, num_chunks, T.int64(chunk_size)):
            with T.block("sum_exp"):
                v0, v1, v2 = T.axis.remap("SSR", [l0, l1, l2])
                with T.init():
                    temp_sum[v0, v1] = T.float32(0)
                temp_sum[v0, v1] += T.if_then_else(
                    v1 * T.int64(chunk_size) + v2 < vocab_size,
                    T.exp(A_pad[v0, v1, v2] - temp_max[v0, v1]),
                    T.float32(0),
                )
        for l0, l1, l2 in T.grid(batch_size, num_chunks, T.int64(1)):
            with T.block("log"):
                v0, v1, v2 = T.axis.remap("SSS", [l0, l1, l2])
                chunked_lse[v0, v1] = T.log(temp_sum[v0, v1]) + temp_max[v0, v1]

    @T.prim_func
    def softmax_with_chunked_lse(var_A: T.handle, var_chunked_lse: T.handle, var_softmax: T.handle):
        T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
        batch_size = T.int64(is_size_var=True)
        vocab_size = T.int64(is_size_var=True)
        num_chunks = T.int64(is_size_var=True)
        A = T.match_buffer(var_A, (batch_size, vocab_size), dtype="float32")
        chunked_lse = T.match_buffer(var_chunked_lse, (batch_size, num_chunks), dtype="float32")
        softmax = T.match_buffer(var_softmax, (batch_size, vocab_size), dtype="float32")
        temp_max = T.alloc_buffer((batch_size,), dtype="float32")
        temp_sum = T.alloc_buffer((batch_size,), dtype="float32")
        lse = T.alloc_buffer((batch_size,), dtype="float32")
        for l0, l1 in T.grid(batch_size, num_chunks):
            with T.block("max"):
                v0, v1 = T.axis.remap("SR", [l0, l1])
                with T.init():
                    temp_max[v0] = T.min_value("float32")
                temp_max[v0] = T.max(temp_max[v0], chunked_lse[v0, v1])
        for l0, l1 in T.grid(batch_size, num_chunks):
            with T.block("sum_exp"):
                v0, v1 = T.axis.remap("SR", [l0, l1])
                with T.init():
                    temp_sum[v0] = T.float32(0)
                temp_sum[v0] += T.exp(chunked_lse[v0, v1] - temp_max[v0])
        for l0 in T.serial(0, batch_size):
            with T.block("log"):
                v0 = T.axis.remap("S", [l0])
                lse[v0] = T.log(temp_sum[v0]) + temp_max[v0]
        for l0, l1, l2 in T.grid(batch_size, num_chunks, T.int64(chunk_size)):
            with T.block("pad"):
                v0, v1, v2 = T.axis.remap("SSS", [l0, l1, l2])
                if v1 * T.int64(chunk_size) + v2 < vocab_size:
                    softmax[v0, v1 * T.int64(chunk_size) + v2] = T.exp(
                        A[v0, v1 * T.int64(chunk_size) + v2] - lse[v0]
                    )

    sch = tvm.tir.Schedule(IRModule({"softmax_with_chunked_lse": softmax_with_chunked_lse}))
    max_threads = get_max_num_threads_per_block(target)
    TX = 32
    TY = max_threads // TX
    unroll_depth = 64
    # pylint: enable=invalid-name

    sch.work_on("softmax_with_chunked_lse")
    sch.compute_inline("log")
    l0, l1, l2 = sch.get_loops("pad")
    bx = sch.fuse(l0, l1)
    sch.bind(bx, "blockIdx.x")
    unroll, ty, tx = sch.split(l2, [None, TY, TX])
    sch.bind(ty, "threadIdx.y")
    sch.bind(tx, "threadIdx.x")
    sch.annotate(unroll, ann_key="pragma_auto_unroll_max_step", ann_val=unroll_depth)
    sch.annotate(unroll, ann_key="pragma_unroll_explicit", ann_val=1)

    for block_name in ["sum_exp", "max"]:
        block = sch.get_block(block_name)
        sch.set_scope(block, buffer_index=0, storage_scope="shared")
        sch.compute_at(block, bx)
        r_loop = sch.get_loops(block)[-1]
        r_loop, tx = sch.split(r_loop, [None, TX])
        sch.reorder(tx, r_loop)
        sch.bind(tx, "threadIdx.x")
        sch.annotate(r_loop, ann_key="pragma_auto_unroll_max_step", ann_val=unroll_depth)
        sch.annotate(r_loop, ann_key="pragma_unroll_explicit", ann_val=1)

    return chunk_lse, sch.mod["softmax_with_chunked_lse"]
