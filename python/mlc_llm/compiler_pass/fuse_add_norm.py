"""A compiler pass that fuses add + rms_norm."""

# pylint: disable=invalid-name

from typing import Optional

import tvm
from tvm import relax
from tvm.relax.analysis import remove_all_unused
from tvm.relax.expr_functor import PyExprMutator, mutator
from tvm.script import tir as T

from ..support.max_thread_check import get_max_num_threads_per_block


def _get_add_rms_norm_decode(hidden_size: int, eps: float, TX: int, in_dtype: str):
    if in_dtype not in ("float16", "bfloat16"):
        raise ValueError(f"Unsupported data type: {in_dtype}")
    inv_hidden_size = T.float32(1.0 / float(hidden_size))
    eps = T.float32(eps)
    add_local_size = hidden_size // TX

    @T.prim_func(private=True)
    def decode_add_rms(  # pylint: disable=too-many-locals
        pA: T.handle, pB: T.handle, pC: T.handle, pO: T.handle, pAdd: T.handle
    ):
        T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
        batch_size = T.int32()
        A = T.match_buffer(pA, (batch_size, 1, hidden_size), in_dtype)
        B = T.match_buffer(pB, (batch_size, 1, hidden_size), in_dtype)
        C = T.match_buffer(pC, (hidden_size,), in_dtype)
        O = T.match_buffer(pO, (batch_size, 1, hidden_size), in_dtype)
        add = T.match_buffer(pAdd, (batch_size, 1, hidden_size), in_dtype)
        add_local = T.alloc_buffer((hidden_size // TX,), in_dtype, scope="local")
        sum_shared = T.alloc_buffer((batch_size, 1), scope="shared")
        sum_local = T.alloc_buffer((TX, batch_size, 1), scope="local")
        for v_bx in T.thread_binding(batch_size, thread="blockIdx.x"):
            for v_tx in T.thread_binding(
                TX,
                thread="threadIdx.x",
                annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1},
            ):
                for i in range(add_local_size):
                    with T.block("T_add"):
                        bx = T.axis.spatial(batch_size, v_bx)
                        h = T.axis.spatial(hidden_size, i * TX + v_tx)
                        add_local[h // TX] = A[bx, 0, h] + B[bx, 0, h]
                    with T.block("T_write_back"):
                        bx = T.axis.spatial(batch_size, v_bx)
                        v_ax1 = T.axis.spatial(1, 0)
                        h = T.axis.spatial(hidden_size, i * TX + v_tx)
                        add[bx, v_ax1, h] = add_local[h // TX]
                with T.block("T_multiply_red_rf_init"):
                    tx, bx = T.axis.remap("SS", [v_tx, v_bx])
                    sum_local[tx, bx, 0] = T.float32(0)
                for v_i, _j in T.grid(add_local_size, 1):
                    with T.block("T_multiply_red_rf_update"):
                        tx, bx, i = T.axis.remap("SSR", [v_tx, v_bx, v_i])
                        sum_local[tx, bx, 0] += T.float32(add_local[i]) * T.float32(add_local[i])
            for _j in range(1):
                for v_tx_2 in T.thread_binding(TX, thread="threadIdx.x"):
                    with T.block("T_multiply_red"):
                        tx, bx = T.axis.remap("RS", [v_tx_2, v_bx])
                        T.reads(sum_local[tx, bx, 0])
                        T.writes(sum_shared[bx, 0])
                        with T.init():
                            sum_shared[bx, 0] = T.float32(0)
                        sum_shared[bx, 0] += sum_local[tx, bx, 0]
            for i in range(add_local_size):
                for v_tx_2 in T.thread_binding(TX, thread="threadIdx.x"):
                    with T.block("T_cast_2"):
                        bx = T.axis.spatial(batch_size, v_bx)
                        h = T.axis.spatial(hidden_size, i * TX + v_tx_2)
                        O[bx, 0, h] = T.cast(
                            T.rsqrt(sum_shared[bx, 0] * inv_hidden_size + eps)
                            * T.float32(add_local[h // TX])
                            * T.float32(C[h]),
                            dtype=in_dtype,
                        )

    return decode_add_rms


def _get_add_rms_norm_prefill(hidden_size: int, eps: float, TX: int, in_dtype: str):
    if in_dtype not in ("float16", "bfloat16"):
        raise ValueError(f"Unsupported data type: {in_dtype}")
    inv_hidden_size = T.float32(1.0 / float(hidden_size))
    eps = T.float32(eps)
    add_local_size = hidden_size // TX

    @T.prim_func(private=True)
    def prefill_add_rms(  # pylint: disable=too-many-locals
        pA: T.handle, pB: T.handle, pC: T.handle, pO: T.handle, pAdd: T.handle
    ):
        T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
        seq_len = T.int32()
        A = T.match_buffer(pA, (1, seq_len, hidden_size), in_dtype)
        B = T.match_buffer(pB, (1, seq_len, hidden_size), in_dtype)
        C = T.match_buffer(pC, (hidden_size,), in_dtype)
        O = T.match_buffer(pO, (1, seq_len, hidden_size), in_dtype)
        add = T.match_buffer(pAdd, (1, seq_len, hidden_size), in_dtype)
        add_local = T.alloc_buffer((hidden_size // TX,), in_dtype, scope="local")
        sum_shared = T.alloc_buffer((1, seq_len), scope="shared")
        sum_local = T.alloc_buffer((TX, 1, seq_len), scope="local")
        for v_bx in T.thread_binding(seq_len, thread="blockIdx.x"):
            for v_tx in T.thread_binding(
                TX,
                thread="threadIdx.x",
                annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1},
            ):
                for v_i in range(add_local_size):
                    with T.block("T_add"):
                        bx = T.axis.spatial(seq_len, v_bx)
                        h = T.axis.spatial(hidden_size, v_i * TX + v_tx)
                        add_local[h // TX] = A[0, bx, h] + B[0, bx, h]
                    with T.block("T_write_back"):
                        bx = T.axis.spatial(seq_len, v_bx)
                        h = T.axis.spatial(hidden_size, v_i * TX + v_tx)
                        add[0, bx, h] = add_local[h // TX]
                with T.block("T_multiply_red_rf_init"):
                    tx, bx = T.axis.remap("SS", [v_tx, v_bx])
                    sum_local[tx, 0, bx] = T.float32(0)
                for v_i, _j in T.grid(add_local_size, 1):
                    with T.block("T_multiply_red_rf_update"):
                        tx, bx, i = T.axis.remap("SSR", [v_tx, v_bx, v_i])
                        sum_local[tx, 0, bx] += T.float32(add_local[i]) * T.float32(add_local[i])
            for _j in range(1):
                for v_tx_2 in T.thread_binding(TX, thread="threadIdx.x"):
                    with T.block("T_multiply_red"):
                        tx, bx = T.axis.remap("RS", [v_tx_2, v_bx])
                        with T.init():
                            sum_shared[0, bx] = T.float32(0)
                        sum_shared[0, bx] = sum_shared[0, bx] + sum_local[tx, 0, bx]
            for v_i in range(add_local_size):
                for v_tx_2 in T.thread_binding(TX, thread="threadIdx.x"):
                    with T.block("T_cast_2"):
                        bx = T.axis.spatial(seq_len, v_bx)
                        v1 = T.axis.spatial(hidden_size, v_i * TX + v_tx_2)
                        O[0, bx, v1] = T.cast(
                            T.rsqrt(sum_shared[0, bx] * inv_hidden_size + eps)
                            * T.float32(add_local[v1 // TX])
                            * T.float32(C[v1]),
                            dtype=in_dtype,
                        )

    return prefill_add_rms


@tvm.transform.module_pass(opt_level=0, name="FuseAddRMSNorm")
class FuseAddRMSNorm:  # pylint: disable=too-few-public-methods
    """A compiler pass that fuses add + rms_norm."""

    def __init__(self, target: tvm.target.Target) -> None:
        """Initializer.

        Parameters
        ----------
        target : tvm.target.Target
            Target device.
        """
        self.target = target

    def transform_module(self, mod: tvm.IRModule, _ctx: tvm.transform.PassContext) -> tvm.IRModule:
        """IRModule-level transformation."""
        return _FuseAddRMSNormRewriter(mod.clone(), self.target).transform()


@mutator
class _FuseAddRMSNormRewriter(PyExprMutator):  # pylint: disable=abstract-method
    def __init__(self, mod: tvm.IRModule, target: tvm.target.Target):
        super().__init__(mod)
        self.mod = mod
        self.prefill_norm_gv: Optional[tvm.ir.GlobalVar] = None
        self.decode_norm_gv: Optional[tvm.ir.GlobalVar] = None
        self.TX = min(1024, get_max_num_threads_per_block(target))

    def transform(self) -> tvm.IRModule:  # pylint: disable=too-many-locals
        """Entry point of the transformation"""
        for g_var, func in self.mod.functions_items():
            if not isinstance(func, relax.Function):
                continue
            new_func = self.visit_expr(func)
            new_func = remove_all_unused(new_func)
            self.builder_.update_func(g_var, new_func)
        return self.builder_.finalize()

    def visit_call_(self, call: relax.Call) -> relax.Expr:  # pylint: disable=arguments-renamed
        call = super().visit_call_(call)

        # Match the "rms_norm(add(x1, x2), w)" pattern
        if call.op != tvm.ir.Op.get("relax.nn.rms_norm") or call.struct_info.dtype not in [
            "bfloat16",
            "float16",
        ]:
            return call
        assert len(call.args) == 2
        weight = call.args[1]
        eps = call.attrs.epsilon
        assert isinstance(call.args[0], relax.Var)
        y = self.lookup_binding(call.args[0])
        if not isinstance(y, relax.Call) or y.op != tvm.ir.Op.get("relax.add"):
            return call
        assert len(y.args) == 2
        x1 = y.args[0]
        x2 = y.args[1]
        # Extra check
        n, _, h = x1.struct_info.shape
        h = int(h)
        if h % self.TX != 0:
            return call

        is_prefill = n == 1
        func_gv = self.prefill_norm_gv if is_prefill else self.decode_norm_gv
        if func_gv is None:
            if is_prefill:
                func_gv = self.builder_.add_func(
                    _get_add_rms_norm_prefill(h, eps, self.TX, call.struct_info.dtype),
                    "fuse_add_norm_prefill",
                )
                self.prefill_norm_gv = func_gv
            else:
                func_gv = self.builder_.add_func(
                    _get_add_rms_norm_decode(h, eps, self.TX, call.struct_info.dtype),
                    "fuse_add_norm_decode",
                )
                self.decode_norm_gv = func_gv

        tuple_output = self.builder_.emit(
            relax.call_tir(
                func_gv,
                [x1, x2, weight],
                out_sinfo=[x1.struct_info, x2.struct_info],
            )
        )
        new_o = relax.TupleGetItem(tuple_output, 0)
        new_y = self.builder_.emit(relax.TupleGetItem(tuple_output, 1))
        self.set_var_remap(call.args[0].vid, new_y)
        return new_o
