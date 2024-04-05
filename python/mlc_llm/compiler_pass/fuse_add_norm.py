"""A compiler pass that fuses add + rms_norm."""

import tvm
from tvm import relax
from tvm.relax.dpl import PatternContext, rewrite_bindings
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.script import tir as T

from ..support.max_thread_check import get_max_num_threads_per_block

# mypy: disable-error-code="attr-defined,valid-type"
# pylint: disable=too-many-locals,invalid-name


def _get_add_rms_norm_decode(hidden_size: int, eps: float, TX: int):
    inv_hidden_size = T.float32(1.0 / float(hidden_size))
    eps = T.float32(eps)
    add_local_size = hidden_size // TX

    @T.prim_func(private=True)
    def decode_add_rms(pA: T.handle, pB: T.handle, pC: T.handle, pO: T.handle, pAdd: T.handle):
        T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
        batch_size = T.int32()
        A = T.match_buffer(pA, (batch_size, 1, hidden_size), "float16")
        B = T.match_buffer(pB, (batch_size, 1, hidden_size), "float16")
        C = T.match_buffer(pC, (hidden_size,), "float16")
        O = T.match_buffer(pO, (batch_size, 1, hidden_size), "float16")
        add = T.match_buffer(pAdd, (batch_size, 1, hidden_size), "float16")
        add_local = T.alloc_buffer((hidden_size // TX,), "float16", scope="local")
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
                        O[bx, 0, h] = T.float16(
                            T.rsqrt(sum_shared[bx, 0] * inv_hidden_size + eps)
                            * T.float32(add_local[h // TX])
                            * T.float32(C[h])
                        )

    return decode_add_rms


def _get_add_rms_norm_prefill(hidden_size: int, eps: float, TX: int):
    inv_hidden_size = T.float32(1.0 / float(hidden_size))
    eps = T.float32(eps)
    add_local_size = hidden_size // TX

    @T.prim_func(private=True)
    def prefill_add_rms(pA: T.handle, pB: T.handle, pC: T.handle, pO: T.handle, pAdd: T.handle):
        T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
        seq_len = T.int32()
        A = T.match_buffer(pA, (1, seq_len, hidden_size), "float16")
        B = T.match_buffer(pB, (1, seq_len, hidden_size), "float16")
        C = T.match_buffer(pC, (hidden_size,), "float16")
        O = T.match_buffer(pO, (1, seq_len, hidden_size), "float16")
        add = T.match_buffer(pAdd, (1, seq_len, hidden_size), "float16")
        add_local = T.alloc_buffer((hidden_size // TX,), "float16", scope="local")
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
                        O[0, bx, v1] = T.float16(
                            T.rsqrt(sum_shared[0, bx] * inv_hidden_size + eps)
                            * T.float32(add_local[v1 // TX])
                            * T.float32(C[v1])
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
        self.TX = 1024  # default

        max_num_threads_per_block = get_max_num_threads_per_block(target)
        if max_num_threads_per_block < self.TX:
            self.TX = max_num_threads_per_block

    def transform_module(self, mod: tvm.IRModule, _ctx: tvm.transform.PassContext) -> tvm.IRModule:
        """IRModule-level transformation."""
        with PatternContext() as ctx:
            pat_x1 = wildcard()
            pat_x2 = wildcard()
            pat_y = is_op("relax.add")(pat_x1, pat_x2)
            pat_w = wildcard()
            pat_o = is_op("relax.nn.rms_norm")(pat_y, pat_w)

        def rewriter(matchings, bindings):
            x1 = matchings[pat_x1]
            x2 = matchings[pat_x2]
            weight = matchings[pat_w]
            y = matchings[pat_y]
            o = matchings[pat_o]
            eps = bindings[o].attrs.epsilon
            if x1.struct_info.dtype != "float16":
                return {}
            n, _, h = x1.struct_info.shape
            func_name = "fuse_add_norm_prefill" if n == 1 else "fuse_add_norm_decode"

            if all(gv.name_hint != func_name for gv in mod.functions):
                h = int(h)
                if h % self.TX != 0:
                    return {}
                if n == 1:
                    func = _get_add_rms_norm_prefill(h, eps, self.TX)
                else:
                    func = _get_add_rms_norm_decode(h, eps, self.TX)
                mod[func_name] = func
                gvar = mod.get_global_var(func_name)
                relax.expr._update_struct_info(  # pylint: disable=protected-access
                    gvar,
                    relax.FuncStructInfo.opaque_func(ret=relax.ObjectStructInfo()),
                )
            else:
                gvar = mod.get_global_var(func_name)
            o_y_tuple = relax.call_tir(
                gvar,
                [x1, x2, weight],
                out_sinfo=[x1.struct_info, x1.struct_info],
            )
            return {
                o: relax.TupleGetItem(o_y_tuple, 0),
                y: relax.TupleGetItem(o_y_tuple, 1),
            }

        new_mod = {}
        for gvar, func in mod.functions.items():
            if isinstance(func, relax.Function):
                func = rewrite_bindings(ctx, rewriter, func)
            new_mod[gvar] = func

        for gvar, func in mod.functions.items():
            if isinstance(func, tvm.tir.PrimFunc) and gvar not in new_mod:
                new_mod[gvar] = func

        new_mod = tvm.IRModule(new_mod, mod.type_definitions, mod.attrs, mod.global_infos)
        return new_mod
