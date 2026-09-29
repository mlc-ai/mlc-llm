"""The pass that attaches logit processor functions to the IRModule."""

import tvm
from tvm import IRModule, relax, tirx
from tvm.relax import BlockBuilder, TensorType
from tvm.script import s_tir as Ts
from tvm.script import tirx as T


@tvm.transform.module_pass(opt_level=0, name="AttachSpecDecodeAuxFuncs")
class AttachSpecDecodeAuxFuncs:
    """Attach logit processing TIR functions to IRModule."""

    tensor_parallel_shards: int

    def __init__(self, tensor_parallel_shards: int):
        self.tensor_parallel_shards = tensor_parallel_shards

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """Entrypoint"""
        mod = mod.clone()
        bb = BlockBuilder(mod)
        bb.add_func(
            _get_scatter_2d_inplace(dtype="float32", global_symbol="scatter_probs"),
            "scatter_probs",
        )
        bb.add_func(
            _get_gather_2d_inplace(dtype="float32", global_symbol="gather_probs"),
            "gather_probs",
        )
        if "prefill_to_last_hidden_states" in mod:
            hidden_states_struct_info = mod["prefill_to_last_hidden_states"].ret_ty.fields[0]
            dtype = hidden_states_struct_info.dtype
            _add_gather_hidden_states(bb, self.tensor_parallel_shards, dtype)
            _add_scatter_hidden_states(bb, self.tensor_parallel_shards, dtype)
        return bb.finalize()


def _get_scatter_2d_inplace(dtype: str, global_symbol: str):
    batch_size = T.dynamic("batch_size", "int32")
    m = T.dynamic("m", "int32")
    n = T.dynamic("n", "int32")

    @Ts.prim_func
    def _scatter_2d(
        src: T.Buffer((batch_size, n), dtype),
        indices: T.Buffer((batch_size,), "int32"),
        dst: T.Buffer((m, n), dtype),
    ):
        T.func_attr({"global_symbol": global_symbol, "tirx.noalias": True})
        for b, j in T.grid(batch_size, n):
            with Ts.sblock("scatter_2d"):
                vb, vj = Ts.axis.remap("SS", [b, j])
                dst[indices[vb], vj] = src[vb, vj]

    return _scatter_2d


def _get_gather_2d_inplace(dtype: str, global_symbol: str):
    batch_size = T.dynamic("batch_size", "int32")
    m = T.dynamic("m", "int32")
    n = T.dynamic("n", "int32")

    @Ts.prim_func
    def _gather_2d(
        src: T.Buffer((m, n), dtype),
        indices: T.Buffer((batch_size,), "int32"),
        dst: T.Buffer((batch_size, n), dtype),
    ):
        T.func_attr({"global_symbol": global_symbol, "tirx.noalias": True})
        for b, j in T.grid(batch_size, n):
            with Ts.sblock("gather_2d"):
                vb, vj = Ts.axis.remap("SS", [b, j])
                dst[vb, vj] = src[indices[vb], vj]

    return _gather_2d


def _add_scatter_hidden_states(bb: BlockBuilder, tensor_parallel_shards: int, dtype: str):
    batch_size = tirx.Var("batch_size", "int64")
    m = tirx.Var("m", "int64")
    n = tirx.Var("n", "int64")
    src = relax.Var("src", ty=TensorType([batch_size, n], dtype))
    indices = relax.Var("indices", ty=TensorType([batch_size], "int32"))
    dst = relax.Var("dst", ty=TensorType([m, n], dtype))
    with bb.function("scatter_hidden_states", [src, indices, dst]):
        with bb.dataflow():
            if tensor_parallel_shards > 1:
                indices = relax.op.ccl.broadcast_from_worker0(indices)
            output = bb.emit_output(
                relax.op.call_tir_inplace(
                    bb.add_func(
                        _get_scatter_2d_inplace(dtype, "_scatter_hidden_states"),
                        "_scatter_hidden_states",
                    ),
                    [src, indices, dst],
                    2,
                    dst.ty,
                )
            )
        gv = bb.emit_func_output(output)
    return gv


def _add_gather_hidden_states(bb: BlockBuilder, tensor_parallel_shards: int, dtype: str):
    batch_size = tirx.Var("batch_size", "int64")
    m = tirx.Var("m", "int64")
    n = tirx.Var("n", "int64")
    src = relax.Var("src", ty=TensorType([m, n], dtype))
    indices = relax.Var("indices", ty=TensorType([batch_size], "int32"))
    dst = relax.Var("dst", ty=TensorType([batch_size, n], dtype))
    with bb.function("gather_hidden_states", [src, indices, dst]):
        with bb.dataflow():
            if tensor_parallel_shards > 1:
                indices = relax.op.ccl.broadcast_from_worker0(indices)
            output = bb.emit_output(
                relax.op.call_tir_inplace(
                    bb.add_func(
                        _get_gather_2d_inplace(dtype, "_gather_hidden_states"),
                        "_gather_hidden_states",
                    ),
                    [src, indices, dst],
                    2,
                    dst.ty,
                )
            )
        gv = bb.emit_func_output(output)
    return gv
