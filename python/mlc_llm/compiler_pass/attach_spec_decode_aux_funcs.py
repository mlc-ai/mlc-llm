"""The pass that attaches logit processor functions to the IRModule."""

import tvm
from tvm import IRModule, relax, tir
from tvm.relax import BlockBuilder, TensorStructInfo
from tvm.script import tir as T


@tvm.transform.module_pass(opt_level=0, name="AttachSpecDecodeAuxFuncs")
class AttachSpecDecodeAuxFuncs:  # pylint: disable=too-few-public-methods
    """Attach logit processing TIR functions to IRModule."""

    tensor_parallel_shards: int

    def __init__(self, tensor_parallel_shards: int):
        self.tensor_parallel_shards = tensor_parallel_shards

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """Entrypoint"""
        mod = mod.clone()
        bb = BlockBuilder(mod)
        bb.add_func(
            _get_scatter_2d_inplace(dtype="float32", global_symbol="scatter_probs"), "scatter_probs"
        )
        bb.add_func(
            _get_gather_2d_inplace(dtype="float32", global_symbol="gather_probs"), "gather_probs"
        )
        if "prefill_to_last_hidden_states" in mod:
            hidden_states_struct_info = mod["prefill_to_last_hidden_states"].ret_struct_info.fields[
                0
            ]  # pylint: disable=no-member
            dtype = hidden_states_struct_info.dtype
            _add_gather_hidden_states(bb, self.tensor_parallel_shards, dtype)
            _add_scatter_hidden_states(bb, self.tensor_parallel_shards, dtype)
        return bb.finalize()


def _get_scatter_2d_inplace(dtype: str, global_symbol: str):
    @T.prim_func
    def _scatter_2d(var_src: T.handle, var_indices: T.handle, var_dst: T.handle):
        T.func_attr({"global_symbol": global_symbol, "tir.noalias": True})
        batch_size = T.int32(is_size_var=True)
        m = T.int32(is_size_var=True)
        n = T.int32(is_size_var=True)
        src = T.match_buffer(var_src, (batch_size, n), dtype)
        indices = T.match_buffer(var_indices, (batch_size,), "int32")
        dst = T.match_buffer(var_dst, (m, n), dtype)
        for b, j in T.grid(batch_size, n):
            with T.block("scatter_2d"):
                vb, vj = T.axis.remap("SS", [b, j])
                dst[indices[vb], vj] = src[vb, vj]

    return _scatter_2d


def _get_gather_2d_inplace(dtype: str, global_symbol: str):
    @T.prim_func
    def _gather_2d(var_src: T.handle, var_indices: T.handle, var_dst: T.handle):
        T.func_attr({"global_symbol": global_symbol, "tir.noalias": True})
        batch_size = T.int32(is_size_var=True)
        m = T.int32(is_size_var=True)
        n = T.int32(is_size_var=True)
        src = T.match_buffer(var_src, (m, n), dtype)
        indices = T.match_buffer(var_indices, (batch_size,), "int32")
        dst = T.match_buffer(var_dst, (batch_size, n), dtype)
        for b, j in T.grid(batch_size, n):
            with T.block("gather_2d"):
                vb, vj = T.axis.remap("SS", [b, j])
                dst[vb, vj] = src[indices[vb], vj]

    return _gather_2d


def _add_scatter_hidden_states(bb: BlockBuilder, tensor_parallel_shards: int, dtype: str):
    batch_size = tir.SizeVar("batch_size", "int64")
    m = tir.SizeVar("m", "int64")
    n = tir.SizeVar("n", "int64")
    src = relax.Var("src", struct_info=TensorStructInfo([batch_size, n], dtype))
    indices = relax.Var("indices", struct_info=TensorStructInfo([batch_size], "int32"))
    dst = relax.Var("dst", struct_info=TensorStructInfo([m, n], dtype))
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
                    dst.struct_info,  # pylint: disable=no-member
                )
            )
        gv = bb.emit_func_output(output)
    return gv


def _add_gather_hidden_states(bb: BlockBuilder, tensor_parallel_shards: int, dtype: str):
    batch_size = tir.SizeVar("batch_size", "int64")
    m = tir.SizeVar("m", "int64")
    n = tir.SizeVar("n", "int64")
    src = relax.Var("src", struct_info=TensorStructInfo([m, n], dtype))
    indices = relax.Var("indices", struct_info=TensorStructInfo([batch_size], "int32"))
    dst = relax.Var("dst", struct_info=TensorStructInfo([batch_size, n], dtype))
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
                    dst.struct_info,  # pylint: disable=no-member
                )
            )
        gv = bb.emit_func_output(output)
    return gv
