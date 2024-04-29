"""The pass that attaches logit processor functions to the IRModule."""

import tvm
from tvm import IRModule
from tvm.script import tir as T


@tvm.transform.module_pass(opt_level=0, name="AttachSpecDecodeAuxFuncs")
class AttachSpecDecodeAuxFuncs:  # pylint: disable=too-few-public-methods
    """Attach logit processing TIR functions to IRModule."""

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """Entrypoint"""
        mod = mod.clone()
        mod["scatter_probs"] = _get_scatter_2d_inplace(
            dtype="float32", global_symbol="scatter_probs"
        )
        mod["gather_probs"] = _get_gather_2d_inplace(dtype="float32", global_symbol="gather_probs")
        if "prefill_to_last_hidden_states" in mod:
            hidden_states_struct_info = mod["prefill_to_last_hidden_states"].ret_struct_info.fields[
                0
            ]  # pylint: disable=no-member
            dtype = hidden_states_struct_info.dtype
            mod["scatter_hidden_states"] = _get_scatter_2d_inplace(
                dtype, global_symbol="scatter_hidden_states"
            )
            mod["gather_hidden_states"] = _get_gather_2d_inplace(
                dtype, global_symbol="gather_hidden_states"
            )
        return mod


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
