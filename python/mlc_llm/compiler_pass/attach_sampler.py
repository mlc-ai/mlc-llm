"""The pass that attaches GPU sampler functions to the IRModule."""

from typing import Dict

import tvm
from tvm import IRModule, relax, te, tir
from tvm.relax.frontend import nn
from tvm.script import tir as T
from ..op.batch_spec_verify import batch_spec_verify


@tvm.transform.module_pass(opt_level=0, name="AttachGPUSamplingFunc")
class AttachGPUSamplingFunc:  # pylint: disable=too-few-public-methods
    """Attach GPU sampling functions to IRModule."""

    def __init__(self, target: tvm.target.Target, variable_bounds: Dict[str, int]):
        # Specifically for RWKV workloads, which contains -1 max_seq_len
        max_batch_size = variable_bounds["batch_size"]
        self.variable_bounds = {
            "batch_size": max_batch_size,
            "num_samples": max_batch_size,
            "num_positions": 6 * max_batch_size,
        }
        self.non_negative_var = ["vocab_size"]
        self.target = target

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """Entrypoint"""
        if str(self.target.kind) != "cuda":
            # Only enable GPU sampling for CUDA.
            return mod

        bb = relax.BlockBuilder(mod)
        # Prefill method exists in base models.
        # Prefill_to_last_hidden method exists in base model and speculative small models
        if "prefill" in mod:
            vocab_size = mod["prefill"].ret_struct_info.fields[0].shape[-1]
        else:
            assert (
                "prefill_to_last_hidden_states" in mod
            ), "Everay model should either has 'prefill' or 'prefill_to_last_hidden_states' method"
            vocab_size = mod["prefill_to_last_hidden_states"].ret_struct_info.fields[0].shape[-1]
        gv_names = [
            gv.name_hint
            for gv in [
                _attach_multinomial_sampling_func(bb, vocab_size),
                _attach_argsort_func(bb, vocab_size),
                _attach_sample_with_top_p(bb, vocab_size),
                _attach_take_probs_func(bb, vocab_size),
                _attach_batch_verifier(bb, vocab_size),
            ]
        ]

        mod = bb.finalize()
        for gv_name in gv_names:
            mod[gv_name] = (
                mod[gv_name]
                .with_attr("tir_var_upper_bound", self.variable_bounds)
                .with_attr("tir_non_negative_var", self.non_negative_var)
            )
        return mod


def _attach_multinomial_sampling_func(bb: relax.BlockBuilder, vocab_size: tir.PrimExpr):
    batch_size = tir.Var("batch_size", "int64")
    num_samples = tir.Var("num_samples", "int64")
    probs = relax.Var("probs", relax.TensorStructInfo((batch_size, vocab_size), "float32"))
    uniform_samples = relax.Var(
        "uniform_samples", relax.TensorStructInfo((num_samples,), "float32")
    )
    sample_indices = relax.Var("sample_indices", relax.TensorStructInfo((num_samples,), "int32"))
    with bb.function("multinomial_from_uniform", [probs, uniform_samples, sample_indices]):
        with bb.dataflow():
            sample_shape = relax.ShapeExpr([num_samples, 1])
            probs_tensor = nn.wrap_nested(probs, name="probs")
            uniform_samples_tensor = nn.wrap_nested(
                relax.call_pure_packed(
                    "vm.builtin.reshape",
                    uniform_samples,
                    sample_shape,
                    sinfo_args=relax.TensorStructInfo(sample_shape, "float32"),
                ),
                name="uniform_samples",
            )
            sample_indices_tensor = nn.wrap_nested(
                relax.call_pure_packed(
                    "vm.builtin.reshape",
                    sample_indices,
                    sample_shape,
                    sinfo_args=relax.TensorStructInfo(sample_shape, "int32"),
                ),
                name="sample_indices",
            )
            result_tensor = nn.multinomial_from_uniform(  # pylint:disable=too-many-function-args
                probs_tensor, uniform_samples_tensor, sample_indices_tensor, "int32"
            )
            result = bb.emit(
                relax.call_pure_packed(
                    "vm.builtin.reshape",
                    result_tensor._expr,  # pylint: disable=protected-access
                    sample_indices.struct_info.shape,  # pylint: disable=no-member
                    sinfo_args=sample_indices.struct_info,  # pylint: disable=no-member
                )
            )
        gv = bb.emit_func_output(result)
    return gv


def _attach_argsort_func(bb: relax.BlockBuilder, vocab_size: tir.PrimExpr):
    batch_size = tir.Var("batch_size", "int64")
    probs = relax.Var("probs", relax.TensorStructInfo((batch_size, vocab_size), "float32"))
    with bb.function("argsort_probs", [probs]):
        with bb.dataflow():
            sorted_indices = bb.emit(relax.op.argsort(probs, descending=True, dtype="int32"))
            sorted_values = bb.emit_te(
                lambda unsorted_probs, sorted_indices: te.compute(
                    (batch_size, vocab_size),
                    lambda i, j: unsorted_probs[i, sorted_indices[i, j]],
                    name="take_sorted_probs",
                ),
                probs,
                sorted_indices,
                primfunc_name_hint="take_sorted_probs",
            )
            output = (sorted_values, sorted_indices)
            bb.emit_output(output)
        gv = bb.emit_func_output(output)
    return gv


def _attach_sample_with_top_p(  # pylint: disable=too-many-locals
    bb: relax.BlockBuilder, vocab_size: tir.PrimExpr
):
    batch_size = tir.Var("batch_size", "int64")
    num_samples = tir.Var("num_samples", "int64")
    sorted_probs = relax.Var(
        "sorted_probs", relax.TensorStructInfo((batch_size, vocab_size), "float32")
    )
    sorted_indices = relax.Var(
        "sorted_indices", relax.TensorStructInfo((batch_size, vocab_size), "int32")
    )
    uniform_samples = relax.Var(
        "uniform_samples", relax.TensorStructInfo((num_samples,), "float32")
    )
    sample_indices = relax.Var("sample_indices", relax.TensorStructInfo((num_samples,), "int32"))
    top_p = relax.Var("top_p", relax.TensorStructInfo((batch_size,), "float32"))

    @T.prim_func
    def full(var_result: T.handle, value: T.int32):
        batch_size = T.int32(is_size_var=True)
        result = T.match_buffer(var_result, (batch_size, 1), "int32")
        for i in T.serial(batch_size):
            with T.block("block"):
                vi = T.axis.spatial(batch_size, i)
                result[vi, 0] = value

    with bb.function(
        "sample_with_top_p",
        [sorted_probs, sorted_indices, uniform_samples, sample_indices, top_p],
    ):
        with bb.dataflow():
            sample_shape = relax.ShapeExpr([num_samples, 1])
            top_p_shape = relax.ShapeExpr([batch_size, 1])
            sorted_probs_tensor = nn.wrap_nested(sorted_probs, name="sorted_probs")
            sorted_indices_tensor = nn.wrap_nested(sorted_indices, name="sorted_indices")
            uniform_samples_tensor = nn.wrap_nested(
                relax.call_pure_packed(
                    "vm.builtin.reshape",
                    uniform_samples,
                    sample_shape,
                    sinfo_args=relax.TensorStructInfo(sample_shape, "float32"),
                ),
                name="uniform_samples",
            )
            sample_indices_tensor = nn.wrap_nested(
                relax.call_pure_packed(
                    "vm.builtin.reshape",
                    sample_indices,
                    sample_shape,
                    sinfo_args=relax.TensorStructInfo(sample_shape, "int32"),
                ),
                name="sample_indices",
            )
            top_p_tensor = nn.wrap_nested(
                relax.call_pure_packed(
                    "vm.builtin.reshape",
                    top_p,
                    top_p_shape,
                    sinfo_args=relax.TensorStructInfo(top_p_shape, "float32"),
                ),
                name="sample_indices",
            )
            top_k_tensor = nn.tensor_ir_op(
                full,
                name_hint="full",
                args=[vocab_size],
                out=nn.Tensor.placeholder(
                    [batch_size, 1],
                    "int32",
                ),
            )

            result_tensor = (
                nn.sample_top_p_top_k_from_sorted_prob(  # pylint:disable=too-many-function-args
                    sorted_probs_tensor,
                    sorted_indices_tensor,
                    top_p_tensor,
                    top_k_tensor,
                    uniform_samples_tensor,
                    sample_indices_tensor,
                )
            )
            result = bb.emit(
                relax.call_pure_packed(
                    "vm.builtin.reshape",
                    result_tensor._expr,  # pylint: disable=protected-access
                    sample_indices.struct_info.shape,  # pylint: disable=no-member
                    sinfo_args=sample_indices.struct_info,  # pylint: disable=no-member
                )
            )
            bb.emit_output(result)
        gv = bb.emit_func_output(result)
    return gv


def _attach_take_probs_func(bb: relax.BlockBuilder, vocab_size: tir.PrimExpr):
    batch_size = tir.Var("batch_size", "int64")
    num_samples = tir.Var("num_samples", "int64")
    num_positions = tir.Var("num_positions", "int64")
    unsorted_probs = relax.Var(
        "unsorted_probs", relax.TensorStructInfo((batch_size, vocab_size), "float32")
    )
    sorted_indices = relax.Var(
        "sorted_indices", relax.TensorStructInfo((batch_size, vocab_size), "int32")
    )
    sample_indices = relax.Var("sample_indices", relax.TensorStructInfo((num_samples,), "int32"))
    sampling_results = relax.Var("sampling_result", relax.TensorStructInfo((num_samples,), "int32"))
    top_prob_offsets = relax.Var(
        "lobprob_offsets", relax.TensorStructInfo((num_positions,), "int32")
    )

    @T.prim_func
    def sampler_take_probs_tir(  # pylint: disable=too-many-locals,too-many-arguments
        var_unsorted_probs: T.handle,
        var_sorted_indices: T.handle,
        var_sample_indices: T.handle,
        var_sampling_results: T.handle,
        var_top_prob_offsets: T.handle,
        var_sampled_values: T.handle,
        var_top_prob_probs: T.handle,
        var_top_prob_indices: T.handle,
    ):
        batch_size = T.int32(is_size_var=True)
        num_samples = T.int32(is_size_var=True)
        num_positions = T.int32(is_size_var=True)
        vocab_size = T.int32(is_size_var=True)
        unsorted_probs = T.match_buffer(var_unsorted_probs, (batch_size, vocab_size), "float32")
        sorted_indices = T.match_buffer(var_sorted_indices, (batch_size, vocab_size), "int32")
        sample_indices = T.match_buffer(var_sample_indices, (num_samples,), "int32")
        sampling_results = T.match_buffer(var_sampling_results, (num_samples,), "int32")
        top_prob_offsets = T.match_buffer(var_top_prob_offsets, (num_positions,), "int32")
        sampled_values = T.match_buffer(var_sampled_values, (num_samples,), "float32")
        top_prob_probs = T.match_buffer(var_top_prob_probs, (num_positions,), "float32")
        top_prob_indices = T.match_buffer(var_top_prob_indices, (num_positions,), "int32")
        for i in T.serial(num_positions + num_samples):
            with T.block("block"):
                vi = T.axis.spatial(num_positions + num_samples, i)
                if vi < num_positions:
                    row = T.floordiv(top_prob_offsets[vi], vocab_size)
                    col = T.floormod(top_prob_offsets[vi], vocab_size)
                    top_prob_indices[vi] = sorted_indices[row, col]
                    top_prob_probs[vi] = unsorted_probs[row, sorted_indices[row, col]]
                else:
                    vj: T.int32 = vi - num_positions
                    sampled_values[vj] = unsorted_probs[sample_indices[vj], sampling_results[vj]]

    args = [unsorted_probs, sorted_indices, sample_indices, sampling_results, top_prob_offsets]
    with bb.function("sampler_take_probs", args):
        with bb.dataflow():
            taken_probs_indices = bb.emit(
                relax.call_tir(
                    bb.add_func(sampler_take_probs_tir, "sampler_take_probs_tir"),
                    args,
                    out_sinfo=[
                        relax.TensorStructInfo((num_samples,), "float32"),
                        relax.TensorStructInfo((num_positions,), "float32"),
                        relax.TensorStructInfo((num_positions,), "int32"),
                    ],
                )
            )
            bb.emit_output(taken_probs_indices)
        gv = bb.emit_func_output(taken_probs_indices)
    return gv


def _attach_batch_verifier(bb: relax.BlockBuilder, vocab_size: tir.PrimExpr):
    num_nodes = tir.Var("num_nodes", "int64")
    nbatch = tir.Var("nbatch", "int64")
    draft_probs = relax.Var(
        "draft_probs", relax.TensorStructInfo((num_nodes, vocab_size), "float32")
    )
    draft_tokens = relax.Var("draft_tokens", relax.TensorStructInfo((num_nodes,), "int32"))
    model_probs = relax.Var(
        "model_probs", relax.TensorStructInfo((num_nodes, vocab_size), "float32")
    )
    token_tree_first_child = relax.Var(
        "token_tree_first_child", relax.TensorStructInfo((num_nodes,), "int32")
    )
    token_tree_next_sibling = relax.Var(
        "token_tree_next_sibling", relax.TensorStructInfo((num_nodes,), "int32")
    )
    uniform_samples = relax.Var("uniform_samples", relax.TensorStructInfo((num_nodes,), "float32"))
    token_tree_parent_ptr = relax.Var(
        "token_tree_parent_ptr", relax.TensorStructInfo((nbatch,), "int32")
    )
    args = [
        draft_probs,
        draft_tokens,
        model_probs,
        token_tree_first_child,
        token_tree_next_sibling,
        uniform_samples,
        token_tree_parent_ptr,
    ]
    with bb.function("sampler_verify_draft_tokens", args):
        with bb.dataflow():
            res = bb.emit(
                relax.call_tir_inplace(
                    bb.add_func(batch_spec_verify(vocab_size), "batch_verify_on_gpu_single_kernel"),
                    args,
                    inplace_indices=[args.index(model_probs), args.index(token_tree_parent_ptr)],
                    out_sinfo=[model_probs.struct_info, token_tree_parent_ptr.struct_info],
                )
            )
            bb.emit_output(res)
        gv = bb.emit_func_output(res)
    return gv
