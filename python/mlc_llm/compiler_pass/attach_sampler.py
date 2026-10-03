"""The pass that attaches GPU sampler functions to the IRModule."""

from typing import Any, Dict, Optional  # noqa: UP035

import tvm
from tvm import IRModule, relax, te, tirx
from tvm.relax.frontend import nn
from tvm.script import s_tir as Ts
from tvm.script import tirx as T

from mlc_llm.op.batch_spec_verify import batch_spec_verify
from mlc_llm.op.top_p_pivot import top_p_pivot, top_p_renorm
from mlc_llm.support.max_thread_check import get_max_num_threads_per_block


@tvm.transform.module_pass(opt_level=0, name="AttachGPUSamplingFunc")
class AttachGPUSamplingFunc:
    """Attach GPU sampling functions to IRModule."""

    def __init__(
        self,
        target: tvm.target.Target,
        variable_bounds: Dict[str, int],  # noqa: UP006
        metadata: Optional[Dict[str, Any]] = None,  # noqa: UP006
    ):
        # Specifically for RWKV workloads, which contains -1 max_seq_len
        max_batch_size = variable_bounds["batch_size"]
        self.variable_bounds = {
            "batch_size": max_batch_size,
            "num_samples": max_batch_size,
            "num_positions": 6 * max_batch_size,
        }
        self.non_negative_var = ["vocab_size"]
        self.target = target
        self.active_vocab_size = metadata.get("active_vocab_size") if metadata else None

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """Entrypoint"""
        target_kind = self.target.kind.name
        if target_kind not in ["cuda", "vulkan", "metal", "webgpu"]:
            # Only enable GPU sampling for CUDA, Vulkan, Metal, and WebGPU.
            return mod

        bb = relax.BlockBuilder(mod)
        if target_kind == "webgpu":
            # Only attach functions that do not contain i8s for WebGPU
            gv_names = [
                gv.name_hint
                for gv in [
                    _attach_greedy_sampling_func(bb, self.target, self.active_vocab_size),
                    _attach_argsort_func(bb),
                    _attach_sample_with_top_p(bb),
                ]
            ]
        else:
            gv_names = [
                gv.name_hint
                for gv in [
                    _attach_multinomial_sampling_func(bb),
                    _attach_argsort_func(bb),
                    _attach_sample_with_top_p(bb),
                    _attach_take_probs_func(bb),
                    _attach_batch_verifier(bb),
                    _attach_renormalize_by_top_p(bb, self.target),
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


def _attach_multinomial_sampling_func(bb: relax.BlockBuilder):
    batch_size = tirx.Var("batch_size", "int64")
    num_samples = tirx.Var("num_samples", "int64")
    vocab_size = tirx.Var("vocab_size", "int64")
    probs = relax.Var("probs", relax.TensorType((batch_size, vocab_size), "float32"))
    uniform_samples = relax.Var("uniform_samples", relax.TensorType((num_samples,), "float32"))
    sample_indices = relax.Var("sample_indices", relax.TensorType((num_samples,), "int32"))
    with bb.function("multinomial_from_uniform", [probs, uniform_samples, sample_indices]):
        with bb.dataflow():
            sample_shape = relax.ShapeExpr([num_samples, 1])
            probs_tensor = nn.wrap_nested(probs, name="probs")
            uniform_samples_tensor = nn.wrap_nested(
                relax.call_pure_packed(
                    "vm.builtin.reshape",
                    uniform_samples,
                    sample_shape,
                    ty_args=relax.TensorType(sample_shape, "float32"),
                ),
                name="uniform_samples",
            )
            sample_indices_tensor = nn.wrap_nested(
                relax.call_pure_packed(
                    "vm.builtin.reshape",
                    sample_indices,
                    sample_shape,
                    ty_args=relax.TensorType(sample_shape, "int32"),
                ),
                name="sample_indices",
            )
            result_tensor = nn.multinomial_from_uniform(
                probs_tensor,
                uniform_samples_tensor,
                sample_indices_tensor,
                "int32",
                name="nn_multinomial_from_uniform",
            )
            result = bb.emit(
                relax.call_pure_packed(
                    "vm.builtin.reshape",
                    result_tensor._expr,
                    sample_indices.ty.shape,
                    ty_args=sample_indices.ty,
                )
            )
            output = bb.emit_output(result)
        gv = bb.emit_func_output(output)
    return gv


def _attach_argsort_func(bb: relax.BlockBuilder):
    batch_size = tirx.Var("batch_size", "int64")
    vocab_size = tirx.Var("vocab_size", "int64")
    probs = relax.Var("probs", relax.TensorType((batch_size, vocab_size), "float32"))
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
            output = bb.emit_output((sorted_values, sorted_indices))
        gv = bb.emit_func_output(output)
    return gv


def _attach_greedy_sampling_func(
    bb: relax.BlockBuilder,
    target: tvm.target.Target,
    active_vocab_size: Optional[int] = None,
):
    batch_size = tirx.Var("batch_size", "int64")
    vocab_size = tirx.Var("vocab_size", "int64")
    logits = relax.Var("logits", relax.TensorType((batch_size, 1, vocab_size), "float32"))
    with bb.function("sample_with_temperature_zero", [logits]):
        with bb.dataflow():
            sampled_tokens = bb.emit(
                relax.call_tir(
                    bb.add_func(
                        _get_greedy_argmax_func(target, active_vocab_size),
                        "greedy_argmax",
                    ),
                    args=[logits],
                    out_ty=relax.TensorType((batch_size,), "int32"),
                )
            )
            output = bb.emit_output(sampled_tokens)
        gv = bb.emit_func_output(output)
    return gv


def _get_greedy_argmax_func(target: tvm.target.Target, active_vocab_size: Optional[int] = None):
    threads = min(256, get_max_num_threads_per_block(target))
    invalid_index = -1

    def choose_lhs(lhs_index, lhs_value, rhs_index, rhs_value):
        return tvm.tirx.all(
            lhs_index != invalid_index,
            tvm.tirx.any(
                rhs_index == invalid_index,
                lhs_value > rhs_value,
                tvm.tirx.all(lhs_value == rhs_value, lhs_index < rhs_index),
            ),
        )

    def combine(lhs_index, lhs_value, rhs_index, rhs_value):
        lhs_wins = choose_lhs(lhs_index, lhs_value, rhs_index, rhs_value)
        return (
            T.Select(lhs_wins, lhs_index, rhs_index),
            T.Select(lhs_wins, lhs_value, rhs_value),
        )

    batch_size = T.dynamic("batch_size", "int32")
    vocab_size = T.dynamic("vocab_size", "int32")
    # Rows are scanned only up to the model's active vocabulary when it is
    # smaller than the padded vocabulary the logits are laid out with.
    scan_limit = vocab_size if active_vocab_size is None else active_vocab_size

    @Ts.prim_func(private=True)
    def greedy_argmax(
        logits: T.Buffer((batch_size, 1, vocab_size), "float32"),
        output: T.Buffer((batch_size,), "int32"),
    ):
        T.func_attr({"tirx.is_scheduled": 1, "tirx.noalias": True})
        with Ts.sblock("kernel"):
            best_index = Ts.sblock_alloc_buffer((1,), "int32", scope="local")
            best_value = Ts.sblock_alloc_buffer((1,), "float32", scope="local")
            reduced_index = Ts.sblock_alloc_buffer((1,), "int32", scope="local")
            reduced_value = Ts.sblock_alloc_buffer((1,), "float32", scope="local")
            for block in T.thread_binding(0, batch_size, thread="blockIdx.x"):
                for thread in T.thread_binding(0, threads, thread="threadIdx.x"):
                    with Ts.sblock("argmax"):
                        batch = Ts.axis.S(batch_size, block)
                        tx = Ts.axis.S(threads, thread)
                        best_index[0] = invalid_index
                        best_value[0] = T.min_value("float32")
                        for offset in T.serial(T.ceildiv(scan_limit, threads)):
                            index = T.meta_var(offset * threads + tx)
                            if index < vocab_size and index < scan_limit:
                                value = T.meta_var(logits[batch, 0, index])
                                if choose_lhs(index, value, best_index[0], best_value[0]):
                                    best_index[0] = index
                                    best_value[0] = value
                        with Ts.sblock("cross_thread"):
                            Ts.reads(best_index[0], best_value[0])
                            Ts.writes(reduced_index[0], reduced_value[0])
                            T.attr(
                                T.comm_reducer(
                                    combine,
                                    [invalid_index, T.min_value("float32")],
                                ),
                                "reduce_scope",
                                T.int32(0),
                            )
                            T.tvm_thread_allreduce(
                                T.uint32(2),
                                best_index[0],
                                best_value[0],
                                True,
                                reduced_index[0],
                                reduced_value[0],
                                tx,
                                dtype="void",
                            )
                        if tx == 0:
                            output[batch] = reduced_index[0]

    return greedy_argmax


batch_size = T.dynamic("batch_size", "int32")


@Ts.prim_func
def full(value: T.int64, result: T.Buffer((batch_size, 1), "int32")):
    """The filling function for top k."""
    for i in T.serial(batch_size):
        with Ts.sblock("block"):
            vi = Ts.axis.spatial(batch_size, i)
            result[vi, 0] = T.cast(value, "int32")


def _attach_sample_with_top_p(bb: relax.BlockBuilder):
    batch_size = tirx.Var("batch_size", "int64")
    num_samples = tirx.Var("num_samples", "int64")
    vocab_size = tirx.Var("vocab_size", "int64")
    sorted_probs = relax.Var("sorted_probs", relax.TensorType((batch_size, vocab_size), "float32"))
    sorted_indices = relax.Var(
        "sorted_indices", relax.TensorType((batch_size, vocab_size), "int32")
    )
    uniform_samples = relax.Var("uniform_samples", relax.TensorType((num_samples,), "float32"))
    sample_indices = relax.Var("sample_indices", relax.TensorType((num_samples,), "int32"))
    top_p = relax.Var("top_p", relax.TensorType((batch_size,), "float32"))

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
                    ty_args=relax.TensorType(sample_shape, "float32"),
                ),
                name="uniform_samples",
            )
            sample_indices_tensor = nn.wrap_nested(
                relax.call_pure_packed(
                    "vm.builtin.reshape",
                    sample_indices,
                    sample_shape,
                    ty_args=relax.TensorType(sample_shape, "int32"),
                ),
                name="sample_indices",
            )
            top_p_tensor = nn.wrap_nested(
                relax.call_pure_packed(
                    "vm.builtin.reshape",
                    top_p,
                    top_p_shape,
                    ty_args=relax.TensorType(top_p_shape, "float32"),
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

            result_tensor = nn.sample_top_p_top_k_from_sorted_prob(
                sorted_probs_tensor,
                sorted_indices_tensor,
                top_p_tensor,
                top_k_tensor,
                uniform_samples_tensor,
                sample_indices_tensor,
            )
            result = bb.emit_output(
                relax.call_pure_packed(
                    "vm.builtin.reshape",
                    result_tensor._expr,
                    sample_indices.ty.shape,
                    ty_args=sample_indices.ty,
                )
            )
        gv = bb.emit_func_output(result)
    return gv


def _attach_renormalize_by_top_p(bb: relax.BlockBuilder, target: tvm.target.Target):
    batch_size = tirx.Var("batch_size", "int64")
    vocab_size = tirx.Var("vocab_size", "int64")
    num_pivots = 3
    probs = relax.Var("probs", relax.TensorType((batch_size, vocab_size), "float32"))
    top_p = relax.Var("top_p", relax.TensorType((batch_size,), "float32"))
    init_pivots = relax.Var("init_pivots", relax.TensorType((batch_size, num_pivots), "float32"))
    with bb.function("renormalize_by_top_p", [probs, top_p, init_pivots]):
        with bb.dataflow():
            cutoff_output = bb.emit(
                relax.call_tir(
                    bb.add_func(top_p_pivot(num_pivots, target), "top_p_pivot_cutoff"),
                    args=[probs, top_p, init_pivots],
                    out_ty=[top_p.ty, top_p.ty],
                )
            )
            final_pivot = cutoff_output[0]
            renorm_sum = cutoff_output[1]
            renormalized_probs = bb.emit_output(
                relax.call_tir(
                    bb.add_func(top_p_renorm(target), "top_p_renorm_after_cutoff"),
                    args=[probs, final_pivot, renorm_sum],
                    out_ty=probs.ty,
                )
            )
        gv = bb.emit_func_output(renormalized_probs)
    return gv


def _attach_take_probs_func(bb: relax.BlockBuilder):
    batch_size = T.dynamic("batch_size", "int32")
    num_samples = T.dynamic("num_samples", "int32")
    num_positions = T.dynamic("num_positions", "int32")
    vocab_size = T.dynamic("vocab_size", "int32")

    @Ts.prim_func
    def sampler_take_probs_tir(
        unsorted_probs: T.Buffer((batch_size, vocab_size), "float32"),
        sorted_indices: T.Buffer((batch_size, vocab_size), "int32"),
        sample_indices: T.Buffer((num_samples,), "int32"),
        sampling_results: T.Buffer((num_samples,), "int32"),
        top_prob_offsets: T.Buffer((num_positions,), "int32"),
        sampled_values: T.Buffer((num_samples,), "float32"),
        top_prob_probs: T.Buffer((num_positions,), "float32"),
        top_prob_indices: T.Buffer((num_positions,), "int32"),
    ):
        for i in T.serial(num_positions):
            with Ts.sblock("top_prob"):
                vi = Ts.axis.spatial(num_positions, i)
                # Reads are data-dependent gathers; declare full-buffer read
                # regions explicitly so tirx does not infer data-dependent regions.
                Ts.reads(
                    top_prob_offsets[vi],
                    sorted_indices[0:batch_size, 0:vocab_size],
                    unsorted_probs[0:batch_size, 0:vocab_size],
                )
                Ts.writes(top_prob_indices[vi], top_prob_probs[vi])
                row = T.floordiv(top_prob_offsets[vi], vocab_size)
                col = T.floormod(top_prob_offsets[vi], vocab_size)
                top_prob_indices[vi] = sorted_indices[row, col]
                top_prob_probs[vi] = unsorted_probs[row, sorted_indices[row, col]]
        for i in T.serial(num_samples):
            with Ts.sblock("sample"):
                vj = Ts.axis.spatial(num_samples, i)
                Ts.reads(
                    sample_indices[vj],
                    sampling_results[vj],
                    unsorted_probs[0:batch_size, 0:vocab_size],
                )
                Ts.writes(sampled_values[vj])
                sampled_values[vj] = unsorted_probs[sample_indices[vj], sampling_results[vj]]

    batch_size = tirx.Var("batch_size", "int64")
    num_samples = tirx.Var("num_samples", "int64")
    num_positions = tirx.Var("num_positions", "int64")
    vocab_size = tirx.Var("vocab_size", "int64")
    unsorted_probs = relax.Var(
        "unsorted_probs", relax.TensorType((batch_size, vocab_size), "float32")
    )
    sorted_indices = relax.Var(
        "sorted_indices", relax.TensorType((batch_size, vocab_size), "int32")
    )
    sample_indices = relax.Var("sample_indices", relax.TensorType((num_samples,), "int32"))
    sampling_results = relax.Var("sampling_result", relax.TensorType((num_samples,), "int32"))
    top_prob_offsets = relax.Var("lobprob_offsets", relax.TensorType((num_positions,), "int32"))

    args = [
        unsorted_probs,
        sorted_indices,
        sample_indices,
        sampling_results,
        top_prob_offsets,
    ]
    with bb.function("sampler_take_probs", args):
        with bb.dataflow():
            taken_probs_indices = bb.emit_output(
                relax.call_tir(
                    bb.add_func(sampler_take_probs_tir, "sampler_take_probs_tir"),
                    args,
                    out_ty=[
                        relax.TensorType((num_samples,), "float32"),
                        relax.TensorType((num_positions,), "float32"),
                        relax.TensorType((num_positions,), "int32"),
                    ],
                )
            )
        gv = bb.emit_func_output(taken_probs_indices)
    return gv


def _attach_batch_verifier(bb: relax.BlockBuilder):
    num_nodes = tirx.Var("num_nodes", "int64")
    nbatch = tirx.Var("nbatch", "int64")
    vocab_size = tirx.Var("vocab_size", "int64")
    draft_probs = relax.Var("draft_probs", relax.TensorType((num_nodes, vocab_size), "float32"))
    draft_tokens = relax.Var("draft_tokens", relax.TensorType((num_nodes,), "int32"))
    model_probs = relax.Var("model_probs", relax.TensorType((num_nodes, vocab_size), "float32"))
    token_tree_first_child = relax.Var(
        "token_tree_first_child", relax.TensorType((num_nodes,), "int32")
    )
    token_tree_next_sibling = relax.Var(
        "token_tree_next_sibling", relax.TensorType((num_nodes,), "int32")
    )
    uniform_samples = relax.Var("uniform_samples", relax.TensorType((num_nodes,), "float32"))
    token_tree_parent_ptr = relax.Var("token_tree_parent_ptr", relax.TensorType((nbatch,), "int32"))
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
            res = bb.emit_output(
                relax.call_tir_inplace(
                    bb.add_func(
                        batch_spec_verify(vocab_size),
                        "batch_verify_on_gpu_single_kernel",
                    ),
                    args,
                    inplace_indices=[
                        args.index(model_probs),
                        args.index(token_tree_parent_ptr),
                    ],
                    out_ty=[
                        model_probs.ty,
                        token_tree_parent_ptr.ty,
                    ],
                )
            )
        gv = bb.emit_func_output(res)
    return gv
