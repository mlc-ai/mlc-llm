"""The pass that attaches logit processor functions to the IRModule."""

import tvm
from tvm import IRModule
from tvm.script import tir as T

from ..support.max_thread_check import (
    check_thread_limits,
    get_max_num_threads_per_block,
)


@tvm.transform.module_pass(opt_level=0, name="AttachLogitProcessFunc")
class AttachLogitProcessFunc:  # pylint: disable=too-few-public-methods
    """Attach logit processing TIR functions to IRModule."""

    def __init__(self, target: tvm.target.Target):
        """Initializer.

        Parameters
        ----------
        target : tvm.target.Target
            The target of the model compilation.
        """
        self.target = target

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """Entrypoint"""
        mod = mod.clone()
        if str(self.target.kind) == "llvm":
            mod["apply_logit_bias_inplace"] = _get_apply_logit_bias_inplace_cpu()
            mod["apply_penalty_inplace"] = _get_apply_penalty_inplace_cpu()
            mod["apply_bitmask_inplace"] = _get_apply_bitmask_inplace_cpu()
        else:
            mod["apply_logit_bias_inplace"] = _get_apply_logit_bias_inplace(self.target)
            mod["apply_penalty_inplace"] = _get_apply_penalty_inplace(self.target)
            mod["apply_bitmask_inplace"] = _get_apply_bitmask_inplace(self.target)
        return mod


def _get_apply_logit_bias_inplace_cpu():
    @T.prim_func
    def _apply_logit_bias_inplace(
        var_logits: T.handle,
        var_pos2seq_id: T.handle,
        var_token_ids: T.handle,
        var_logit_bias: T.handle,
    ) -> None:
        """Function that applies logit bias in place."""
        T.func_attr(
            {
                "global_symbol": "apply_logit_bias_inplace",
                "tir.noalias": True,
                "tir.is_scheduled": True,
            }
        )
        batch_size = T.int32(is_size_var=True)
        vocab_size = T.int32(is_size_var=True)
        num_token = T.int32(is_size_var=True)
        logits = T.match_buffer(var_logits, (batch_size, vocab_size), "float32")
        # seq_ids
        pos2seq_id = T.match_buffer(var_pos2seq_id, (num_token,), "int32")
        token_ids = T.match_buffer(var_token_ids, (num_token,), "int32")
        logit_bias = T.match_buffer(var_logit_bias, (num_token,), "float32")

        for i in range(num_token):
            logits[pos2seq_id[i], token_ids[i]] += logit_bias[i]

    return _apply_logit_bias_inplace


def _get_apply_logit_bias_inplace(target: tvm.target.Target):
    tx = 1024  # default
    max_num_threads_per_block = get_max_num_threads_per_block(target)
    tx = min(tx, max_num_threads_per_block)
    check_thread_limits(target, bdx=tx, bdy=1, bdz=1, gdz=1)

    @T.prim_func
    def _apply_logit_bias_inplace(
        var_logits: T.handle,
        var_pos2seq_id: T.handle,
        var_token_ids: T.handle,
        var_logit_bias: T.handle,
    ) -> None:
        """Function that applies logit bias in place."""
        T.func_attr(
            {
                "global_symbol": "apply_logit_bias_inplace",
                "tir.noalias": True,
                "tir.is_scheduled": True,
            }
        )
        batch_size = T.int32(is_size_var=True)
        vocab_size = T.int32(is_size_var=True)
        num_token = T.int32(is_size_var=True)
        logits = T.match_buffer(var_logits, (batch_size, vocab_size), "float32")
        # seq_ids
        pos2seq_id = T.match_buffer(var_pos2seq_id, (num_token,), "int32")
        token_ids = T.match_buffer(var_token_ids, (num_token,), "int32")
        logit_bias = T.match_buffer(var_logit_bias, (num_token,), "float32")

        for p0 in T.thread_binding(0, (num_token + tx - 1) // tx, "blockIdx.x"):
            for p1 in T.thread_binding(0, tx, "threadIdx.x"):
                with T.block("block"):
                    vp = T.axis.spatial(num_token, p0 * tx + p1)
                    T.where(p0 * tx + p1 < num_token)
                    logits[pos2seq_id[vp], token_ids[vp]] += logit_bias[vp]

    return _apply_logit_bias_inplace


def _get_apply_penalty_inplace_cpu():
    @T.prim_func
    def _apply_penalty_inplace(  # pylint: disable=too-many-arguments,too-many-locals
        var_logits: T.handle,
        var_seq_ids: T.handle,
        var_pos2seq_id: T.handle,
        var_token_ids: T.handle,
        var_token_cnt: T.handle,
        var_penalties: T.handle,
    ) -> None:
        """Function that applies penalties in place."""
        T.func_attr(
            {
                "global_symbol": "apply_penalty_inplace",
                "tir.noalias": True,
                "tir.is_scheduled": True,
            }
        )
        batch_size = T.int32(is_size_var=True)
        vocab_size = T.int32(is_size_var=True)
        num_token = T.int32(is_size_var=True)
        num_seq = T.int32(is_size_var=True)
        logits = T.match_buffer(var_logits, (batch_size, vocab_size), "float32")
        seq_ids = T.match_buffer(var_seq_ids, (num_seq,), "int32")
        pos2seq_id = T.match_buffer(var_pos2seq_id, (num_token,), "int32")
        token_ids = T.match_buffer(var_token_ids, (num_token,), "int32")
        token_cnt = T.match_buffer(var_token_cnt, (num_token,), "int32")
        penalties = T.match_buffer(var_penalties, (num_seq, 3), "float32")

        for token in T.serial(num_token):
            with T.block("block"):
                vp = T.axis.spatial(num_token, token)
                logits[seq_ids[pos2seq_id[vp]], token_ids[vp]] -= (
                    penalties[pos2seq_id[vp], 0] + token_cnt[vp] * penalties[pos2seq_id[vp], 1]
                )
                logits[seq_ids[pos2seq_id[vp]], token_ids[vp]] = T.if_then_else(
                    logits[seq_ids[pos2seq_id[vp]], token_ids[vp]] < 0,
                    logits[seq_ids[pos2seq_id[vp]], token_ids[vp]] * penalties[pos2seq_id[vp], 2],
                    logits[seq_ids[pos2seq_id[vp]], token_ids[vp]] / penalties[pos2seq_id[vp], 2],
                )

    return _apply_penalty_inplace


def _get_apply_penalty_inplace(target: tvm.target.Target):
    tx = 1024  # default
    max_num_threads_per_block = get_max_num_threads_per_block(target)
    tx = min(tx, max_num_threads_per_block)
    check_thread_limits(target, bdx=tx, bdy=1, bdz=1, gdz=1)

    @T.prim_func
    def _apply_penalty_inplace(  # pylint: disable=too-many-arguments,too-many-locals
        var_logits: T.handle,
        var_seq_ids: T.handle,
        var_pos2seq_id: T.handle,
        var_token_ids: T.handle,
        var_token_cnt: T.handle,
        var_penalties: T.handle,
    ) -> None:
        """Function that applies penalties in place."""
        T.func_attr(
            {
                "global_symbol": "apply_penalty_inplace",
                "tir.noalias": True,
                "tir.is_scheduled": True,
            }
        )
        batch_size = T.int32(is_size_var=True)
        vocab_size = T.int32(is_size_var=True)
        num_token = T.int32(is_size_var=True)
        num_seq = T.int32(is_size_var=True)
        logits = T.match_buffer(var_logits, (batch_size, vocab_size), "float32")
        seq_ids = T.match_buffer(var_seq_ids, (num_seq,), "int32")
        pos2seq_id = T.match_buffer(var_pos2seq_id, (num_token,), "int32")
        token_ids = T.match_buffer(var_token_ids, (num_token,), "int32")
        token_cnt = T.match_buffer(var_token_cnt, (num_token,), "int32")
        penalties = T.match_buffer(var_penalties, (num_seq, 3), "float32")

        for p0 in T.thread_binding(0, (num_token + tx - 1) // tx, "blockIdx.x"):
            for p1 in T.thread_binding(0, tx, "threadIdx.x"):
                with T.block("block"):
                    vp = T.axis.spatial(num_token, p0 * tx + p1)
                    T.where(p0 * tx + p1 < num_token)
                    # Penalties: (presence_penalty, frequency_penalty, repetition_penalty)
                    logits[seq_ids[pos2seq_id[vp]], token_ids[vp]] -= (
                        penalties[pos2seq_id[vp], 0] + token_cnt[vp] * penalties[pos2seq_id[vp], 1]
                    )
                    logits[seq_ids[pos2seq_id[vp]], token_ids[vp]] = T.if_then_else(
                        logits[seq_ids[pos2seq_id[vp]], token_ids[vp]] < 0,
                        logits[seq_ids[pos2seq_id[vp]], token_ids[vp]]
                        * penalties[pos2seq_id[vp], 2],
                        logits[seq_ids[pos2seq_id[vp]], token_ids[vp]]
                        / penalties[pos2seq_id[vp], 2],
                    )

    return _apply_penalty_inplace


def _get_apply_bitmask_inplace_cpu():
    @T.prim_func
    def _apply_bitmask_inplace(
        var_logits: T.handle,
        var_seq_ids: T.handle,
        var_bitmask: T.handle,
    ) -> None:
        """Function that applies vocabulary masking in place."""
        T.func_attr(
            {
                "global_symbol": "apply_bitmask_inplace",
                "tir.noalias": True,
                "tir.is_scheduled": True,
            }
        )
        batch_size = T.int32(is_size_var=True)
        vocab_size = T.int32(is_size_var=True)
        num_seq = T.int32(is_size_var=True)
        logits = T.match_buffer(var_logits, (batch_size, vocab_size), "float32")
        seq_ids = T.match_buffer(var_seq_ids, (num_seq,), "int32")
        bitmask = T.match_buffer(var_bitmask, (batch_size, (vocab_size + 31) // 32), "int32")

        for token in T.serial(num_seq * vocab_size):
            with T.block("block"):
                vs = T.axis.spatial(num_seq, (token) // vocab_size)
                vv = T.axis.spatial(vocab_size, (token) % vocab_size)

                logits[seq_ids[vs], vv] = T.if_then_else(
                    (bitmask[seq_ids[vs], vv // 32] >> (vv % 32)) & 1 == 1,
                    logits[seq_ids[vs], vv],
                    T.min_value("float32"),
                )

    return _apply_bitmask_inplace


def _get_apply_bitmask_inplace(target: tvm.target.Target):
    tx = 1024  # default
    max_num_threads_per_block = get_max_num_threads_per_block(target)
    tx = min(tx, max_num_threads_per_block)
    check_thread_limits(target, bdx=tx, bdy=1, bdz=1, gdz=1)

    @T.prim_func
    def _apply_bitmask_inplace(
        var_logits: T.handle,
        var_seq_ids: T.handle,
        var_bitmask: T.handle,
    ) -> None:
        """Function that applies vocabulary masking in place."""
        T.func_attr(
            {
                "global_symbol": "apply_bitmask_inplace",
                "tir.noalias": True,
                "tir.is_scheduled": True,
            }
        )
        batch_size = T.int32(is_size_var=True)
        vocab_size = T.int32(is_size_var=True)
        num_seq = T.int32(is_size_var=True)
        logits = T.match_buffer(var_logits, (batch_size, vocab_size), "float32")
        seq_ids = T.match_buffer(var_seq_ids, (num_seq,), "int32")
        bitmask = T.match_buffer(var_bitmask, (batch_size, (vocab_size + 31) // 32), "int32")

        for fused_s_v_0 in T.thread_binding(0, (num_seq * vocab_size + tx - 1) // tx, "blockIdx.x"):
            for fused_s_v_1 in T.thread_binding(0, tx, "threadIdx.x"):
                with T.block("block"):
                    vs = T.axis.spatial(num_seq, (fused_s_v_0 * tx + fused_s_v_1) // vocab_size)
                    vv = T.axis.spatial(vocab_size, (fused_s_v_0 * tx + fused_s_v_1) % vocab_size)
                    T.where(fused_s_v_0 * tx + fused_s_v_1 < num_seq * vocab_size)
                    logits[seq_ids[vs], vv] = T.if_then_else(
                        (bitmask[seq_ids[vs], vv // 32] >> (vv % 32)) & 1 == 1,
                        logits[seq_ids[vs], vv],
                        T.min_value("float32"),
                    )

    return _apply_bitmask_inplace
