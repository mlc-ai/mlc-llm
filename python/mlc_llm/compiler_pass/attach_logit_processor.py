"""The pass that attaches logit processor functions to the IRModule."""

import tvm
from tvm import IRModule
from tvm.script import s_tir as Ts
from tvm.script import tirx as T

from ..support.max_thread_check import (
    check_thread_limits,
    get_max_num_threads_per_block,
)


@tvm.transform.module_pass(opt_level=0, name="AttachLogitProcessFunc")
class AttachLogitProcessFunc:
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
        if self.target.kind.name == "llvm":
            mod["apply_logit_bias_inplace"] = _get_apply_logit_bias_inplace_cpu()
            mod["apply_penalty_inplace"] = _get_apply_penalty_inplace_cpu()
            mod["apply_bitmask_inplace"] = _get_apply_bitmask_inplace_cpu()
        else:
            mod["apply_logit_bias_inplace"] = _get_apply_logit_bias_inplace(self.target)
            mod["apply_penalty_inplace"] = _get_apply_penalty_inplace(self.target)
            mod["apply_bitmask_inplace"] = _get_apply_bitmask_inplace(self.target)
        return mod


def _get_apply_logit_bias_inplace_cpu():
    batch_size = T.dynamic("batch_size", "int32")
    vocab_size = T.dynamic("vocab_size", "int32")
    num_token = T.dynamic("num_token", "int32")

    @Ts.prim_func
    def _apply_logit_bias_inplace(
        logits: T.Tensor((batch_size, vocab_size), "float32"),
        # seq_ids
        pos2seq_id: T.Tensor((num_token,), "int32"),
        token_ids: T.Tensor((num_token,), "int32"),
        logit_bias: T.Tensor((num_token,), "float32"),
    ) -> None:
        """Function that applies logit bias in place."""
        T.func_attr(
            {
                "global_symbol": "apply_logit_bias_inplace",
                "tirx.noalias": True,
                "tirx.is_scheduled": True,
            }
        )
        for i in range(num_token):
            logits[pos2seq_id[i], token_ids[i]] += logit_bias[i]

    return _apply_logit_bias_inplace


def _get_apply_logit_bias_inplace(target: tvm.target.Target):
    tx = 1024  # default
    max_num_threads_per_block = get_max_num_threads_per_block(target)
    tx = min(tx, max_num_threads_per_block)
    check_thread_limits(target, bdx=tx, bdy=1, bdz=1, gdz=1)

    batch_size = T.dynamic("batch_size", "int32")
    vocab_size = T.dynamic("vocab_size", "int32")
    num_token = T.dynamic("num_token", "int32")

    @Ts.prim_func
    def _apply_logit_bias_inplace(
        logits: T.Tensor((batch_size, vocab_size), "float32"),
        # seq_ids
        pos2seq_id: T.Tensor((num_token,), "int32"),
        token_ids: T.Tensor((num_token,), "int32"),
        logit_bias: T.Tensor((num_token,), "float32"),
    ) -> None:
        """Function that applies logit bias in place."""
        T.func_attr(
            {
                "global_symbol": "apply_logit_bias_inplace",
                "tirx.noalias": True,
                "tirx.is_scheduled": True,
            }
        )
        for p0 in T.thread_binding(0, (num_token + tx - 1) // tx, "blockIdx.x"):
            for p1 in T.thread_binding(0, tx, "threadIdx.x"):
                with Ts.sblock("block"):
                    vp = Ts.axis.spatial(num_token, p0 * tx + p1)
                    Ts.where(p0 * tx + p1 < num_token)
                    logits[pos2seq_id[vp], token_ids[vp]] += logit_bias[vp]

    return _apply_logit_bias_inplace


def _get_apply_penalty_inplace_cpu():
    batch_size = T.dynamic("batch_size", "int32")
    vocab_size = T.dynamic("vocab_size", "int32")
    num_token = T.dynamic("num_token", "int32")
    num_seq = T.dynamic("num_seq", "int32")

    @Ts.prim_func
    def _apply_penalty_inplace(
        logits: T.Tensor((batch_size, vocab_size), "float32"),
        seq_ids: T.Tensor((num_seq,), "int32"),
        pos2seq_id: T.Tensor((num_token,), "int32"),
        token_ids: T.Tensor((num_token,), "int32"),
        token_cnt: T.Tensor((num_token,), "int32"),
        penalties: T.Tensor((num_seq, 3), "float32"),
    ) -> None:
        """Function that applies penalties in place."""
        T.func_attr(
            {
                "global_symbol": "apply_penalty_inplace",
                "tirx.noalias": True,
                "tirx.is_scheduled": True,
            }
        )
        for token in T.serial(num_token):
            with Ts.sblock("block"):
                vp = Ts.axis.spatial(num_token, token)
                logits[seq_ids[pos2seq_id[vp]], token_ids[vp]] -= (
                    penalties[pos2seq_id[vp], 0] + token_cnt[vp] * penalties[pos2seq_id[vp], 1]
                )
                logits[seq_ids[pos2seq_id[vp]], token_ids[vp]] = T.if_then_else(
                    logits[seq_ids[pos2seq_id[vp]], token_ids[vp]] < T.float32(0),
                    logits[seq_ids[pos2seq_id[vp]], token_ids[vp]] * penalties[pos2seq_id[vp], 2],
                    logits[seq_ids[pos2seq_id[vp]], token_ids[vp]] / penalties[pos2seq_id[vp], 2],
                )

    return _apply_penalty_inplace


def _get_apply_penalty_inplace(target: tvm.target.Target):
    tx = 1024  # default
    max_num_threads_per_block = get_max_num_threads_per_block(target)
    tx = min(tx, max_num_threads_per_block)
    check_thread_limits(target, bdx=tx, bdy=1, bdz=1, gdz=1)

    batch_size = T.dynamic("batch_size", "int32")
    vocab_size = T.dynamic("vocab_size", "int32")
    num_token = T.dynamic("num_token", "int32")
    num_seq = T.dynamic("num_seq", "int32")

    @Ts.prim_func
    def _apply_penalty_inplace(
        logits: T.Tensor((batch_size, vocab_size), "float32"),
        seq_ids: T.Tensor((num_seq,), "int32"),
        pos2seq_id: T.Tensor((num_token,), "int32"),
        token_ids: T.Tensor((num_token,), "int32"),
        token_cnt: T.Tensor((num_token,), "int32"),
        penalties: T.Tensor((num_seq, 3), "float32"),
    ) -> None:
        """Function that applies penalties in place."""
        T.func_attr(
            {
                "global_symbol": "apply_penalty_inplace",
                "tirx.noalias": True,
                "tirx.is_scheduled": True,
            }
        )
        for p0 in T.thread_binding(0, (num_token + tx - 1) // tx, "blockIdx.x"):
            for p1 in T.thread_binding(0, tx, "threadIdx.x"):
                with Ts.sblock("block"):
                    vp = Ts.axis.spatial(num_token, p0 * tx + p1)
                    Ts.where(p0 * tx + p1 < num_token)
                    # Penalties: (presence_penalty, frequency_penalty, repetition_penalty)
                    logits[seq_ids[pos2seq_id[vp]], token_ids[vp]] -= (
                        penalties[pos2seq_id[vp], 0] + token_cnt[vp] * penalties[pos2seq_id[vp], 1]
                    )
                    logits[seq_ids[pos2seq_id[vp]], token_ids[vp]] = T.if_then_else(
                        logits[seq_ids[pos2seq_id[vp]], token_ids[vp]] < T.float32(0),
                        logits[seq_ids[pos2seq_id[vp]], token_ids[vp]]
                        * penalties[pos2seq_id[vp], 2],
                        logits[seq_ids[pos2seq_id[vp]], token_ids[vp]]
                        / penalties[pos2seq_id[vp], 2],
                    )

    return _apply_penalty_inplace


def _get_apply_bitmask_inplace_cpu():
    batch_size = T.dynamic("batch_size", "int32")
    vocab_size = T.dynamic("vocab_size", "int32")
    num_seq = T.dynamic("num_seq", "int32")

    @Ts.prim_func
    def _apply_bitmask_inplace(
        logits: T.Tensor((batch_size, vocab_size), "float32"),
        seq_ids: T.Tensor((num_seq,), "int32"),
        bitmask: T.Tensor((batch_size, (vocab_size + 31) // 32), "int32"),
    ) -> None:
        """Function that applies vocabulary masking in place."""
        T.func_attr(
            {
                "global_symbol": "apply_bitmask_inplace",
                "tirx.noalias": True,
                "tirx.is_scheduled": True,
            }
        )
        for token in T.serial(num_seq * vocab_size):
            with Ts.sblock("block"):
                vs = Ts.axis.spatial(num_seq, (token) // vocab_size)
                vv = Ts.axis.spatial(vocab_size, (token) % vocab_size)

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

    batch_size = T.dynamic("batch_size", "int32")
    vocab_size = T.dynamic("vocab_size", "int32")
    num_seq = T.dynamic("num_seq", "int32")

    @Ts.prim_func
    def _apply_bitmask_inplace(
        logits: T.Tensor((batch_size, vocab_size), "float32"),
        seq_ids: T.Tensor((num_seq,), "int32"),
        bitmask: T.Tensor((batch_size, (vocab_size + 31) // 32), "int32"),
    ) -> None:
        """Function that applies vocabulary masking in place."""
        T.func_attr(
            {
                "global_symbol": "apply_bitmask_inplace",
                "tirx.noalias": True,
                "tirx.is_scheduled": True,
            }
        )
        for fused_s_v_0 in T.thread_binding(0, (num_seq * vocab_size + tx - 1) // tx, "blockIdx.x"):
            for fused_s_v_1 in T.thread_binding(0, tx, "threadIdx.x"):
                with Ts.sblock("block"):
                    vs = Ts.axis.spatial(num_seq, (fused_s_v_0 * tx + fused_s_v_1) // vocab_size)
                    vv = Ts.axis.spatial(vocab_size, (fused_s_v_0 * tx + fused_s_v_1) % vocab_size)
                    Ts.where(fused_s_v_0 * tx + fused_s_v_1 < num_seq * vocab_size)
                    logits[seq_ids[vs], vv] = T.if_then_else(
                        (bitmask[seq_ids[vs], vv // 32] >> (vv % 32)) & 1 == 1,
                        logits[seq_ids[vs], vv],
                        T.min_value("float32"),
                    )

    return _apply_bitmask_inplace
