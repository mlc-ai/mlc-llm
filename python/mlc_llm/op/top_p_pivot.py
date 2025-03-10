"""Operators for choosing the pivot to cut-off top-p percentile"""

import tvm
from tvm.script import tir as T

from mlc_llm.support.max_thread_check import get_max_num_threads_per_block

# mypy: disable-error-code="attr-defined,valid-type,name-defined"
# pylint: disable=too-many-locals,invalid-name,too-many-arguments,unnecessary-lambda
# pylint: disable=too-many-statements,line-too-long,too-many-nested-blocks,too-many-branches


def top_p_pivot(pN, target: tvm.target.Target):
    """Top-p pivot function. This function finds the pivot to cut-off top-p percentile.

    A valide pivot should satisfy the following conditions:
    - lsum >= top_p
    - top_p > lsum - cmin * lmin
    where lsum is the sum of elements that are larger or equal to the pivot,
    lmin is the minimum elements that is larger or equal to the pivot,
    cmin is the count of elements that are equal to lmin,

    Parameters
    ----------
    prob:
        The probability vector

    top_p_arr:
        The top-p threshold

    init_pivots:
        The initial pivot candidates

    final_pivot:
        The final pivot to cut-off top-p percentile

    final_lsum:
        The final sum of the values after top-p filtering.
    """
    TX = 1024
    K = 32
    eps_LR = 1e-7

    max_num_threads_per_block = get_max_num_threads_per_block(target)
    TX = min(TX, max_num_threads_per_block)

    def _var(dtype="int32"):
        return T.alloc_buffer((1,), dtype, scope="local")

    def valid(lsum, lmin, cmin, top_p):
        return tvm.tir.all(lsum >= top_p, top_p > lsum - cmin * lmin)

    # fmt: off
    @T.prim_func(private=True)
    def _func(
        var_prob: T.handle,
        var_top_p_arr: T.handle,
        var_init_pivots: T.handle,
        var_final_pivot: T.handle,
        var_final_lsum: T.handle,
    ):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": True})
        B = T.int32(is_size_var=True)
        N = T.int32(is_size_var=True)
        prob = T.match_buffer(var_prob, (B, N,), "float32")
        top_p_arr = T.match_buffer(var_top_p_arr, (B,), dtype="float32")
        init_pivots = T.match_buffer(var_init_pivots, (B, pN), "float32")
        final_pivot = T.match_buffer(var_final_pivot, (B,), "float32")
        final_lsum = T.match_buffer(var_final_lsum, (B,), "float32")

        with T.block("kernel"):
            pivot = T.alloc_buffer((pN,), "float32", scope="local")
            top_p = _var("float32")

            L = T.alloc_buffer((1,), "float32", scope="shared")
            R = T.alloc_buffer((1,), "float32", scope="shared")
            L_local = _var("float32")
            R_local = _var("float32")

            q = _var("float32")
            lsum = T.alloc_buffer((pN,), "float32", scope="local")
            lmin_broadcast = T.alloc_buffer((1), "float32", scope="shared")
            lmin_broadcast_local = _var("float32")
            lmin = T.alloc_buffer((pN,), "float32", scope="local")
            cmin = T.alloc_buffer((pN,), "int32", scope="local")
            total_sum = _var("float32")

            it = _var("int32")
            es_local = _var("bool")
            es = T.alloc_buffer((1,), "bool", scope="shared")
            find_pivot_local = _var("bool")
            find_pivot = T.alloc_buffer((1,), "bool", scope="shared")

            total_sum_reduce = _var("float32")
            lsum_reduce = _var("float32")
            lmin_reduce = _var("float32")
            cmin_reduce = _var("int32")

            for _bx in T.thread_binding(0, B, thread="blockIdx.x"):
                for _tx in T.thread_binding(0, TX, thread="threadIdx.x"):
                    with T.block("CTA"):
                        b, tx = T.axis.remap("SS", [_bx, _tx])

                        top_p[0] = top_p_arr[b]

                        if tx == 0:
                            # leader thread initializes L, R
                            L[0] = 1.0 - top_p[0]
                            R[0] = eps_LR
                            find_pivot[0] = False
                        T.tvm_storage_sync("shared")

                        L_local[0] = L[0]
                        R_local[0] = R[0]
                        for i in T.unroll(0, pN):
                            # pivots are in descending order
                            pivot[i] = init_pivots[b, i]
                        find_pivot_local[0] = False
                        if L_local[0] - R_local[0] <= eps_LR:
                            # When the initial value is too small, set the result directly.
                            if tx == 0:
                                final_lsum[b] = 1.0
                                final_pivot[b] = 0.0
                            find_pivot_local[0] = True

                        while T.tvm_thread_invariant(
                            L_local[0] - R_local[0] > eps_LR
                            and T.Not(find_pivot_local[0])
                        ):
                            # sync before each iteration
                            T.tvm_storage_sync("shared")

                            ### get lsum, lmin, total_sum
                            for pidx in T.unroll(0, pN):
                                lsum[pidx] = 0.0
                                lmin[pidx] = T.max_value("float32")
                                cmin[pidx] = 0
                            total_sum[0] = 0.0
                            it[0] = 0
                            es_local[0] = False
                            while it[0] < T.ceildiv(N, TX) and T.Not(es_local[0]):
                                idx = T.meta_var(it[0] * TX + tx)
                                q[0] = T.if_then_else(idx < N, prob[b, idx], 0.0)
                                total_sum[0] += q[0]
                                for pidx in T.unroll(0, pN):
                                    if q[0] >= pivot[pidx]:
                                        lsum[pidx] += q[0]
                                        if lmin[pidx] > q[0]:
                                            lmin[pidx] = q[0]
                                            cmin[pidx] = 1
                                        elif lmin[pidx] == q[0]:
                                            cmin[pidx] += 1
                                it[0] += 1

                                # early stop every K iterations
                                if it[0] % K == 0:
                                    # reduce total_sum over tx
                                    # T.tvm_storage_sync("shared")
                                    with T.block("block_cross_thread"):
                                        T.reads(total_sum[0])
                                        T.writes(total_sum_reduce[0])
                                        T.attr(
                                            T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
                                            "reduce_scope",
                                            T.reinterpret("handle", T.uint64(0)),
                                        )
                                        T.tvm_thread_allreduce(T.uint32(1), total_sum[0], True, total_sum_reduce[0], tx, dtype="handle")
                                    # T.tvm_storage_sync("shared")

                                    if tx == 0:
                                        # leader thread checks if we can stop early
                                        es[0] = 1 - total_sum_reduce[0] < pivot[pN - 1]
                                    T.tvm_storage_sync("shared")
                                    es_local[0] = es[0]

                            T.tvm_storage_sync("shared")

                            # reduce lsum, lmin, cmin, over tx
                            for pidx in T.serial(0, pN):
                                # reduce lsum over tx for pivot[j]
                                with T.block("block_cross_thread"):
                                    T.reads(lsum[pidx])
                                    T.writes(lsum_reduce[0])
                                    T.attr(
                                        T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
                                        "reduce_scope",
                                        T.reinterpret("handle", T.uint64(0)),
                                    )
                                    T.tvm_thread_allreduce(T.uint32(1), lsum[pidx], True, lsum_reduce[0], tx, dtype="handle")

                                # reduce lmin over tx for pivot[j]
                                with T.block("block_cross_thread"):
                                    T.reads(lmin[pidx])
                                    T.writes(lmin_reduce[0])
                                    T.attr(
                                        T.comm_reducer(lambda x0, y0: T.min(x0, y0), [T.float32(0)]),
                                        "reduce_scope",
                                        T.reinterpret("handle", T.uint64(0)),
                                    )
                                    T.tvm_thread_allreduce(T.uint32(1), lmin[pidx], True, lmin_reduce[0], tx, dtype="handle")

                                if tx == 0:
                                    # broadcast lmin to all threads
                                    lmin_broadcast[0] = lmin_reduce[0]
                                T.tvm_storage_sync("shared")
                                lmin_broadcast_local[0] = lmin_broadcast[0]
                                if lmin[pidx] > lmin_broadcast_local[0]:
                                    cmin[pidx] = 0
                                if tx == 0:
                                    # only the leader thread updates lsum, lmin
                                    lsum[pidx] = lsum_reduce[0]
                                    lmin[pidx] = lmin_reduce[0]

                                # reduce cmin over tx for pivot[j]
                                with T.block("block_cross_thread"):
                                    T.reads(cmin[pidx])
                                    T.writes(cmin_reduce[0])
                                    T.attr(
                                        T.comm_reducer(lambda x0, y0: x0 + y0, [T.int32(0)]),
                                        "reduce_scope",
                                        T.reinterpret("handle", T.uint64(0)),
                                    )
                                    T.tvm_thread_allreduce(T.uint32(1), cmin[pidx], True, cmin_reduce[0], tx, dtype="handle")

                                if tx == 0:
                                    # only the leader thread updates cmin
                                    cmin[pidx] = cmin_reduce[0]

                            T.tvm_storage_sync("shared")

                            if tx == 0:
                                # leader thread checks if we have found the pivot, or updates L, R
                                it[0] = 0
                                while it[0] < pN and T.Not(find_pivot_local[0]):
                                    pidx = T.meta_var(it[0])
                                    if valid(lsum[pidx], lmin[pidx], cmin[pidx], top_p[0]):
                                        find_pivot[0] = True
                                        find_pivot_local[0] = True
                                        # write back the pivot and lsum
                                        final_pivot[b] = pivot[pidx]
                                        final_lsum[b] = lsum[pidx]
                                    elif lsum[pidx] - lmin[pidx] * cmin[pidx] >= top_p[0]:
                                        R[0] = pivot[pidx]
                                        final_lsum[b] = lsum[pidx]
                                    elif lsum[pidx] < top_p[0]:
                                        L[0] = pivot[pidx]
                                    it[0] += 1

                            T.tvm_storage_sync("shared")

                            L_local[0] = L[0]
                            R_local[0] = R[0]
                            find_pivot_local[0] = find_pivot[0]
                            # new pivots for next iteration
                            # uniform spacing between L and R
                            for pidx in T.unroll(0, pN):
                                pivot[pidx] = L[0] - (pidx + 1) * (L_local[0] - R_local[0]) / (pN + 1)

                        if tx == 0:
                            # leader thread writes back the pivot
                            if T.Not(find_pivot_local[0]):
                                final_pivot[b] = R_local[0]
                                if R_local[0] == eps_LR:
                                    final_lsum[b] = lsum[pN - 1]
    # fmt: on

    return _func


def top_p_renorm(target: tvm.target.Target = None):
    """Top-p renormalization function. This function renormalizes the probability vector.

    Given the pivot, the probability vector is renormalized as follows:
    - if prob >= pivot, renorm_prob = prob / lsum
    - otherwise, renorm_prob = 0

    Parameters
    ----------
    prob:
        The probability vector

    final_pivot:
        The final pivot to cut-off top-p percentile

    final_lsum:
        The sum of elements that are larger or equal to the pivot

    renorm_prob:
        The renormalized probability vector
    """
    TX = 1024
    CTA_COUNT = 512

    if target:
        max_num_threads_per_block = get_max_num_threads_per_block(target)
        TX = min(TX, max_num_threads_per_block)

    def _var(dtype="int32"):
        return T.alloc_buffer((1,), dtype, scope="local")

    # fmt: off
    @T.prim_func(private=True)
    def _func(
        var_prob: T.handle,
        var_final_pivot: T.handle,
        var_final_lsum: T.handle,
        var_renorm_prob: T.handle,
    ):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": True})
        B = T.int32(is_size_var=True)
        N = T.int32(is_size_var=True)
        prob = T.match_buffer(var_prob, (B, N,), "float32")
        final_pivot = T.match_buffer(var_final_pivot, (B,), "float32")
        final_lsum = T.match_buffer(var_final_lsum, (B,), "float32")
        renorm_prob = T.match_buffer(var_renorm_prob, (B, N,), "float32")

        with T.block("kernel"):
            pivot = _var("float32")
            lsum = _var("float32")
            BX = T.meta_var(T.ceildiv(CTA_COUNT, B))

            for _by in T.thread_binding(0, B, thread="blockIdx.y"):
                for _bx in T.thread_binding(0, BX, thread="blockIdx.x"):
                    for _tx in T.thread_binding(0, TX, thread="threadIdx.x"):
                        with T.block("CTA"):
                            by, bx, tx = T.axis.remap("SSS", [_by, _bx, _tx])

                            pivot[0] = final_pivot[by]
                            lsum[0] = final_lsum[by]

                            for i in T.serial(T.ceildiv(N, BX * TX)):
                                idx = T.meta_var(i * BX * TX + bx * TX + tx)
                                if idx < N:
                                    renorm_prob[by, idx] = T.if_then_else(prob[by, idx] >= pivot[0], prob[by, idx] / lsum[0], 0.0)
    # fmt: on

    return _func
