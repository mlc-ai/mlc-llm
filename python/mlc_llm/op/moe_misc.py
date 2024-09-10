"""Mixture of Experts operators"""

from functools import reduce
from typing import Tuple, Union

from tvm import te, tir
from tvm.relax.frontend.nn import Tensor, op
from tvm.script import tir as T

# mypy: disable-error-code="attr-defined,name-defined"
# pylint: disable=line-too-long,too-many-locals,invalid-name


def moe_sum(x: Tensor, dim: int) -> Tensor:
    """Compute the sum of the input tensor along the given axis. It is specialized for the MoE
    case where `x.ndim == 3` and `x.shape[1] == num_experts_per_tok (which is 2)`.
    """
    if x.ndim == 3 and x.shape[1] == 2:
        return op.tensor_expr_op(
            lambda x: te.compute(
                (x.shape[0], x.shape[2]),
                lambda i, j: x[i, 0, j] + x[i, 1, j],
                name="sum_2",
            ),
            "sum",
            args=[x],
        )
    return op.sum(x, axis=dim)


def gating_softmax_topk(  # pylint: disable=too-many-statements
    x: Tensor, k: int, norm_topk_prob=True
) -> Tuple[Tensor, Tensor]:
    """Compute the softmax score, choose the top-k experts, and returns selected scores.

    Parameters
    ----------
    x : Tensor
        The input tensor with shape [batch_size, num_local_experts].

    k : int
        The number of top elements to be selected, which is `num_experts_per_tok` in MoE.

    norm_topk_prob : bool
        Whether to normalize the top-k expert scores.

    Returns
    -------
    expert_weights: Tensor
        The top-k expert scores with shape [batch_size, k].

    expert_indices: Tensor
        The top-k expert indices with shape [batch_size, k].
    """
    (batch_size, num_local_experts), dtype = x.shape, x.dtype
    index_dtype = "int32"

    TX = 1024
    SCAN_LEN_2 = 2
    SCAN_LEN_4 = 4

    # specialized kernel for top 2 case
    @T.prim_func(private=True)
    def top2_softmax_norm_func(
        var_x: T.handle,
        var_out: T.handle,
        var_out_index: T.handle,
    ) -> None:
        T.func_attr({"tir.noalias": True, "tir.is_scheduled": True})
        batch_size = T.int64()
        x = T.match_buffer(var_x, (batch_size, num_local_experts), dtype)
        out = T.match_buffer(var_out, (batch_size, SCAN_LEN_2), dtype)
        out_index = T.match_buffer(var_out_index, (batch_size, SCAN_LEN_2), index_dtype)
        local_top_k = T.alloc_buffer((SCAN_LEN_2,), dtype=dtype, scope="local")
        local_top_k_index = T.alloc_buffer((SCAN_LEN_2,), dtype=index_dtype, scope="local")
        local_top_k_f32 = T.alloc_buffer((SCAN_LEN_2,), dtype="float32", scope="local")
        local_top_k_max = T.alloc_buffer((1,), dtype="float32", scope="local")
        for io in T.thread_binding(0, T.ceildiv(batch_size, TX), "blockIdx.x"):
            for ii in T.thread_binding(0, TX, "threadIdx.x"):
                with T.block("top_k"):
                    vi = T.axis.spatial(batch_size, io * TX + ii)
                    T.where(io * TX + ii < batch_size)
                    with T.block("init"):
                        local_top_k[0] = T.min_value(dtype)
                        local_top_k[1] = T.min_value(dtype)
                        local_top_k_index[0] = 0
                        local_top_k_index[1] = 1
                    for k in range(num_local_experts):
                        with T.block("update"):
                            vk = T.axis.remap("S", [k])
                            # N.B. This snippet is specialized for k = 2
                            if x[vi, vk] > local_top_k[0]:
                                local_top_k[1] = local_top_k[0]
                                local_top_k_index[1] = local_top_k_index[0]
                                local_top_k[0] = x[vi, vk]
                                local_top_k_index[0] = vk
                            elif x[vi, vk] > local_top_k[1]:
                                local_top_k[1] = x[vi, vk]
                                local_top_k_index[1] = vk
                    for j in T.unroll(SCAN_LEN_2):
                        with T.block("cast"):
                            vj = T.axis.remap("S", [j])
                            local_top_k_f32[vj] = T.cast(local_top_k[vj], "float32")
                    with T.block("max"):
                        local_top_k_max[0] = T.max(local_top_k_f32[0], local_top_k_f32[1])
                    for j in T.unroll(SCAN_LEN_2):
                        with T.block("output"):
                            vj = T.axis.remap("S", [j])
                            out[vi, vj] = T.cast(
                                T.exp(local_top_k_f32[vj] - local_top_k_max[0])
                                / (
                                    T.exp(local_top_k_f32[0] - local_top_k_max[0])
                                    + T.exp(local_top_k_f32[1] - local_top_k_max[0])
                                ),
                                dtype,
                            )
                            out_index[vi, vj] = local_top_k_index[vj]

    # specialized kernel for top 4 case
    @T.prim_func(private=True)
    def top4_softmax_norm_func(
        var_x: T.handle,
        var_out: T.handle,
        var_out_index: T.handle,
    ) -> None:
        T.func_attr({"tir.noalias": True, "tir.is_scheduled": True})
        batch_size = T.int64()
        x = T.match_buffer(var_x, (batch_size, num_local_experts), dtype)
        out = T.match_buffer(var_out, (batch_size, SCAN_LEN_4), dtype)
        out_index = T.match_buffer(var_out_index, (batch_size, SCAN_LEN_4), index_dtype)
        local_top_k = T.alloc_buffer((SCAN_LEN_4,), dtype=dtype, scope="local")
        local_top_k_index = T.alloc_buffer((SCAN_LEN_4,), dtype=index_dtype, scope="local")
        for io in T.thread_binding(0, T.ceildiv(batch_size, TX), "blockIdx.x"):
            for ii in T.thread_binding(0, TX, "threadIdx.x"):
                with T.block("top_k"):
                    vi = T.axis.spatial(batch_size, io * TX + ii)
                    T.where(io * TX + ii < batch_size)
                    with T.block("init"):
                        local_top_k[0] = T.min_value(dtype)
                        local_top_k[1] = T.min_value(dtype)
                        local_top_k[2] = T.min_value(dtype)
                        local_top_k[3] = T.min_value(dtype)
                        local_top_k_index[0] = 0
                        local_top_k_index[1] = 1
                        local_top_k_index[2] = 2
                        local_top_k_index[3] = 3
                    for k in range(num_local_experts):
                        with T.block("update"):
                            vk = T.axis.remap("S", [k])
                            # N.B. This snippet is specialized for k = 4
                            if x[vi, vk] > local_top_k[0]:
                                local_top_k[3] = local_top_k[2]
                                local_top_k_index[3] = local_top_k_index[2]
                                local_top_k[2] = local_top_k[1]
                                local_top_k_index[2] = local_top_k_index[1]
                                local_top_k[1] = local_top_k[0]
                                local_top_k_index[1] = local_top_k_index[0]
                                local_top_k[0] = x[vi, vk]
                                local_top_k_index[0] = vk
                            elif x[vi, vk] > local_top_k[1]:
                                local_top_k[3] = local_top_k[2]
                                local_top_k_index[3] = local_top_k_index[2]
                                local_top_k[2] = local_top_k[1]
                                local_top_k_index[2] = local_top_k_index[1]
                                local_top_k[1] = x[vi, vk]
                                local_top_k_index[1] = vk
                            elif x[vi, vk] > local_top_k[2]:
                                local_top_k[3] = local_top_k[2]
                                local_top_k_index[3] = local_top_k_index[2]
                                local_top_k[2] = x[vi, vk]
                                local_top_k_index[2] = vk
                            elif x[vi, vk] > local_top_k[3]:
                                local_top_k[3] = x[vi, vk]
                                local_top_k_index[3] = vk
                    for j in T.unroll(SCAN_LEN_4):
                        with T.block("output"):
                            vj = T.axis.remap("S", [j])
                            out[vi, vj] = local_top_k[vj]
                            out_index[vi, vj] = local_top_k_index[vj]

    # fast path for Mixtral
    if k == 2 and norm_topk_prob:
        return op.tensor_ir_op(
            top2_softmax_norm_func,
            "top2_softmax",
            args=[x],
            out=(
                Tensor.placeholder([batch_size, 2], dtype),
                Tensor.placeholder([batch_size, 2], index_dtype),
            ),
        )
    if k == 4 and not norm_topk_prob:
        expert_score = op.softmax(x.astype("float32"), axis=-1).astype(dtype)
        return op.tensor_ir_op(
            top4_softmax_norm_func,
            "top4_softmax",
            args=[expert_score],
            out=(
                Tensor.placeholder([batch_size, 4], dtype),
                Tensor.placeholder([batch_size, 4], index_dtype),
            ),
        )
    if norm_topk_prob:
        # Compute topk first and then softmax to avoid extra re-normalize
        expert_score, expert_indices = op.topk(
            x, k, axis=-1, ret_type="both", largest=True, dtype=index_dtype
        )
        expert_score = op.softmax(expert_score.astype("float32"), axis=-1).astype(dtype)
    else:
        expert_score = op.softmax(x.astype("float32"), axis=-1).astype(dtype)
        expert_score, expert_indices = op.topk(
            expert_score, k, axis=-1, ret_type="both", largest=True, dtype=index_dtype
        )
    return expert_score, expert_indices


def moe_cumsum(expert_indices: Tensor, num_local_experts: int) -> Tensor:
    """An operator that returns the cumsum array in MoE.

    The input `expert_indices` of shape [batch_size, experts_per_tok] indicates the indices of
    the activated experts for each instance in a batch. This operator first converts it to
    `expert_mask`, a boolean mask with shape [batch_size, num_local_experts], and then computes
    cumsum over the transpose-then-flattened array of `expert_mask`.

    A position `(e, b)` in the result `cumsum`, where `e` is the expert id and `b` is the batch id,
    indicates a shuffling plan that moves the `b`-th instance that ensures the inputs to the `e`-th
    expert is contiguous.

    Parameters
    ----------
    expert_indices : Tensor
        The topk indices with shape [batch_size, experts_per_tok], int32, where
        `experts_per_tok` is the number of activated experts.

    num_local_experts : int
        The number of totally experts.

    Returns
    -------
    cumsum: Tensor
        The cumsum result with shape [num_local_experts * batch_size], int32.

    Example
    -------
    Suppose `batch_size` is 4, `experts_per_tok` is 2, the total number of experts is 6, and
    `expert_indices` is the 2D tensor below:

        [
            [0, 1],
            [1, 2],
            [3, 4],
            [2, 5],
        ]

    , then the `expert_mask` is a tensor of shape [batch_size, num_local_experts] below:

        [
            [1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 0, 1],
        ]

    . The result cumsum of the transposed `expert_mask` is a flattened version of 2D tensor below:

        [
            [1, 1, 1, 1],
            [2, 3, 3, 3],
            [3, 4, 4, 5],
            [5, 5, 6, 6],
            [6, 6, 7, 7],
            [7, 7, 7, 8],
        ]
    """
    batch_size, experts_per_tok = expert_indices.shape
    expert_mask = (
        op.tensor_expr_op(  # pylint: disable=too-many-function-args
            lambda expert_indices: te.compute(
                (batch_size, num_local_experts),
                lambda i, j: tir.expr.Select(
                    reduce(
                        tir.Or,
                        [expert_indices[i, k] == j for k in range(experts_per_tok)],
                    ),
                    true_value=tir.const(1, "int32"),
                    false_value=tir.const(0, "int32"),
                ),
            ),
            "expert_mask",
            args=[expert_indices],
        )
        .permute_dims(1, 0)
        .reshape(batch_size * num_local_experts)
    )

    return op.cumsum(expert_mask, axis=0, exclusive=False, dtype="int32")


def get_indices(cumsum: Tensor, expert_indices: Tensor) -> Tuple[Tensor, Tensor]:
    """Returns a 1D tensor of indices that represents the shuffling plan for each instance in a
    batch, so that the inputs to each experts are contiguous and the indices for reverse permutation
    (scatter) to the original order.

    If `reverse_indices[i] = (b, j)`, it means the `b`-th instance in the batch should be moved to the
    `i`-th position in shuffling, and `j` doesn not matter only meaning `expert_indices[b, j]`
    corresponds to the expert at position `i` in the shuffling plan. We also compute
    `token_indices[i] = b` so that we can use `relax.op.take` for shuffling.

    Effectively it is equivalent to the following Python code:

    .. code-block:: python

        for b in range(batch_size):
            for j in range(experts_per_tok):
                e = expert_indices[b, j]
                reverse_indices[cumsum[e * batch_size + b] - 1] = b * experts_per_tok + j
                token_indices[cumsum[e * batch_size + b] - 1

    Parameters
    ----------
    cumsum : Tensor
        A flattened 1D tensor whose original shape is [experts_per_tok, batch_size].

    expert_indices : Tensor
        The indices of the experts with shape [batch_size, experts_per_tok].

    Returns
    -------
    reverse_indices : Tensor
        The indices for scattering with shape [batch_size * experts_per_tok].

    token_indices : Tensor
        The indices for shuffling with shape [batch_size * experts_per_tok].
    """
    TX = 1024
    batch_size, experts_per_tok = expert_indices.shape

    @T.prim_func(private=True)
    def _func(
        var_cumsum: T.handle,
        var_expert_indices: T.handle,
        var_reverse_indices: T.handle,
        var_token_indices: T.handle,
    ):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": True})
        batch_size = T.SizeVar("batch_size", "int32")
        cumsum_len = T.SizeVar("cumsum_len", "int32")  # [experts_per_tok * batch_size]
        cumsum = T.match_buffer(var_cumsum, [cumsum_len], "int32")
        expert_indices = T.match_buffer(var_expert_indices, [batch_size, experts_per_tok], "int32")
        reverse_indices = T.match_buffer(
            var_reverse_indices, [batch_size * experts_per_tok], "int32"
        )
        token_indices = T.match_buffer(var_token_indices, [batch_size * experts_per_tok], "int32")
        for bj_o in T.thread_binding(0, T.ceildiv(batch_size * experts_per_tok, TX), "blockIdx.x"):
            for bj_i in T.thread_binding(0, TX, "threadIdx.x"):
                with T.block("indices"):
                    T.reads(expert_indices[:, :], cumsum[:])
                    T.writes(reverse_indices[:], token_indices[:])
                    if bj_o * TX + bj_i < batch_size * experts_per_tok:
                        b: T.int32 = T.floordiv(bj_o * TX + bj_i, experts_per_tok)
                        j: T.int32 = T.floormod(bj_o * TX + bj_i, experts_per_tok)
                        e: T.int32 = expert_indices[b, j]
                        reverse_indices[cumsum[e * batch_size + b] - 1] = b * experts_per_tok + j
                        token_indices[cumsum[e * batch_size + b] - 1] = b

    return op.tensor_ir_op(
        _func,
        "get_indices",
        args=[cumsum, expert_indices],
        out=[Tensor.placeholder([batch_size * experts_per_tok], "int32") for _ in range(2)],
    )


def get_indptr(
    cumsum: Tensor,
    num_local_experts: int,
    batch_size: Union[int, tir.Var],
    inclusive: bool,
    out_dtype: str,
) -> Tensor:
    """Extract the `indptr` array from MoE cumsum array. The MoE cumsum array is a flattened tensor
    whose original shape is [num_local_experts, batch_size], and the `indptr` array is a 1D tensor
    of length `num_local_experts + 1`. The range `[indptr[i], indptr[i + 1])` indicates instances in
    the batch that corresponds to the `i`-th expert.

    Effectively, this operator is equivalent to the following numpy code:

    .. code-block:: python

        indptr = np.zeros(num_local_experts + 1, dtype=np.int32)
        indptr[0] = 0
        for i in range(1, num_local_experts + 1):
            indptr[i] = cumsum[i * batch_size - 1]
        return indptr

    Parameters
    ----------
    cumsum : Tensor
        The prefix sum of the sparse array with shape [batch_size * num_local_experts], int32.

    num_local_experts : int
        The number of experts.

    batch_size : int | tir.Var
        The batch size. Note that the batch size here refers to `batch_size * seq_len` in MoE,
        and we name is `batch_size` for simplicity here only because the two dimensions are fused
        in Mixtral.

    inclusive : bool
        Whether to compute inclusive or exclusive prefix sum as the indptr. If `inclusive` is False,
        the 0-th element of the `indptr` array, which always equals to 0, will be omitted.

    out_dtype : str
        The output dtype.

    Returns
    -------
    indptr : Tensor
        The `indptr` array with shape [num_local_experts + 1] if `inclusive` is True, otherwise
        [num_local_experts]. The `indptr` array is of type `out_dtype`.
    """

    out_shape = [num_local_experts if inclusive else num_local_experts + 1]

    @T.prim_func(private=True)
    def _func_exclusive(var_cumsum: T.handle, var_indptr: T.handle, batch_size: T.int64):
        T.func_attr({"tir.noalias": True})
        cumsum = T.match_buffer(var_cumsum, shape=[batch_size * num_local_experts], dtype="int32")
        indptr = T.match_buffer(var_indptr, shape=out_shape, dtype=out_dtype)
        for vi in T.serial(0, out_shape[0]):
            with T.block("indptr"):
                i = T.axis.spatial(out_shape[0], vi)
                indptr[i] = T.Select(i > 0, cumsum[i * batch_size - 1], T.int32(0))

    @T.prim_func(private=True)
    def _func_inclusive(var_cumsum: T.handle, var_indptr: T.handle, batch_size: T.int64):
        T.func_attr({"tir.noalias": True})
        cumsum = T.match_buffer(var_cumsum, shape=[batch_size * num_local_experts], dtype="int32")
        indptr = T.match_buffer(var_indptr, shape=out_shape, dtype=out_dtype)
        for vi in T.serial(0, out_shape[0]):
            with T.block("indptr"):
                i = T.axis.spatial(out_shape[0], vi)
                indptr[i] = cumsum[(i + 1) * batch_size - 1]

    assert cumsum.ndim == 1
    return op.tensor_ir_op(
        _func_inclusive if inclusive else _func_exclusive,
        "get_expert_instance_indptr",
        args=[cumsum, batch_size],  # type: ignore[list-item]
        out=Tensor.placeholder(out_shape, out_dtype),
    )


def scatter_output(x: Tensor, indices: Tensor) -> Tensor:
    """Scatter the output of MoE experts back to the original positions.

    Parameters
    ----------
    x : Tensor
        The output of MoE experts with shape [batch_size * num_experts_per_tok, hidden_size].

    indices : Tensor
        The indices of the experts with shape [batch_size * num_experts_per_tok].

    Returns
    -------
    out : Tensor
        The output of MoE experts with shape [batch_size * num_experts_per_tok, hidden_size].
    """
    dtype = x.dtype
    _, hidden_size = x.shape

    @T.prim_func(private=True)
    def _func(var_x: T.handle, var_indices: T.handle, var_out: T.handle):
        T.func_attr({"tir.noalias": True})
        indices_len = T.int64()
        x = T.match_buffer(var_x, [indices_len, hidden_size], dtype)
        indices = T.match_buffer(var_indices, [indices_len], "int32")
        out = T.match_buffer(var_out, [indices_len, hidden_size], dtype)
        for i in T.serial(0, indices_len):
            for j in T.serial(0, hidden_size):
                with T.block("scatter"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    out[indices[vi], vj] = x[vi, vj]

    return op.tensor_ir_op(
        _func,
        "scatter_output",
        args=[x, indices],
        out=Tensor.placeholder(x.shape, dtype),
    )
