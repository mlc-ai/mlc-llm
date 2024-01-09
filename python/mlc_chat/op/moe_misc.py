"""Mixture of Experts operators"""
from functools import reduce
from typing import Tuple, Union

from tvm import te, tir
from tvm.relax.frontend.nn import Tensor, op
from tvm.script import tir as T
from tvm.target import Target
from tvm.topi.cuda.scan import inclusive_scan
from tvm.topi.cuda.sort import topk as topi_topk

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


def topk(x: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    """Top-k operator specialized for MoE usecases.

    Parameters
    ----------
    x : Tensor
        The input tensor with shape [batch_size, num_local_experts].

    k : int
        The number of top elements to be selected, which is `num_experts_per_tok` in MoE.

    Returns
    -------
    values : Tensor
        The top-k values with shape [batch_size, k].

    indices : Tensor
        The top-k indices with shape [batch_size, k].
    """
    (batch_size, num_local_experts), dtype = x.shape, x.dtype
    index_dtype = "int32"

    TX = 1024
    SCAN_LEN = 2

    # specialized kernel for top 2 case
    @T.prim_func(private=True)
    def topk_func(
        var_x: T.handle,
        var_out: T.handle,
        var_out_index: T.handle,
    ) -> None:
        T.func_attr({"tir.noalias": True, "tir.is_scheduled": True})
        batch_size = T.int64()
        x = T.match_buffer(var_x, (batch_size, num_local_experts), dtype)
        out = T.match_buffer(var_out, (batch_size, SCAN_LEN), dtype)
        out_index = T.match_buffer(var_out_index, (batch_size, SCAN_LEN), index_dtype)
        local_top_k = T.alloc_buffer((SCAN_LEN,), dtype=dtype, scope="local")
        local_top_k_index = T.alloc_buffer((SCAN_LEN,), dtype=index_dtype, scope="local")
        for io in T.thread_binding(0, T.ceildiv(batch_size, TX), "blockIdx.x"):
            for ii in T.thread_binding(0, T.min(batch_size, TX), "threadIdx.x"):
                with T.block("top_k"):
                    vi = T.axis.spatial(batch_size, io * TX + ii)
                    T.where(io * TX + ii < batch_size)
                    with T.block("init"):
                        local_top_k[0] = T.min_value(dtype)
                        local_top_k_index[0] = 0
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
                    for j in T.unroll(SCAN_LEN):
                        with T.block("output"):
                            vj = T.axis.remap("S", [j])
                            out[vi, vj] = local_top_k[vj]
                            out_index[vi, vj] = local_top_k_index[vj]

    if k == 2:
        return op.tensor_ir_op(
            topk_func,
            "top2",
            args=[x],
            out=(
                Tensor.placeholder([batch_size, 2], dtype),
                Tensor.placeholder([batch_size, 2], index_dtype),
            ),
        )
    return op.tensor_expr_op(topi_topk, "topk", args=[x, k, -1, "both", False, index_dtype])  # type: ignore[list-item]


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
    topk_mask : Tensor
        The boolean mask with shape [batch_size, num_local_experts], int32.

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
    with Target.current(allow_none=True) or Target(
        {
            "kind": "cuda",
            "max_num_threads": 1024,
            "arch": "sm_50",
        }
    ):
        return op.tensor_expr_op(inclusive_scan, "cumsum", args=[expert_mask, 0, "int32"])  # type: ignore[list-item]


def get_indices(cumsum: Tensor, expert_indices: Tensor) -> Tensor:
    """Returns a 1D tensor of indices that represents the shuffling plan for each instance in a
    batch, so that the inputs to each experts are contiguous.

    If `indices[i] = (b, j)`, it means the `b`-th instance in the batch should be moved to the
    `i`-th position in shuffling, and `j` doesn not matter only meaning `expert_indices[b, j]`
    corresponds to the expert at position `i` in the shuffling plan.

    Effectively it is equivalent to the following Python code:

    .. code-block:: python

        for b in range(batch_size):
            for j in range(experts_per_tok):
                e = expert_indices[b, j]
                indices[cumsum[e * batch_size + b] - 1] = b * experts_per_tok + j

    Parameters
    ----------
    cumsum : Tensor
        A flattened 1D tensor whose original shape is [experts_per_tok, batch_size].

    expert_indices : Tensor
        The indices of the experts with shape [batch_size, experts_per_tok].

    Returns
    -------
    indices : Tensor
        The indices of the experts with shape [batch_size * experts_per_tok].
    """
    TX = 1024
    batch_size, experts_per_tok = expert_indices.shape

    @T.prim_func(private=True)
    def _func(var_cumsum: T.handle, var_expert_indices: T.handle, var_indices: T.handle):
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": True})
        batch_size = T.SizeVar("batch_size", "int32")
        cumsum_len = T.SizeVar("cumsum_len", "int32")  # [experts_per_tok * batch_size]
        cumsum = T.match_buffer(var_cumsum, [cumsum_len], "int32")
        expert_indices = T.match_buffer(var_expert_indices, [batch_size, experts_per_tok], "int32")
        indices = T.match_buffer(var_indices, [batch_size * experts_per_tok], "int32")
        for bj_o in T.thread_binding(0, T.ceildiv(cumsum_len, TX), "blockIdx.x"):
            for bj_i in T.thread_binding(0, TX, "threadIdx.x"):
                with T.block("indices"):
                    T.reads(expert_indices[:, :], cumsum[:])
                    T.writes(indices[:])
                    if bj_o * TX + bj_i < cumsum_len:
                        b: T.int32 = T.floordiv(bj_o * TX + bj_i, experts_per_tok)
                        j: T.int32 = T.floormod(bj_o * TX + bj_i, experts_per_tok)
                        e: T.int32 = expert_indices[b, j]
                        indices[cumsum[e * batch_size + b] - 1] = b * experts_per_tok + j

    return op.tensor_ir_op(
        _func,
        "get_flattened_expert_indices",
        args=[cumsum, expert_indices],
        out=Tensor.placeholder([batch_size * experts_per_tok], "int32"),
    )


def get_indptr(cumsum: Tensor, num_local_experts: int, batch_size: Union[int, tir.Var]) -> Tensor:
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

    Returns
    -------
    indptr : Tensor
        The `indptr` array with shape [num_local_experts + 1], int32.
    """

    @T.prim_func(private=True)
    def _func(var_cumsum: T.handle, var_indptr: T.handle, batch_size: T.int32):
        T.func_attr({"tir.noalias": True})
        cumsum = T.match_buffer(var_cumsum, shape=[batch_size * num_local_experts], dtype="int32")
        indptr = T.match_buffer(var_indptr, shape=[num_local_experts + 1], dtype="int32")
        for vi in T.serial(0, num_local_experts + 1):
            with T.block("indptr"):
                i = T.axis.spatial(num_local_experts + 1, vi)
                indptr[i] = T.Select(i > 0, cumsum[i * batch_size - 1], T.int32(0))

    assert cumsum.ndim == 1
    return op.tensor_ir_op(
        _func,
        "get_expert_instance_indptr",
        args=[cumsum, batch_size],  # type: ignore[list-item]
        out=Tensor.placeholder([num_local_experts + 1], "int32"),
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

    @T.prim_func(private=True)
    def _func(var_x: T.handle, var_indices: T.handle, var_out: T.handle):
        T.func_attr({"tir.noalias": True})
        hidden_size = T.int64()
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
