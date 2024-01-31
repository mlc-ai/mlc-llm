"""Operators for token embeddings."""
from tvm.relax.frontend.nn import Tensor, op
from tvm.script import tir as T


def inplace_embedding_take(  # pylint: disable=too-many-arguments
    embedding_table: Tensor,
    token_ids: Tensor,
    embedding_dst: Tensor,
    offset: int,
    hidden_size: int,
    dtype: str,
) -> Tensor:
    """The in-place embedding lookup op.

    Parameters
    ----------
    embedding_table : Tensor
        The model's embedding table for all token ids.

    token_ids : Tensor
        The token ids to take embeddings.

    embedding_dst : Tensor
        The destination tensor of embedding lookup, which is updated in-place.

    offset : int
        The offset to write into the destination tensor.

    hidden_size : int
        The hidden size of the embedding.

    dtype : str
        The dtype of the embedding

    Returns
    -------
    embedding_dst_updated : Tensor
        The updated embedding destination tensor.
    """

    @T.prim_func
    def inplace_take(
        var_weight: T.handle, var_pos: T.handle, var_embeddings: T.handle, offset: T.int64
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        vocab_size = T.int64()
        weight = T.match_buffer(var_weight, (vocab_size, hidden_size), dtype)
        seq_len = T.int64()
        total_seq_len = T.int64()
        pos = T.match_buffer(var_pos, (seq_len,), "int32")
        embeddings = T.match_buffer(var_embeddings, (total_seq_len, hidden_size), dtype)
        for ax0, ax1 in T.grid(seq_len, hidden_size):
            with T.block("T_take"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(weight[pos[v0], v1], pos[v0])
                T.writes(embeddings[v0, v1])
                embeddings[v0 + offset, v1] = weight[pos[v0], v1]

    return op.tensor_ir_inplace_op(
        inplace_take,
        "inplace_take",
        args=[embedding_table, token_ids, embedding_dst, offset],
        inplace_indices=2,
        out=Tensor.placeholder(embedding_dst.shape, embedding_dst.dtype),
    )
