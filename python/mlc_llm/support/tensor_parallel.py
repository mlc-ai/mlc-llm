"""Sharding operators for tensor parallelism."""

import dataclasses
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from tvm import te, tir, topi
from tvm.relax.frontend import nn


@dataclasses.dataclass
class ShardSingleDim:
    """
    Shard a tensor by a single dimension.


    Parameters
    ----------
    name : str
        The name of the shard func

    dim : int
        The dimension to shard

    segs : Optional[List[int]]
        The length of segments along `dim`. Default to None. If specified,
        shard a tensor by its "segmented" dimension, where each segment has a different length
        and sharded evenly on each worker.

    """

    name: str
    dim: int
    segs: Optional[List[int]] = None

    def gen_tir(self, shards: int, weight: nn.Tensor) -> tir.PrimFunc:
        """Generate a TIR function that shards the weight tensor by its rows."""
        shape = weight.shape
        segs = self.segs or [shape[self.dim]]
        assert sum(segs) == shape[self.dim]
        # NOTE: we use int64 to prevent int32 overflow
        shape = [tir.IntImm("int64", v) for v in shape]
        segs = [tir.IntImm("int64", v) for v in segs]
        w = te.placeholder(
            [tir.IntImm("int64", v) for v in self._compute_in_shape(shards, weight)],
            weight.dtype,
            name="w",
        )
        ws: List[te.Tensor] = []
        offset = 0
        for idx, sub_seg in enumerate(segs):
            ws.append(
                topi.transpose(
                    topi.reshape(
                        te.compute(
                            (*shape[: self.dim], sub_seg * shards, *shape[self.dim + 1 :]),
                            lambda *idx: w[
                                idx[: self.dim]
                                + (idx[self.dim] + offset,)  # pylint: disable=cell-var-from-loop
                                + idx[self.dim + 1 :]
                            ],
                            name=f"w_{idx}",
                        ),
                        (
                            *shape[: self.dim],
                            tir.IntImm("int64", shards),
                            sub_seg,
                            *shape[self.dim + 1 :],
                        ),
                    ),
                    [self.dim, *range(self.dim), *range(self.dim + 1, len(shape) + 1)],
                )
            )
            offset += sub_seg * shards
        o = topi.concatenate(ws, axis=1 + self.dim)
        func = te.create_prim_func([w, o])
        return func

    def gen_shard_info(self, shards: int, weight: nn.Tensor) -> Dict[str, Any]:
        """Generate shard info for this sharding strategy."""
        return {
            "func_name": self.name,
            "in_shape": self._compute_in_shape(shards, weight),
            "out_shape": (shards, *weight.shape),
            "out_dtype": weight.dtype,
        }

    def _compute_in_shape(self, shards: int, weight: nn.Tensor) -> List[int]:
        """Compute the weight shape before sharding."""
        shape = weight.shape
        return [*shape[: self.dim], shape[self.dim] * shards, *shape[self.dim + 1 :]]


@contextmanager
def shard_bias(linear: nn.Linear, tensor_parallel_shards: int):
    """
    A context manager to shard the bias of a linear into `tensor_parallel_shards` shards.


    Parameters
    ----------
    linear : nn.Linear
        The linear layer whose bias would be sharded.

    tensor_parallel_shards : int
        The number of shards.
    """
    original_bias = linear.bias
    if tensor_parallel_shards > 1:
        linear.bias = linear.bias / tensor_parallel_shards
    yield
    linear.bias = original_bias
