"""Sharding operators for tensor parallelism."""
# pylint: disable=invalid-name
import dataclasses
from typing import Any, Dict, List

from tvm import te, tir, topi
from tvm.relax.frontend import nn


@dataclasses.dataclass
class Row:
    """Shard a 2D tensor by its rows."""

    name: str
    row: int
    col: int

    def gen_tir(self, shards: int, weight: nn.Tensor) -> tir.PrimFunc:
        """Generate a TIR function that shards the weight tensor by its rows."""
        assert weight.shape == [self.row, self.col]
        w = te.placeholder([self.row * shards, self.col], weight.dtype, name="w")
        o = topi.reshape(w, (shards, self.row, self.col))
        func = te.create_prim_func([w, o])
        return func

    def gen_shard_info(self, shards: int, weight: nn.Tensor) -> Dict[str, Any]:
        """Generate shard info for this sharding strategy."""
        assert weight.shape == [self.row, self.col]
        return {
            "func": self.name,
            "out_shape": (shards, self.row, self.col),
            "out_dtype": weight.dtype,
        }


@dataclasses.dataclass
class Col:
    """Shard a 2D tensor by its columns."""

    name: str
    row: int
    col: int

    def gen_tir(self, shards: int, weight: nn.Tensor) -> tir.PrimFunc:
        """Generate a TIR function that shards the weight tensor by its columns."""
        assert weight.shape == [self.row, self.col]
        w = te.placeholder([self.row, self.col * shards], weight.dtype, name="w")
        o = topi.transpose(topi.reshape(w, (self.row, shards, self.col)), (1, 0, 2))
        func = te.create_prim_func([w, o])
        return func

    def gen_shard_info(self, shards: int, weight: nn.Tensor) -> Dict[str, Any]:
        """Generate shard info for this sharding strategy."""
        assert weight.shape == [self.row, self.col]
        return {
            "func_name": self.name,
            "out_shape": (shards, self.row, self.col),
            "out_dtype": weight.dtype,
        }


@dataclasses.dataclass
class RowSeg:
    """Shard a 2D tensor by its "segmented" rows, where each segment has a different number of rows
    and sharded evenly on each worker.


    => Step #1:

    [#shards, rows_1 // g, g, col]
    [#shards, rows_2 // g, g, col]
    ...
    [#shards, rows_n // g, g, col]

    => Step #2:

    [#shards, sum(rows) // g, g, col]

    => Step #3:

    [#shards, sum(rows), col]

    """

    name: str
    rows: List[int]
    col: int
    groups: int

    @property
    def row(self) -> int:
        """Number of rows in total"""
        return sum(self.rows)

    def gen_tir(self, shards: int, weight: nn.Tensor) -> tir.PrimFunc:
        """Generate a TIR function that shards the weight tensor by its row segments."""
        assert weight.shape == [self.row, self.col]
        w = te.placeholder([self.row * shards, self.col], weight.dtype, name="w")
        ws: List[te.Tensor] = []
        offset = 0
        for idx, sub_row in enumerate(self.rows):
            assert sub_row % self.groups == 0
            ws.append(
                topi.reshape(
                    te.compute(
                        (shards * sub_row, self.col),
                        lambda i, j: w[i + offset, j],  # pylint: disable=cell-var-from-loop
                        name=f"w_{idx}",
                    ),
                    (shards, sub_row // self.groups, self.groups, self.col),
                )
            )
            offset += sub_row * shards
        o = topi.reshape(topi.concatenate(ws, axis=1), (shards, self.row, self.col))
        func = te.create_prim_func([w, o])
        return func

    def gen_shard_info(self, shards: int, weight: nn.Tensor) -> Dict[str, Any]:
        """Generate shard info for this sharding strategy."""
        assert weight.shape == [self.row, self.col]
        return {
            "func_name": self.name,
            "out_shape": (shards, self.row, self.col),
            "out_dtype": weight.dtype,
        }
