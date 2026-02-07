"""Sharding operators for tensor parallelism."""

import dataclasses
import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

from tvm import te, tir, topi
from tvm.relax.frontend import nn

from .numa_utils import get_numa_topology, is_numa_available

logger = logging.getLogger(__name__)


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


@dataclasses.dataclass
class NUMATensorParallelConfig:
    """
    Configuration for NUMA-aware tensor parallelism.

    Parameters
    ----------
    enable_numa_tp : bool
        Whether to enable NUMA-aware tensor parallelism.
    numa_nodes : Optional[List[int]]
        List of NUMA nodes to use. If None, will auto-detect optimal distribution.
    node_affinity : Optional[Dict[int, int]]
        Mapping from worker ID to NUMA node ID. If None, will auto-assign.
    inter_node_bandwidth_penalty : float
        Penalty factor for communication between different NUMA nodes (0.0-1.0).
    prefer_local_memory : bool
        Whether to prefer allocating memory on the local NUMA node.
    """

    enable_numa_tp: bool = False
    numa_nodes: Optional[List[int]] = None
    node_affinity: Optional[Dict[int, int]] = None
    inter_node_bandwidth_penalty: float = 0.3
    prefer_local_memory: bool = True


class NUMATensorParallelManager:
    """
    Manager for NUMA-aware tensor parallel operations.

    This class handles the coordination of tensor parallel operations across
    multiple NUMA nodes, optimizing for bandwidth utilization and memory locality.
    """

    def __init__(self, config: NUMATensorParallelConfig, num_workers: int):
        self.config = config
        self.num_workers = num_workers
        self.numa_topology = get_numa_topology()
        self.worker_to_node: Dict[int, int] = {}
        self.node_to_workers: Dict[int, List[int]] = {}
        self._communication_costs: Dict[Tuple[int, int], float] = {}

        if config.enable_numa_tp and is_numa_available():
            self._setup_numa_affinity()
            self._calculate_communication_costs()
        else:
            # Fallback to single NUMA node
            for i in range(num_workers):
                self.worker_to_node[i] = 0
                self.node_to_workers.setdefault(0, []).append(i)

    def _setup_numa_affinity(self) -> None:
        """Set up NUMA node affinity for workers."""
        if self.config.node_affinity:
            self.worker_to_node = self.config.node_affinity.copy()
        else:
            # Auto-assign workers to NUMA nodes
            if self.config.numa_nodes:
                available_nodes = self.config.numa_nodes
            else:
                available_nodes = list(self.numa_topology.nodes.keys())

            # Distribute workers across available NUMA nodes
            for worker_id in range(self.num_workers):
                node_id = available_nodes[worker_id % len(available_nodes)]
                self.worker_to_node[worker_id] = node_id
                self.node_to_workers.setdefault(node_id, []).append(worker_id)

    def _calculate_communication_costs(self) -> None:
        """Calculate communication costs between NUMA nodes."""
        for node1 in self.numa_topology.nodes:
            for node2 in self.numa_topology.nodes:
                if node1 == node2:
                    self._communication_costs[(node1, node2)] = 0.0
                else:
                    # Estimate cost based on whether nodes share memory bus
                    # This is a simplified model - real systems would need calibration
                    self._communication_costs[(node1, node2)] = (
                        self.config.inter_node_bandwidth_penalty
                    )

    def get_worker_numa_node(self, worker_id: int) -> int:
        """Get the NUMA node for a given worker."""
        return self.worker_to_node.get(worker_id, 0)

    def get_workers_on_node(self, node_id: int) -> List[int]:
        """Get all workers running on a specific NUMA node."""
        return self.node_to_workers.get(node_id, [])

    def get_communication_cost(self, worker1: int, worker2: int) -> float:
        """Get the communication cost between two workers."""
        node1 = self.get_worker_numa_node(worker1)
        node2 = self.get_worker_numa_node(worker2)
        return self._communication_costs.get((node1, node2), 0.0)

    def optimize_tensor_placement(
        self, _tensor_name: str, _tensor_shape: List[int], current_worker: int
    ) -> int:
        """
        Optimize tensor placement based on NUMA topology.

        Returns the optimal worker ID for placing the tensor to minimize
        communication costs and maximize memory locality.
        """
        if not self.config.enable_numa_tp:
            return current_worker

        current_node = self.get_worker_numa_node(current_worker)

        # If preferring local memory, try to keep tensor on current node
        if self.config.prefer_local_memory:
            local_workers = self.get_workers_on_node(current_node)
            if local_workers:
                # Choose worker with lowest load on the same node
                return min(local_workers, key=self._estimate_worker_load)

        # Otherwise, choose worker with minimal communication cost
        min_cost = float("inf")
        optimal_worker = current_worker

        for worker_id in range(self.num_workers):
            cost = self.get_communication_cost(current_worker, worker_id)
            load_penalty = self._estimate_worker_load(worker_id)

            total_cost = cost + load_penalty
            if total_cost < min_cost:
                min_cost = total_cost
                optimal_worker = worker_id

        return optimal_worker

    def _estimate_worker_load(self, _worker_id: int) -> float:
        """Estimate the current load of a worker (simplified)."""
        # This is a placeholder - real implementation would track actual worker load
        return 0.0

    def should_use_inter_node_communication(self, worker1: int, worker2: int) -> bool:
        """Determine if inter-node communication should be used."""
        if not self.config.enable_numa_tp:
            return False

        node1 = self.get_worker_numa_node(worker1)
        node2 = self.get_worker_numa_node(worker2)
        return node1 != node2

    def get_numa_optimized_allreduce_strategy(
        self, participating_workers: List[int]
    ) -> Dict[str, Any]:
        """
        Get an optimized all-reduce strategy for NUMA topology.

        Returns a strategy dictionary with communication plan optimized for NUMA.
        """
        if not self.config.enable_numa_tp:
            return {"strategy": "ring", "workers": participating_workers}

        # Group workers by NUMA node
        node_groups: Dict[int, List[int]] = {}
        for worker in participating_workers:
            node = self.get_worker_numa_node(worker)
            node_groups.setdefault(node, []).append(worker)

        # Choose strategy based on node distribution
        if len(node_groups) == 1:
            # All workers on same node - use standard ring allreduce
            return {"strategy": "ring", "workers": participating_workers}

        # Workers across multiple nodes - use hierarchical allreduce
        return {
            "strategy": "hierarchical",
            "node_groups": node_groups,
            "inter_node_penalty": self.config.inter_node_bandwidth_penalty,
        }


# pylint: disable=too-many-arguments
def create_numa_tensor_parallel_manager(
    enable_numa_tp: bool = False,
    num_workers: int = 1,
    numa_nodes: Optional[List[int]] = None,
    node_affinity: Optional[Dict[int, int]] = None,
    inter_node_bandwidth_penalty: float = 0.3,
    prefer_local_memory: bool = True,
) -> NUMATensorParallelManager:
    """
    Create a NUMA-aware tensor parallel manager.

    Parameters
    ----------
    enable_numa_tp : bool
        Whether to enable NUMA-aware tensor parallelism.
    num_workers : int
        Number of tensor parallel workers.
    numa_nodes : Optional[List[int]]
        List of NUMA nodes to use.
    node_affinity : Optional[Dict[int, int]]
        Mapping from worker ID to NUMA node ID.
    inter_node_bandwidth_penalty : float
        Penalty factor for inter-node communication.
    prefer_local_memory : bool
        Whether to prefer local memory allocation.

    Returns
    -------
    NUMATensorParallelManager
        Configured NUMA tensor parallel manager.
    """
    config = NUMATensorParallelConfig(
        enable_numa_tp=enable_numa_tp,
        numa_nodes=numa_nodes,
        node_affinity=node_affinity,
        inter_node_bandwidth_penalty=inter_node_bandwidth_penalty,
        prefer_local_memory=prefer_local_memory,
    )
    return NUMATensorParallelManager(config, num_workers)
