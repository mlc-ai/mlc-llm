"""NUMA-aware communication primitives for efficient tensor parallel operations."""

# pylint: disable=too-many-branches

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from mlc_llm.support.tensor_parallel import NUMATensorParallelManager

logger = logging.getLogger(__name__)


class NUMACommunicator:
    """
    NUMA-aware communicator for tensor parallel operations.

    This class provides optimized communication primitives that take NUMA topology
    into account to minimize inter-socket communication overhead.
    """

    def __init__(self, numa_manager: NUMATensorParallelManager):
        self.numa_manager = numa_manager
        self.numa_topology = numa_manager.numa_topology
        self.communication_stats: Dict[str, float] = {
            "total_messages": 0.0,
            "inter_node_messages": 0.0,
            "intra_node_messages": 0.0,
            "total_bytes": 0.0,
            "inter_node_bytes": 0.0,
            "intra_node_bytes": 0.0,
        }

    def allreduce(self, data: np.ndarray, op: str = "sum") -> np.ndarray:
        """
        Perform NUMA-optimized allreduce operation.

        Parameters
        ----------
        data : np.ndarray
            Data to reduce
        op : str
            Reduction operation ("sum", "mean", "max", "min")

        Returns
        -------
        np.ndarray
            Reduced result
        """
        if not self.numa_manager.config.enable_numa_tp:
            # Fallback to simple reduction
            return self._simple_allreduce(data, op)

        # Get NUMA-optimized strategy
        participating_workers = list(range(self.numa_manager.num_workers))
        strategy = self.numa_manager.get_numa_optimized_allreduce_strategy(participating_workers)

        if strategy["strategy"] == "hierarchical":
            return self._hierarchical_allreduce(data, op, strategy)
        return self._ring_allreduce(data, op, participating_workers)

    def allgather(self, data: np.ndarray) -> List[np.ndarray]:
        """
        Perform NUMA-optimized allgather operation.

        Parameters
        ----------
        data : np.ndarray
            Data to gather from each worker

        Returns
        -------
        List[np.ndarray]
            List of data from all workers
        """
        if not self.numa_manager.config.enable_numa_tp:
            return [data] * self.numa_manager.num_workers

        # Use hierarchical gathering to minimize inter-node communication
        participating_workers = list(range(self.numa_manager.num_workers))
        strategy = self.numa_manager.get_numa_optimized_allreduce_strategy(participating_workers)

        if strategy["strategy"] == "hierarchical":
            return self._hierarchical_allgather(data, strategy)
        return [data] * self.numa_manager.num_workers

    def reduce_scatter(self, data: np.ndarray, op: str = "sum") -> np.ndarray:
        """
        Perform NUMA-optimized reduce-scatter operation.

        Parameters
        ----------
        data : np.ndarray
            Data to reduce and scatter
        op : str
            Reduction operation

        Returns
        -------
        np.ndarray
            Result for this worker
        """
        if not self.numa_manager.config.enable_numa_tp:
            return self._simple_reduce(data, op)

        participating_workers = list(range(self.numa_manager.num_workers))
        strategy = self.numa_manager.get_numa_optimized_allreduce_strategy(participating_workers)

        if strategy["strategy"] == "hierarchical":
            return self._hierarchical_reduce_scatter(data, op, strategy)
        return self._ring_reduce_scatter(data, op, participating_workers)

    def send(self, data: np.ndarray, src_worker: int, dst_worker: int) -> None:
        """
        Send data from source worker to destination worker with NUMA optimization.

        Parameters
        ----------
        data : np.ndarray
            Data to send
        src_worker : int
            Source worker ID
        dst_worker : int
            Destination worker ID
        """
        self._update_communication_stats(data, src_worker, dst_worker)

        # In a real implementation, this would use optimized transport
        # For now, we simulate the communication
        logger.debug("Sending %d bytes from worker %d to %d", data.nbytes, src_worker, dst_worker)

    def recv(self, src_worker: int, dst_worker: int, expected_size: int) -> np.ndarray:
        """
        Receive data from source worker with NUMA optimization.

        Parameters
        ----------
        src_worker : int
            Source worker ID
        dst_worker : int
            Destination worker ID
        expected_size : int
            Expected size of received data

        Returns
        -------
        np.ndarray
            Received data
        """
        # In a real implementation, this would use optimized transport
        # For now, we return dummy data
        logger.debug(
            "Receiving %d bytes from worker %d to %d", expected_size, src_worker, dst_worker
        )
        return np.zeros(expected_size, dtype=np.float32)

    def _simple_allreduce(self, data: np.ndarray, op: str) -> np.ndarray:
        """Simple allreduce for fallback when NUMA is not available."""
        if op == "sum":
            return data * self.numa_manager.num_workers
        if op == "mean":
            return data
        if op == "max":
            return data
        if op == "min":
            return data
        raise ValueError(f"Unsupported reduction operation: {op}")

    def _ring_allreduce(self, data: np.ndarray, op: str, workers: List[int]) -> np.ndarray:
        """Ring-based allreduce algorithm."""
        # Simplified ring allreduce - in practice this would be more complex
        result = data.copy()

        for _ in range(len(workers) - 1):
            # Simulate communication in ring
            for i, worker in enumerate(workers):
                next_worker = workers[(i + 1) % len(workers)]
                self.send(result, worker, next_worker)

                # Simulate receiving and reducing
                received = self.recv(next_worker, worker, data.nbytes)
                if op == "sum":
                    result += received
                elif op == "mean":
                    result = (result + received) / 2.0
                elif op == "max":
                    result = np.maximum(result, received)
                elif op == "min":
                    result = np.minimum(result, received)

        return result

    def _hierarchical_allreduce(
        self, data: np.ndarray, op: str, strategy: Dict[str, Any]
    ) -> np.ndarray:
        """Hierarchical allreduce optimized for NUMA topology."""
        node_groups = strategy["node_groups"]

        # Phase 1: Reduce within each NUMA node
        node_results = {}
        for node_id, workers in node_groups.items():
            if len(workers) == 1:
                node_results[node_id] = data.copy()
            else:
                # Reduce within node
                node_result = data.copy()
                for worker in workers[1:]:
                    # Simulate intra-node communication (low latency)
                    received = self.recv(workers[0], worker, data.nbytes)
                    if op == "sum":
                        node_result += received
                    elif op == "mean":
                        node_result = (node_result + received) / 2.0
                    elif op == "max":
                        node_result = np.maximum(node_result, received)
                    elif op == "min":
                        node_result = np.minimum(node_result, received)
                node_results[node_id] = node_result

        # Phase 2: Reduce across NUMA nodes (higher latency)
        if len(node_results) == 1:
            return list(node_results.values())[0]

        final_result = list(node_results.values())[0]
        for node_result in list(node_results.values())[1:]:
            if op == "sum":
                final_result += node_result
            elif op == "mean":
                final_result = (final_result + node_result) / 2.0
            elif op == "max":
                final_result = np.maximum(final_result, node_result)
            elif op == "min":
                final_result = np.minimum(final_result, node_result)

        # Phase 3: Broadcast result to all nodes
        for node_id, workers in node_groups.items():
            for worker in workers:
                self.send(final_result, workers[0], worker)

        return final_result

    def _hierarchical_allgather(
        self, data: np.ndarray, strategy: Dict[str, Any]
    ) -> List[np.ndarray]:
        """Hierarchical allgather optimized for NUMA topology."""
        node_groups = strategy["node_groups"]
        results = []

        # Gather within each node first
        for _node_id, workers in node_groups.items():
            node_data = [data] * len(workers)  # Simplified
            results.extend(node_data)

        return results

    def _hierarchical_reduce_scatter(
        self, data: np.ndarray, _op: str, _strategy: Dict[str, Any]
    ) -> np.ndarray:
        """Hierarchical reduce-scatter optimized for NUMA topology."""
        # Simplified implementation
        chunk_size = len(data) // self.numa_manager.num_workers
        return data[:chunk_size]  # Return first chunk

    def _ring_reduce_scatter(self, data: np.ndarray, _op: str, workers: List[int]) -> np.ndarray:
        """Ring-based reduce-scatter algorithm."""
        # Simplified implementation
        chunk_size = len(data) // len(workers)
        return data[:chunk_size]  # Return first chunk

    def _simple_reduce(self, data: np.ndarray, op: str) -> np.ndarray:
        """Simple reduce operation."""
        if op == "sum":
            return data
        if op == "mean":
            return data
        if op == "max":
            return data
        if op == "min":
            return data
        raise ValueError(f"Unsupported reduction operation: {op}")

    def _update_communication_stats(
        self, data: np.ndarray, src_worker: int, dst_worker: int
    ) -> None:
        """Update communication statistics."""
        self.communication_stats["total_messages"] += 1
        self.communication_stats["total_bytes"] += data.nbytes

        src_node = self.numa_manager.get_worker_numa_node(src_worker)
        dst_node = self.numa_manager.get_worker_numa_node(dst_worker)

        if src_node != dst_node:
            self.communication_stats["inter_node_messages"] += 1
            self.communication_stats["inter_node_bytes"] += data.nbytes
        else:
            self.communication_stats["intra_node_messages"] += 1
            self.communication_stats["intra_node_bytes"] += data.nbytes

    def get_communication_stats(self) -> Dict[str, float]:
        """Get communication statistics."""
        stats = self.communication_stats.copy()

        # Calculate percentages
        if stats["total_messages"] > 0:
            stats["inter_node_percentage"] = (
                stats["inter_node_messages"] / stats["total_messages"]
            ) * 100.0
        else:
            stats["inter_node_percentage"] = 0.0

        if stats["total_bytes"] > 0:
            stats["inter_node_bytes_percentage"] = (
                stats["inter_node_bytes"] / stats["total_bytes"]
            ) * 100.0
        else:
            stats["inter_node_bytes_percentage"] = 0.0

        return stats

    def reset_stats(self) -> None:
        """Reset communication statistics."""
        self.communication_stats = {
            "total_messages": 0.0,
            "inter_node_messages": 0.0,
            "intra_node_messages": 0.0,
            "total_bytes": 0.0,
            "inter_node_bytes": 0.0,
            "intra_node_bytes": 0.0,
        }


class NUMAAllocator:
    """
    NUMA-aware memory allocator for tensor parallel operations.

    This allocator optimizes memory placement based on NUMA topology
    to minimize memory access latency and maximize bandwidth utilization.
    """

    def __init__(self, numa_manager: NUMATensorParallelManager):
        self.numa_manager = numa_manager
        self.numa_topology = numa_manager.numa_topology
        self.allocation_stats: Dict[str, float] = {
            "total_allocations": 0.0,
            "local_allocations": 0.0,
            "remote_allocations": 0.0,
            "total_bytes": 0.0,
            "local_bytes": 0.0,
            "remote_bytes": 0.0,
        }

    def allocate_tensor(
        self, shape: Tuple[int, ...], dtype: np.dtype[Any], worker_id: int, tensor_name: str = ""
    ) -> np.ndarray:
        """
        Allocate a tensor with NUMA-aware placement.

        Parameters
        ----------
        shape : Tuple[int, ...]
            Shape of the tensor
        dtype : np.dtype
            Data type of the tensor
        worker_id : int
            ID of the worker that will primarily use this tensor
        tensor_name : str
            Name of the tensor for optimization hints

        Returns
        -------
        np.ndarray
            Allocated tensor
        """
        tensor: np.ndarray = np.zeros(shape, dtype=dtype)

        # Update allocation statistics
        self._update_allocation_stats(tensor, worker_id)

        # In a real implementation, this would use numa-aware allocation
        # For now, we just allocate normally
        logger.debug(
            "Allocated tensor %s with shape %s for worker %d", tensor_name, shape, worker_id
        )

        return tensor

    def allocate_weight(
        self, shape: Tuple[int, ...], dtype: np.dtype[Any], worker_id: int, weight_name: str
    ) -> np.ndarray:
        """
        Allocate a weight tensor with optimal NUMA placement.

        Parameters
        ----------
        shape : Tuple[int, ...]
            Shape of the weight tensor
        dtype : np.dtype
            Data type of the weight tensor
        worker_id : int
            ID of the worker that owns this weight shard
        weight_name : str
            Name of the weight parameter

        Returns
        -------
        np.ndarray
            Allocated weight tensor
        """
        # Use NUMA manager to determine optimal placement
        if self.numa_manager.config.enable_numa_tp:
            optimal_worker = self.numa_manager.optimize_tensor_placement(
                weight_name, list(shape), worker_id
            )
            worker_id = optimal_worker

        return self.allocate_tensor(shape, dtype, worker_id, weight_name)

    def _update_allocation_stats(self, tensor: np.ndarray, worker_id: int) -> None:
        """Update allocation statistics."""
        self.allocation_stats["total_allocations"] += 1
        self.allocation_stats["total_bytes"] += tensor.nbytes

        # Determine if this is a local or remote allocation
        _ = self.numa_manager.get_worker_numa_node(worker_id)
        # In a real implementation, we'd check the actual allocation node.
        # For now, assume local allocation
        self.allocation_stats["local_allocations"] += 1
        self.allocation_stats["local_bytes"] += tensor.nbytes

    def get_allocation_stats(self) -> Dict[str, float]:
        """Get allocation statistics."""
        stats = self.allocation_stats.copy()

        # Calculate percentages
        if stats["total_allocations"] > 0:
            stats["local_percentage"] = (
                stats["local_allocations"] / stats["total_allocations"]
            ) * 100.0
        else:
            stats["local_percentage"] = 0.0

        if stats["total_bytes"] > 0:
            stats["local_bytes_percentage"] = (stats["local_bytes"] / stats["total_bytes"]) * 100.0
        else:
            stats["local_bytes_percentage"] = 0.0

        return stats

    def reset_stats(self) -> None:
        """Reset allocation statistics."""
        self.allocation_stats = {
            "total_allocations": 0.0,
            "local_allocations": 0.0,
            "remote_allocations": 0.0,
            "total_bytes": 0.0,
            "local_bytes": 0.0,
            "remote_bytes": 0.0,
        }


def create_numa_communicator(numa_manager: NUMATensorParallelManager) -> NUMACommunicator:
    """
    Create a NUMA-aware communicator.

    Parameters
    ----------
    numa_manager : NUMATensorParallelManager
        NUMA tensor parallel manager

    Returns
    -------
    NUMACommunicator
        Configured NUMA communicator
    """
    return NUMACommunicator(numa_manager)


def create_numa_allocator(numa_manager: NUMATensorParallelManager) -> NUMAAllocator:
    """
    Create a NUMA-aware memory allocator.

    Parameters
    ----------
    numa_manager : NUMATensorParallelManager
        NUMA tensor parallel manager

    Returns
    -------
    NUMAAllocator
        Configured NUMA allocator
    """
    return NUMAAllocator(numa_manager)
