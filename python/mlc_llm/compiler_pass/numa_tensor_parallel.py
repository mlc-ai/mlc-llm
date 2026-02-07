"""NUMA-aware tensor parallel compilation passes for MLC LLM."""

import logging
from typing import Any, Dict, List, Optional

import tvm
from tvm.relax.dpl import Pattern
from tvm.relax.dpl.pattern import wildcard

from mlc_llm.serve.config import EngineConfig
from mlc_llm.support.numa_utils import is_numa_available
from mlc_llm.support.tensor_parallel import (
    NUMATensorParallelConfig,
    NUMATensorParallelManager,
)

logger = logging.getLogger(__name__)


class NUMATensorParallelPass:  # pylint: disable=too-few-public-methods
    """
    Compilation pass for NUMA-aware tensor parallelism.

    This pass analyzes the model and applies transformations to optimize
    tensor parallel operations for NUMA topology.
    """

    def __init__(self, engine_config: EngineConfig):
        self.engine_config = engine_config
        self.numa_manager: Optional[NUMATensorParallelManager] = None

        if engine_config.numa_tensor_parallel and is_numa_available():
            numa_config = NUMATensorParallelConfig(
                enable_numa_tp=True,
                numa_nodes=engine_config.numa_nodes,
                inter_node_bandwidth_penalty=engine_config.numa_inter_node_penalty,
                prefer_local_memory=engine_config.numa_prefer_local_memory,
            )
            self.numa_manager = NUMATensorParallelManager(
                numa_config, engine_config.tensor_parallel_shards or 1
            )

    def apply(self, mod: tvm.ir.IRModule) -> tvm.ir.IRModule:
        """
        Apply NUMA-aware tensor parallel transformations to the IR module.

        Parameters
        ----------
        mod : tvm.ir.IRModule
            Input IR module

        Returns
        -------
        tvm.ir.IRModule
            Transformed IR module with NUMA optimizations
        """
        if not self.numa_manager:
            logger.info("NUMA tensor parallel not enabled, skipping pass")
            return mod

        logger.info("Applying NUMA-aware tensor parallel transformations")

        # Apply various NUMA optimizations
        mod = self._optimize_communication_patterns(mod)
        mod = self._optimize_memory_layout(mod)
        mod = self._add_numa_aware_primitives(mod)
        mod = self._optimize_reduction_operations(mod)

        return mod

    def _optimize_communication_patterns(self, mod: tvm.ir.IRModule) -> tvm.ir.IRModule:
        """Optimize communication patterns for NUMA topology."""
        # This would analyze allreduce and other collective operations
        # and replace them with NUMA-optimized versions

        logger.debug("Optimizing communication patterns for NUMA")
        # Placeholder - in a real implementation this would transform
        # collective operations to use NUMA-aware algorithms

        return mod

    def _optimize_memory_layout(self, mod: tvm.ir.IRModule) -> tvm.ir.IRModule:
        """Optimize memory layout for NUMA-aware access patterns."""
        # This would analyze tensor access patterns and optimize
        # memory layout to minimize cross-NUMA-node access

        logger.debug("Optimizing memory layout for NUMA")
        # Placeholder - in a real implementation this would transform
        # memory allocation and access patterns

        return mod

    def _add_numa_aware_primitives(self, mod: tvm.ir.IRModule) -> tvm.ir.IRModule:
        """Add NUMA-aware primitive operations."""
        # This would add new TIR primitives that are NUMA-aware

        logger.debug("Adding NUMA-aware primitives")
        # Placeholder - in a real implementation this would add
        # new TIR functions for NUMA-optimized operations

        return mod

    def _optimize_reduction_operations(self, mod: tvm.ir.IRModule) -> tvm.ir.IRModule:
        """Optimize reduction operations for NUMA topology."""
        # This would transform reduction operations to use
        # hierarchical algorithms that respect NUMA boundaries

        logger.debug("Optimizing reduction operations for NUMA")
        # Placeholder - in a real implementation this would transform
        # reduction operations to use NUMA-aware algorithms

        return mod


class NUMACommunicationOptimizer:
    """
    Optimizer for NUMA-aware communication in tensor parallel operations.

    This class provides patterns and transformations for optimizing
    inter-worker communication based on NUMA topology.
    """

    def __init__(self, numa_manager: NUMATensorParallelManager):
        self.numa_manager = numa_manager

    def get_optimized_allreduce_pattern(self) -> Pattern:
        """
        Get an optimized pattern for allreduce operations.

        Returns
        -------
        Pattern
            Relax pattern for NUMA-optimized allreduce
        """
        # This would return a pattern that matches allreduce operations
        # and replaces them with NUMA-optimized versions

        # Placeholder implementation
        return wildcard()

    def get_optimized_allgather_pattern(self) -> Pattern:
        """
        Get an optimized pattern for allgather operations.

        Returns
        -------
        Pattern
            Relax pattern for NUMA-optimized allgather
        """
        # This would return a pattern that matches allgather operations
        # and replaces them with NUMA-optimized versions

        # Placeholder implementation
        return wildcard()

    def should_use_hierarchical_communication(
        self, _operation: str, participating_workers: List[int]
    ) -> bool:
        """
        Determine if hierarchical communication should be used.

        Parameters
        ----------
        operation : str
            Type of collective operation
        participating_workers : List[int]
            List of participating worker IDs

        Returns
        -------
        bool
            True if hierarchical communication should be used
        """
        if not self.numa_manager.config.enable_numa_tp:
            return False

        # Check if workers span multiple NUMA nodes
        nodes = set()
        for worker in participating_workers:
            nodes.add(self.numa_manager.get_worker_numa_node(worker))

        return len(nodes) > 1


class NUMAMemoryOptimizer:
    """
    Optimizer for NUMA-aware memory operations.

    This class provides optimizations for memory allocation and access
    patterns based on NUMA topology.
    """

    def __init__(self, numa_manager: NUMATensorParallelManager):
        self.numa_manager = numa_manager

    def optimize_tensor_allocation(
        self, tensor_name: str, shape: List[int], _dtype: str, worker_id: int
    ) -> Dict[str, Any]:
        """
        Optimize tensor allocation for NUMA topology.

        Parameters
        ----------
        tensor_name : str
            Name of the tensor
        shape : List[int]
            Shape of the tensor
        dtype : str
            Data type of the tensor
        worker_id : int
            Worker that will primarily use this tensor

        Returns
        -------
        Dict[str, Any]
            Optimization hints for tensor allocation
        """
        if not self.numa_manager.config.enable_numa_tp:
            return {"strategy": "default"}

        # Determine optimal NUMA node for allocation
        optimal_worker = self.numa_manager.optimize_tensor_placement(tensor_name, shape, worker_id)
        optimal_node = self.numa_manager.get_worker_numa_node(optimal_worker)

        return {
            "strategy": "numa_optimized",
            "preferred_numa_node": optimal_node,
            "worker_affinity": optimal_worker,
            "memory_locality_hint": "high" if optimal_worker == worker_id else "medium",
        }

    def optimize_weight_placement(
        self, weight_name: str, weight_info: Dict[str, Any], worker_id: int
    ) -> Dict[str, Any]:
        """
        Optimize weight placement for NUMA topology.

        Parameters
        ----------
        weight_name : str
            Name of the weight parameter
        weight_info : Dict[str, Any]
            Information about the weight (shape, dtype, etc.)
        worker_id : int
            Worker that owns this weight shard

        Returns
        -------
        Dict[str, Any]
            Optimization hints for weight placement
        """
        if not self.numa_manager.config.enable_numa_tp:
            return {"strategy": "default"}

        shape = weight_info.get("shape", [])
        optimal_worker = self.numa_manager.optimize_tensor_placement(weight_name, shape, worker_id)

        # Determine replication vs sharding strategy
        strategy = self._determine_weight_strategy(weight_name, weight_info)

        return {
            "strategy": strategy,
            "preferred_worker": optimal_worker,
            "numa_node": self.numa_manager.get_worker_numa_node(optimal_worker),
            "replication_factor": 1 if strategy == "sharded" else self.numa_manager.num_workers,
        }

    def _determine_weight_strategy(self, _weight_name: str, weight_info: Dict[str, Any]) -> str:
        """Determine the optimal strategy for weight placement."""
        # Analyze weight characteristics to determine strategy
        access_pattern = weight_info.get("access_pattern", "read_write")
        communication_frequency = weight_info.get("communication_frequency", "medium")

        if access_pattern == "read_mostly" and communication_frequency == "low":
            return "replicated"  # Embeddings, biases
        if access_pattern == "read_write" and communication_frequency == "high":
            return "sharded"  # Attention weights, MLP weights
        return "sharded"  # Default to sharded


def create_numa_tensor_parallel_pass(engine_config: EngineConfig) -> NUMATensorParallelPass:
    """
    Create a NUMA-aware tensor parallel compilation pass.

    Parameters
    ----------
    engine_config : EngineConfig
        Engine configuration with NUMA settings

    Returns
    -------
    NUMATensorParallelPass
        Configured NUMA tensor parallel pass
    """
    return NUMATensorParallelPass(engine_config)


def create_numa_communication_optimizer(
    numa_manager: NUMATensorParallelManager,
) -> NUMACommunicationOptimizer:
    """
    Create a NUMA communication optimizer.

    Parameters
    ----------
    numa_manager : NUMATensorParallelManager
        NUMA tensor parallel manager

    Returns
    -------
    NUMACommunicationOptimizer
        Configured NUMA communication optimizer
    """
    return NUMACommunicationOptimizer(numa_manager)


def create_numa_memory_optimizer(numa_manager: NUMATensorParallelManager) -> NUMAMemoryOptimizer:
    """
    Create a NUMA memory optimizer.

    Parameters
    ----------
    numa_manager : NUMATensorParallelManager
        NUMA tensor parallel manager

    Returns
    -------
    NUMAMemoryOptimizer
        Configured NUMA memory optimizer
    """
    return NUMAMemoryOptimizer(numa_manager)
