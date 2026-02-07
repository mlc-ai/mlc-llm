"""NUMA-aware weight distribution for tensor parallelism."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

from mlc_llm.serve.config import EngineConfig
from mlc_llm.support.numa_utils import (
    get_numa_topology,
    is_numa_available,
)
from mlc_llm.support.tensor_parallel import (
    NUMATensorParallelConfig,
    NUMATensorParallelManager,
)

logger = logging.getLogger(__name__)


class NUMAWeightDistributor:
    """
    Distributes model weights across NUMA nodes for optimal tensor parallelism.

    This class analyzes model weight characteristics and distributes them across
    NUMA nodes to minimize inter-node communication and maximize local memory access.
    """

    def __init__(self, engine_config: EngineConfig, model_path: str):
        self.engine_config = engine_config
        self.model_path = Path(model_path)
        self.numa_topology = get_numa_topology()
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

        # Weight distribution plan
        self.weight_distribution: Dict[str, Dict[str, Any]] = {}
        self.node_memory_usage: Dict[int, int] = {}  # Memory usage per NUMA node in MB

    def analyze_and_plan_distribution(self) -> Dict[str, Any]:
        """
        Analyze model weights and create an optimal NUMA distribution plan.

        Returns
        -------
        Dict[str, Any]
            Distribution plan with weight assignments and memory estimates
        """
        if not self.numa_manager:
            return {"strategy": "single_node", "reason": "NUMA not enabled or available"}

        # Load model metadata to understand weight structure
        model_metadata = self._load_model_metadata()
        if not model_metadata:
            return {"strategy": "single_node", "reason": "Could not load model metadata"}

        # Analyze weight characteristics
        weight_analysis = self._analyze_weights(model_metadata)

        # Create distribution plan
        distribution_plan = self._create_distribution_plan(weight_analysis)

        # Estimate memory usage per node
        self._estimate_memory_usage(distribution_plan)

        return {
            "strategy": "numa_optimized",
            "num_nodes": len(self.numa_topology.nodes),
            "weight_distribution": self.weight_distribution,
            "memory_usage": self.node_memory_usage,
            "communication_overhead": self._estimate_communication_overhead(distribution_plan),
        }

    def get_weight_placement(self, weight_name: str) -> Tuple[int, str]:
        """
        Get the optimal NUMA node and placement strategy for a weight.

        Parameters
        ----------
        weight_name : str
            Name of the weight parameter

        Returns
        -------
        Tuple[int, str]
            (numa_node_id, placement_strategy)
        """
        if weight_name in self.weight_distribution:
            placement = self.weight_distribution[weight_name]
            return placement["numa_node"], placement["strategy"]

        # Default placement
        return 0, "replicated"

    def get_numa_affinity_for_worker(self, worker_id: int) -> int:
        """Get the NUMA node affinity for a tensor parallel worker."""
        if self.numa_manager:
            return self.numa_manager.get_worker_numa_node(worker_id)
        return 0

    def _load_model_metadata(self) -> Optional[Dict[str, Any]]:
        """Load model metadata to understand weight structure."""
        try:
            # Try to load from mlc-chat-config.json
            config_path = self.model_path / "mlc-chat-config.json"
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)

                # Extract tensor parallel information
                metadata = {
                    "tensor_parallel_shards": config.get("tensor_parallel_shards", 1),
                    "model_type": config.get("model_type", "unknown"),
                    "vocab_size": config.get("vocab_size", 0),
                    "hidden_size": config.get("hidden_size", 0),
                    "num_hidden_layers": config.get("num_hidden_layers", 0),
                }
                return metadata

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("Could not load model metadata: %s", exc)

        return None

    def _analyze_weights(self, model_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze weight characteristics for distribution planning."""
        analysis: Dict[str, Any] = {
            "total_parameters": 0,
            "weight_categories": {},
            "communication_patterns": {},
            "memory_hierarchy": {},
        }

        # Estimate based on model architecture
        model_type = model_metadata.get("model_type", "unknown")
        hidden_size = model_metadata.get("hidden_size", 768)
        num_layers = model_metadata.get("num_hidden_layers", 12)
        vocab_size = model_metadata.get("vocab_size", 30000)

        if model_type in ["llama", "gpt", "opt"]:
            # Transformer-style models
            analysis["weight_categories"] = {
                "embeddings": {
                    "size_mb": (vocab_size * hidden_size * 2)
                    // (1024 * 1024),  # embeddings + lm_head
                    "access_pattern": "read_mostly",
                    "communication_frequency": "low",
                },
                "attention_weights": {
                    "size_mb": (num_layers * hidden_size * hidden_size * 12)
                    // (1024 * 1024),  # QKV + O
                    "access_pattern": "read_write",
                    "communication_frequency": "high",
                },
                "mlp_weights": {
                    "size_mb": (num_layers * hidden_size * hidden_size * 8)
                    // (1024 * 1024),  # MLP layers
                    "access_pattern": "read_write",
                    "communication_frequency": "medium",
                },
            }
        else:
            # Generic estimation
            total_params = vocab_size * hidden_size + num_layers * hidden_size * hidden_size * 16
            analysis["weight_categories"] = {
                "all_weights": {
                    "size_mb": (total_params * 2) // (1024 * 1024),  # 2 bytes per parameter (FP16)
                    "access_pattern": "read_write",
                    "communication_frequency": "medium",
                }
            }

        # Calculate total
        categories = cast(Dict[str, Dict[str, Any]], analysis["weight_categories"])
        analysis["total_parameters"] = sum(
            int(cat.get("size_mb", 0)) for cat in categories.values()
        )

        return analysis

    def _create_distribution_plan(self, weight_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create an optimal weight distribution plan across NUMA nodes."""
        plan = {"node_assignments": {}, "replication_strategy": {}, "communication_reduction": 0.0}

        available_nodes = list(self.numa_topology.nodes.keys())
        num_workers = self.engine_config.tensor_parallel_shards or 1

        # Strategy 1: Distribute attention weights across nodes for parallel computation
        if "attention_weights" in weight_analysis["weight_categories"]:
            attention_size = weight_analysis["weight_categories"]["attention_weights"]["size_mb"]
            per_node_size = attention_size // len(available_nodes)

            for i, node_id in enumerate(available_nodes):
                self.weight_distribution[f"attention_layer_{i}"] = {
                    "numa_node": node_id,
                    "strategy": "sharded",
                    "size_mb": per_node_size,
                    "workers": [i % num_workers],
                }

        # Strategy 2: Replicate embeddings across all nodes (read-mostly, low communication)
        if "embeddings" in weight_analysis["weight_categories"]:
            embedding_size = weight_analysis["weight_categories"]["embeddings"]["size_mb"]

            for node_id in available_nodes:
                self.weight_distribution[f"embeddings_node_{node_id}"] = {
                    "numa_node": node_id,
                    "strategy": "replicated",
                    "size_mb": embedding_size,
                    "workers": list(range(num_workers)),  # Available to all workers
                }

        # Strategy 3: Distribute MLP weights based on NUMA topology
        if "mlp_weights" in weight_analysis["weight_categories"]:
            mlp_size = weight_analysis["weight_categories"]["mlp_weights"]["size_mb"]
            per_node_size = mlp_size // len(available_nodes)

            for i, node_id in enumerate(available_nodes):
                self.weight_distribution[f"mlp_layer_{i}"] = {
                    "numa_node": node_id,
                    "strategy": "sharded",
                    "size_mb": per_node_size,
                    "workers": [i % num_workers],
                }

        return plan

    def _estimate_memory_usage(self, _distribution_plan: Dict[str, Any]) -> None:
        """Estimate memory usage per NUMA node."""
        for _weight_name, placement in self.weight_distribution.items():
            node_id = placement["numa_node"]
            size_mb = placement["size_mb"]

            if placement["strategy"] == "replicated":
                # Replicated weights count for each node
                self.node_memory_usage[node_id] = self.node_memory_usage.get(node_id, 0) + size_mb
            else:
                # Sharded weights are distributed
                self.node_memory_usage[node_id] = self.node_memory_usage.get(node_id, 0) + size_mb

    def _estimate_communication_overhead(self, _distribution_plan: Dict[str, Any]) -> float:
        """Estimate the communication overhead reduction achieved by NUMA distribution."""
        if not self.numa_manager:
            return 0.0

        # Simplified estimation based on weight distribution
        total_weights = len(self.weight_distribution)
        local_weights = sum(
            1 for w in self.weight_distribution.values() if w["strategy"] == "replicated"
        )

        # Calculate communication reduction as percentage of weights that are local
        if total_weights > 0:
            return (local_weights / total_weights) * 100.0

        return 0.0

    def export_distribution_config(self, output_path: str) -> None:
        """Export the weight distribution configuration to a file."""
        config = {
            "numa_tensor_parallel": self.engine_config.numa_tensor_parallel,
            "num_numa_nodes": len(self.numa_topology.nodes),
            "tensor_parallel_shards": self.engine_config.tensor_parallel_shards,
            "weight_distribution": self.weight_distribution,
            "node_memory_usage": self.node_memory_usage,
            "numa_topology": {
                node_id: {"cpus": list(node.cpus), "memory_mb": node.memory_mb}
                for node_id, node in self.numa_topology.nodes.items()
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        logger.info("Exported NUMA weight distribution config to %s", output_path)


def create_numa_weight_distributor(
    engine_config: EngineConfig, model_path: str
) -> NUMAWeightDistributor:
    """
    Create a NUMA weight distributor for optimal tensor parallel weight placement.

    Parameters
    ----------
    engine_config : EngineConfig
        Engine configuration with NUMA settings
    model_path : str
        Path to the model directory

    Returns
    -------
    NUMAWeightDistributor
        Configured NUMA weight distributor
    """
    return NUMAWeightDistributor(engine_config, model_path)
