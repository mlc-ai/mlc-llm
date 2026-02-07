"""Benchmark script for NUMA-aware tensor parallel performance."""

import argparse
import json
import logging
import time
from typing import Any, Dict, List

import numpy as np

from mlc_llm.serve.config import EngineConfig
from mlc_llm.serve.numa_communication import (
    create_numa_allocator,
    create_numa_communicator,
)
from mlc_llm.support.numa_utils import get_numa_topology, is_numa_available
from mlc_llm.support.tensor_parallel import create_numa_tensor_parallel_manager

logger = logging.getLogger(__name__)


class NUMATensorParallelBenchmark:
    """Benchmark suite for NUMA-aware tensor parallel operations."""

    def __init__(self, engine_config: EngineConfig):
        self.engine_config = engine_config
        self.numa_topology = get_numa_topology()

        self.numa_manager = None
        self.communicator = None
        self.allocator = None

        if engine_config.numa_tensor_parallel and is_numa_available():
            self.numa_manager = create_numa_tensor_parallel_manager(
                enable_numa_tp=True,
                num_workers=engine_config.tensor_parallel_shards or 1,
                inter_node_bandwidth_penalty=engine_config.numa_inter_node_penalty,
                prefer_local_memory=engine_config.numa_prefer_local_memory,
            )
            self.communicator = create_numa_communicator(self.numa_manager)
            self.allocator = create_numa_allocator(self.numa_manager)
        else:
            logger.warning("NUMA not available or not enabled, using fallback mode")

    def run_allreduce_benchmark(
        self, tensor_sizes: List[int], num_iterations: int = 100
    ) -> Dict[str, Any]:
        """Benchmark allreduce operations with different tensor sizes."""
        results: Dict[str, Any] = {
            "tensor_sizes": tensor_sizes,
            "numa_enabled": self.numa_manager is not None,
            "results": [],
        }

        world_size = self.engine_config.tensor_parallel_shards or 1
        for size in tensor_sizes:
            logger.info("Benchmarking allreduce with tensor size %d", size)
            if self.allocator is not None:
                tensor = self.allocator.allocate_tensor(
                    (size,),
                    np.dtype(np.float32),
                    0,
                    f"benchmark_{size}",
                )
            else:
                tensor = np.random.randn(size).astype(np.float32)

            start_time = time.time()
            for _ in range(num_iterations):
                if self.communicator is not None:
                    _ = self.communicator.allreduce(tensor, "sum")
                else:
                    _ = tensor * world_size
            elapsed = time.time() - start_time

            avg_time_s = elapsed / num_iterations if num_iterations > 0 else 0.0
            throughput_mbs = (size * tensor.dtype.itemsize) / max(avg_time_s, 1e-12) / (1024 * 1024)
            results["results"].append(
                {
                    "tensor_size": size,
                    "avg_time_ms": avg_time_s * 1000.0,
                    "throughput_mbs": throughput_mbs,
                    "iterations": num_iterations,
                }
            )

        return results

    def run_memory_allocation_benchmark(
        self, allocation_sizes: List[int], num_allocations: int = 1000
    ) -> Dict[str, Any]:
        """Benchmark memory allocation performance."""
        results: Dict[str, Any] = {
            "allocation_sizes": allocation_sizes,
            "numa_enabled": self.numa_manager is not None,
            "results": [],
        }

        for size in allocation_sizes:
            logger.info("Benchmarking allocation of size %d", size)
            start_time = time.time()
            for _ in range(num_allocations):
                if self.allocator is not None:
                    _ = self.allocator.allocate_tensor(
                        (size,),
                        np.dtype(np.float32),
                        0,
                        "alloc_bench",
                    )
                else:
                    _ = np.zeros((size,), dtype=np.float32)
            elapsed = time.time() - start_time
            avg_time_s = elapsed / max(num_allocations, 1)
            results["results"].append(
                {
                    "allocation_size": size,
                    "avg_time_us": avg_time_s * 1e6,
                    "total_allocated_mb": (num_allocations * size * 4) / (1024 * 1024),
                    "allocations_per_second": num_allocations / max(elapsed, 1e-12),
                }
            )
        return results

    def run_communication_pattern_benchmark(self, num_workers_list: List[int]) -> Dict[str, Any]:
        """Benchmark basic communication patterns."""
        results: Dict[str, Any] = {
            "num_workers_list": num_workers_list,
            "numa_enabled": self.numa_manager is not None,
            "results": [],
        }

        tensor_size = 1024 * 1024
        tensor = np.random.randn(tensor_size).astype(np.float32)
        patterns = ["ring", "hierarchical"]

        for num_workers in num_workers_list:
            logger.info("Benchmarking communication with %d workers", num_workers)
            pattern_results: Dict[str, Dict[str, float]] = {}
            for pattern in patterns:
                start_time = time.time()
                if self.communicator is not None:
                    _ = self.communicator.allreduce(tensor, "sum")
                elapsed = time.time() - start_time
                pattern_results[pattern] = {
                    "time_ms": elapsed * 1000.0,
                    "throughput_mbs": (tensor_size * 4) / max(elapsed, 1e-12) / (1024 * 1024),
                }
            results["results"].append({"num_workers": num_workers, "patterns": pattern_results})
        return results

    def run_numa_topology_analysis(self) -> Dict[str, Any]:
        """Analyze NUMA topology and provide optimization recommendations."""
        analysis: Dict[str, Any] = {
            "numa_available": is_numa_available(),
            "num_nodes": self.numa_topology.get_node_count(),
            "topology_info": {},
            "recommendations": [],
        }

        if analysis["numa_available"]:
            for node_id, node in self.numa_topology.nodes.items():
                analysis["topology_info"][node_id] = {
                    "cpus": sorted(node.cpus),
                    "memory_mb": node.memory_mb,
                    "cpu_count": len(node.cpus),
                }
            total_cpus = sum(len(node.cpus) for node in self.numa_topology.nodes.values())
            analysis["recommendations"] = self._generate_recommendations(total_cpus)
        else:
            analysis["recommendations"] = [
                "NUMA not available on this system",
                "Consider multi-socket CPU systems for better tensor parallel performance",
            ]
        return analysis

    def _generate_recommendations(self, total_cpus: int) -> List[str]:
        """Generate optimization recommendations based on system topology."""
        recommendations: List[str] = []
        num_nodes = self.numa_topology.get_node_count()
        if num_nodes > 1:
            recommendations.append(
                f"System has {num_nodes} NUMA nodes; NUMA-aware tensor parallelism is recommended."
            )
            recommendations.append(f"Recommended tensor_parallel_shards: {min(total_cpus, 16)}")
            total_memory_mb = sum(node.memory_mb for node in self.numa_topology.nodes.values())
            recommendations.append(
                f"Approximate memory per node: {total_memory_mb / max(num_nodes, 1):.0f} MB"
            )
        return recommendations

    def run_full_benchmark_suite(self) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        logger.info("Starting NUMA tensor parallel benchmark suite")
        results = {
            "timestamp": time.time(),
            "system_info": self.run_numa_topology_analysis(),
            "allreduce_benchmark": self.run_allreduce_benchmark([1024, 8192, 65536, 524288]),
            "memory_allocation_benchmark": self.run_memory_allocation_benchmark(
                [1024, 8192, 65536]
            ),
            "communication_pattern_benchmark": self.run_communication_pattern_benchmark([2, 4, 8]),
        }
        logger.info("Benchmark suite completed")
        return results

    def print_results(self, results: Dict[str, Any]) -> None:
        """Print benchmark results in a readable format."""
        print("\n" + "=" * 60)
        print("NUMA TENSOR PARALLEL BENCHMARK RESULTS")
        print("=" * 60)

        system_info = results["system_info"]
        print(f"\nNUMA Available: {system_info['numa_available']}")
        print(f"Number of NUMA nodes: {system_info['num_nodes']}")

        if system_info["numa_available"]:
            print("\nNUMA Node Information:")
            for node_id, info in system_info["topology_info"].items():
                print(f"  Node {node_id}: {info['cpu_count']} CPUs, {info['memory_mb']} MB")

        print("\nRecommendations:")
        for recommendation in system_info["recommendations"]:
            print(f"  - {recommendation}")

        allreduce_results = results["allreduce_benchmark"]["results"]
        if allreduce_results:
            print("\nAllreduce Performance:")
            print("  Tensor Size | Avg Time (ms) | Throughput (MB/s)")
            print("  ------------|---------------|-----------------")
            for result in allreduce_results:
                print(
                    f"  {result['tensor_size']:11d} | "
                    f"{result['avg_time_ms']:13.3f} | "
                    f"{result['throughput_mbs']:15.2f}"
                )

        mem_results = results["memory_allocation_benchmark"]["results"]
        if mem_results:
            print("\nMemory Allocation Performance:")
            print("  Alloc Size | Avg Time (us) | Allocs/sec")
            print("  -----------|---------------|-----------")
            for result in mem_results:
                print(
                    f"  {result['allocation_size']:10d} | "
                    f"{result['avg_time_us']:13.3f} | "
                    f"{result['allocations_per_second']:10.2f}"
                )

        print("\n" + "=" * 60)


def main() -> None:
    """Main entry point for NUMA tensor parallel benchmarking."""
    parser = argparse.ArgumentParser(description="NUMA Tensor Parallel Benchmark")
    parser.add_argument(
        "--tensor-parallel-shards", type=int, default=4, help="Number of tensor parallel shards"
    )
    parser.add_argument(
        "--numa-inter-node-penalty",
        type=float,
        default=0.3,
        help="Inter-node bandwidth penalty factor",
    )
    parser.add_argument(
        "--enable-numa-tp",
        action="store_true",
        default=True,
        help="Enable NUMA-aware tensor parallelism",
    )
    parser.add_argument("--output-file", type=str, help="Output file for benchmark results (JSON)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    engine_config = EngineConfig(
        numa_tensor_parallel=args.enable_numa_tp,
        tensor_parallel_shards=args.tensor_parallel_shards,
        numa_inter_node_penalty=args.numa_inter_node_penalty,
        numa_prefer_local_memory=True,
    )

    benchmark = NUMATensorParallelBenchmark(engine_config)
    results = benchmark.run_full_benchmark_suite()
    benchmark.print_results(results)

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as output_file:
            json.dump(results, output_file, indent=2)
        logger.info("Results saved to %s", args.output_file)


if __name__ == "__main__":
    main()
