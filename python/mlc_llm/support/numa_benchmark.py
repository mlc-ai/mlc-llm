"""Benchmark script for NUMA-aware tensor parallel performance."""

import time
import numpy as np
import argparse
from typing import Dict, List, Any
import logging

from mlc_llm.support.numa_utils import (
    get_numa_topology,
    is_numa_available,
    get_optimal_numa_distribution
)
from mlc_llm.support.tensor_parallel import create_numa_tensor_parallel_manager
from mlc_llm.serve.numa_communication import create_numa_communicator, create_numa_allocator
from mlc_llm.serve.config import EngineConfig

logger = logging.getLogger(__name__)


class NUMATensorParallelBenchmark:
    """Benchmark suite for NUMA-aware tensor parallel operations."""

    def __init__(self, engine_config: EngineConfig):
        self.engine_config = engine_config
        self.numa_topology = get_numa_topology()

        # Initialize components
        if engine_config.numa_tensor_parallel and is_numa_available():
            self.numa_manager = create_numa_tensor_parallel_manager(
                enable_numa_tp=True,
                num_workers=engine_config.tensor_parallel_shards or 1,
                inter_node_bandwidth_penalty=engine_config.numa_inter_node_penalty,
                prefer_local_memory=engine_config.numa_prefer_local_memory
            )
            self.communicator = create_numa_communicator(self.numa_manager)
            self.allocator = create_numa_allocator(self.numa_manager)
        else:
            logger.warning("NUMA not available or not enabled, using fallback")
            self.numa_manager = None
            self.communicator = None
            self.allocator = None

    def run_allreduce_benchmark(self, tensor_sizes: List[int], num_iterations: int = 100) -> Dict[str, Any]:
        """Benchmark allreduce operations with different tensor sizes."""
        results = {
            "tensor_sizes": tensor_sizes,
            "numa_enabled": self.numa_manager is not None,
            "results": []
        }

        for size in tensor_sizes:
            logger.info(f"Benchmarking allreduce with tensor size {size}")

            # Create test tensor
            if self.allocator:
                tensor = self.allocator.allocate_tensor((size,), np.float32, 0, f"benchmark_{size}")
            else:
                tensor = np.random.randn(size).astype(np.float32)

            # Benchmark allreduce
            start_time = time.time()
            for _ in range(num_iterations):
                if self.communicator:
                    result = self.communicator.allreduce(tensor, "sum")
                else:
                    # Fallback implementation
                    result = tensor * (self.engine_config.tensor_parallel_shards or 1)
            end_time = time.time()

            avg_time = (end_time - start_time) / num_iterations
            throughput = (size * 4) / avg_time / (1024 * 1024)  # MB/s

            result_entry = {
                "tensor_size": size,
                "avg_time_ms": avg_time * 1000,
                "throughput_mbs": throughput,
                "iterations": num_iterations
            }
            results["results"].append(result_entry)

            logger.info(".2f")

        return results

    def run_memory_allocation_benchmark(self, allocation_sizes: List[int],
                                       num_allocations: int = 1000) -> Dict[str, Any]:
        """Benchmark memory allocation performance."""
        results = {
            "allocation_sizes": allocation_sizes,
            "numa_enabled": self.numa_manager is not None,
            "results": []
        }

        for size in allocation_sizes:
            logger.info(f"Benchmarking allocation of size {size}")

            start_time = time.time()
            for _ in range(num_allocations):
                if self.allocator:
                    tensor = self.allocator.allocate_tensor((size,), np.float32, 0, "alloc_bench")
                else:
                    tensor = np.zeros((size,), dtype=np.float32)
            end_time = time.time()

            avg_time = (end_time - start_time) / num_allocations
            total_allocated = num_allocations * size * 4 / (1024 * 1024)  # MB

            result_entry = {
                "allocation_size": size,
                "avg_time_us": avg_time * 1e6,
                "total_allocated_mb": total_allocated,
                "allocations_per_second": num_allocations / (end_time - start_time)
            }
            results["results"].append(result_entry)

            logger.info(".2f")

        return results

    def run_communication_pattern_benchmark(self, num_workers_list: List[int]) -> Dict[str, Any]:
        """Benchmark different communication patterns."""
        results = {
            "num_workers_list": num_workers_list,
            "numa_enabled": self.numa_manager is not None,
            "results": []
        }

        tensor_size = 1024 * 1024  # 1M elements
        tensor = np.random.randn(tensor_size).astype(np.float32)

        for num_workers in num_workers_list:
            logger.info(f"Benchmarking communication with {num_workers} workers")

            # Test different communication patterns
            patterns = ["ring", "hierarchical"]
            pattern_results = {}

            for pattern in patterns:
                if self.communicator and self.numa_manager:
                    # Configure for this pattern
                    start_time = time.time()
                    result = self.communicator.allreduce(tensor, "sum")
                    end_time = time.time()

                    pattern_results[pattern] = {
                        "time_ms": (end_time - start_time) * 1000,
                        "throughput_mbs": (tensor_size * 4) / (end_time - start_time) / (1024 * 1024)
                    }
                else:
                    pattern_results[pattern] = {
                        "time_ms": 0.0,
                        "throughput_mbs": 0.0
                    }

            result_entry = {
                "num_workers": num_workers,
                "patterns": pattern_results
            }
            results["results"].append(result_entry)

        return results

    def run_numa_topology_analysis(self) -> Dict[str, Any]:
        """Analyze NUMA topology and provide optimization recommendations."""
        analysis = {
            "numa_available": is_numa_available(),
            "num_nodes": self.numa_topology.get_node_count(),
            "topology_info": {},
            "recommendations": []
        }

        if is_numa_available():
            # Analyze each NUMA node
            for node_id in self.numa_topology.nodes:
                node = self.numa_topology.nodes[node_id]
                analysis["topology_info"][node_id] = {
                    "cpus": sorted(list(node.cpus)),
                    "memory_mb": node.memory_mb,
                    "cpu_count": len(node.cpus)
                }

            # Generate recommendations
            total_cpus = sum(len(node.cpus) for node in self.numa_topology.nodes.values())
            analysis["recommendations"] = self._generate_recommendations(total_cpus)
        else:
            analysis["recommendations"] = [
                "NUMA not available on this system",
                "Consider using systems with multiple CPU sockets for better tensor parallel performance"
            ]

        return analysis

    def _generate_recommendations(self, total_cpus: int) -> List[str]:
        """Generate optimization recommendations based on system topology."""
        recommendations = []

        num_nodes = self.numa_topology.get_node_count()
        if num_nodes > 1:
            recommendations.append(
                f"System has {num_nodes} NUMA nodes - NUMA-aware tensor parallelism recommended"
            )

            # Recommend optimal worker distribution
            optimal_workers = min(total_cpus, 16)  # Cap at 16 for most models
            recommendations.append(
                f"Recommended tensor_parallel_shards: {optimal_workers}"
            )

            # Memory distribution advice
            total_memory = sum(node.memory_mb for node in self.numa_topology.nodes.values())
            per_node_memory = total_memory / num_nodes
            recommendations.append(
                ".0f"
            )

        return recommendations

    def run_full_benchmark_suite(self) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        logger.info("Starting NUMA tensor parallel benchmark suite")

        results = {
            "timestamp": time.time(),
            "system_info": self.run_numa_topology_analysis(),
            "allreduce_benchmark": self.run_allreduce_benchmark(
                tensor_sizes=[1024, 8192, 65536, 524288]
            ),
            "memory_allocation_benchmark": self.run_memory_allocation_benchmark(
                allocation_sizes=[1024, 8192, 65536]
            ),
            "communication_pattern_benchmark": self.run_communication_pattern_benchmark(
                num_workers_list=[2, 4, 8]
            )
        }

        logger.info("Benchmark suite completed")
        return results

    def print_results(self, results: Dict[str, Any]) -> None:
        """Print benchmark results in a readable format."""
        print("\n" + "="*60)
        print("NUMA TENSOR PARALLEL BENCHMARK RESULTS")
        print("="*60)

        # System information
        system_info = results["system_info"]
        print(f"\nNUMA Available: {system_info['numa_available']}")
        print(f"Number of NUMA nodes: {system_info['num_nodes']}")

        if system_info["numa_available"]:
            print("\nNUMA Node Information:")
            for node_id, info in system_info["topology_info"].items():
                print(f"  Node {node_id}: {info['cpu_count']} CPUs, {info['memory_mb']} MB")

        print("\nRecommendations:")
        for rec in system_info["recommendations"]:
            print(f"  • {rec}")

        # Allreduce benchmark results
        allreduce_results = results["allreduce_benchmark"]
        if allreduce_results["results"]:
            print("
Allreduce Performance:")
            print("  Tensor Size | Avg Time (ms) | Throughput (MB/s)")
            print("  ------------|---------------|-----------------")
            for result in allreduce_results["results"]:
                print("8d")

        # Memory allocation results
        mem_results = results["memory_allocation_benchmark"]
        if mem_results["results"]:
            print("
Memory Allocation Performance:")
            print("  Alloc Size | Avg Time (μs) | Allocs/sec")
            print("  -----------|---------------|-----------")
            for result in mem_results["results"]:
                print("8d")

        print("\n" + "="*60)


def main():
    """Main entry point for NUMA tensor parallel benchmarking."""
    parser = argparse.ArgumentParser(description="NUMA Tensor Parallel Benchmark")
    parser.add_argument(
        "--tensor-parallel-shards",
        type=int,
        default=4,
        help="Number of tensor parallel shards"
    )
    parser.add_argument(
        "--numa-inter-node-penalty",
        type=float,
        default=0.3,
        help="Inter-node bandwidth penalty factor"
    )
    parser.add_argument(
        "--enable-numa-tp",
        action="store_true",
        default=True,
        help="Enable NUMA-aware tensor parallelism"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for benchmark results (JSON)"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Create engine config
    engine_config = EngineConfig(
        numa_tensor_parallel=args.enable_numa_tp,
        tensor_parallel_shards=args.tensor_parallel_shards,
        numa_inter_node_penalty=args.numa_inter_node_penalty,
        numa_prefer_local_memory=True
    )

    # Run benchmark
    benchmark = NUMATensorParallelBenchmark(engine_config)
    results = benchmark.run_full_benchmark_suite()

    # Print results
    benchmark.print_results(results)

    # Save results if requested
    if args.output_file:
        import json
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
