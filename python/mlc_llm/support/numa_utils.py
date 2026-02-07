"""NUMA (Non-Uniform Memory Access) utilities for CPU tensor parallelism."""

import logging
import os
import subprocess
import threading
from typing import Callable, Dict, List, Optional, Set, cast

logger = logging.getLogger(__name__)


class NUMANode:  # pylint: disable=too-few-public-methods
    """Represents a NUMA node with its properties."""

    def __init__(self, node_id: int, cpus: Set[int], memory_mb: int):
        self.node_id = node_id
        self.cpus = cpus
        self.memory_mb = memory_mb

    def __repr__(self) -> str:
        return f"NUMANode(id={self.node_id}, cpus={sorted(self.cpus)}, memory={self.memory_mb}MB)"


class NUMATopology:
    """Manages NUMA topology detection and node information."""

    def __init__(self):
        self.nodes: Dict[int, NUMANode] = {}
        self.cpu_to_node: Dict[int, int] = {}
        self._detect_topology()

    def _detect_topology(self) -> None:
        """Detect NUMA topology using system utilities."""
        try:
            # Try to use numactl if available
            result = subprocess.run(
                ["numactl", "--hardware"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0:
                self._parse_numactl_output(result.stdout)
                return
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass

        # Fallback to reading /sys/devices/system/node
        self._parse_sysfs_topology()

    def _parse_numactl_output(self, output: str) -> None:
        """Parse numactl --hardware output."""
        # This is a simplified parser - real implementation would be more robust
        lines = output.split("\n")

        for line in lines:
            if line.startswith("node "):
                parts = line.split()
                if len(parts) >= 4:
                    node_id = int(parts[1])
                    cpus_str = parts[3]
                    # Parse CPU ranges like "0-7,16-23"
                    cpus: Set[int] = set()
                    for cpu_range in cpus_str.split(","):
                        if "-" in cpu_range:
                            start, end = map(int, cpu_range.split("-"))
                            cpus.update(range(start, end + 1))
                        else:
                            cpus.add(int(cpu_range))

                    # Estimate memory (simplified)
                    memory_mb = self._get_node_memory_mb(node_id)
                    self.nodes[node_id] = NUMANode(node_id, cpus, memory_mb)

                    for cpu in cpus:
                        self.cpu_to_node[cpu] = node_id

    def _parse_sysfs_topology(self) -> None:
        """Parse NUMA topology from sysfs."""
        sysfs_path = "/sys/devices/system/node"
        if not os.path.exists(sysfs_path):
            # No NUMA support detected
            self._create_single_node_fallback()
            return

        try:
            node_dirs = [
                d for d in os.listdir(sysfs_path) if d.startswith("node") and d[4:].isdigit()
            ]

            for node_dir in node_dirs:
                node_id = int(node_dir[4:])
                cpus = self._get_node_cpus(node_id)
                memory_mb = self._get_node_memory_mb(node_id)

                self.nodes[node_id] = NUMANode(node_id, cpus, memory_mb)
                for cpu in cpus:
                    self.cpu_to_node[cpu] = node_id

        except (OSError, ValueError):
            self._create_single_node_fallback()

    def _get_node_cpus(self, node_id: int) -> Set[int]:
        """Get CPUs belonging to a NUMA node."""
        try:
            with open(
                f"/sys/devices/system/node/node{node_id}/cpulist", "r", encoding="utf-8"
            ) as f:
                cpulist = f.read().strip()
                return self._parse_cpu_list(cpulist)
        except (OSError, ValueError):
            return set()

    def _get_node_memory_mb(self, node_id: int) -> int:
        """Get memory size of a NUMA node in MB."""
        try:
            with open(
                f"/sys/devices/system/node/node{node_id}/meminfo", "r", encoding="utf-8"
            ) as f:
                for line in f:
                    if line.startswith("Node ") and "MemTotal:" in line:
                        # Parse "Node 0 MemTotal: 16384 kB"
                        parts = line.split()
                        if len(parts) >= 4:
                            kb_value = int(parts[3])
                            return kb_value // 1024  # Convert to MB
        except (OSError, ValueError):
            pass
        return 0

    def _parse_cpu_list(self, cpulist: str) -> Set[int]:
        """Parse CPU list string like '0-7,16-23'."""
        cpus: Set[int] = set()
        for cpu_range in cpulist.split(","):
            cpu_range = cpu_range.strip()
            if "-" in cpu_range:
                start, end = map(int, cpu_range.split("-"))
                cpus.update(range(start, end + 1))
            else:
                cpus.add(int(cpu_range))
        return cpus

    def _create_single_node_fallback(self) -> None:
        """Create a single NUMA node fallback when NUMA is not available."""
        # Get total CPU count
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                cpu_count = sum(1 for line in f if line.startswith("processor"))
        except OSError:
            cpu_count = os.cpu_count() or 1

        # Get total memory
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            kb_value = int(parts[1])
                            memory_mb = kb_value // 1024
                            break
                else:
                    memory_mb = 0
        except OSError:
            memory_mb = 0

        cpus = set(range(cpu_count))
        self.nodes[0] = NUMANode(0, cpus, memory_mb)
        for cpu in cpus:
            self.cpu_to_node[cpu] = 0

        logger.info("NUMA not detected, using single node fallback")

    def get_node_count(self) -> int:
        """Get the number of NUMA nodes."""
        return len(self.nodes)

    def get_cpus_for_node(self, node_id: int) -> Set[int]:
        """Get CPUs belonging to a specific NUMA node."""
        return self.nodes.get(node_id, NUMANode(node_id, set(), 0)).cpus

    def get_node_for_cpu(self, cpu: int) -> int:
        """Get the NUMA node ID for a given CPU."""
        return self.cpu_to_node.get(cpu, 0)

    def get_optimal_node_distribution(self, num_workers: int) -> List[List[int]]:
        """Get optimal distribution of workers across NUMA nodes."""
        if num_workers <= 0:
            return []

        nodes = list(self.nodes.keys())
        if not nodes:
            return [[0] * num_workers]  # Fallback

        # Sort nodes by CPU count (descending)
        nodes.sort(key=lambda n: len(self.nodes[n].cpus), reverse=True)

        distribution = []
        worker_idx = 0

        while worker_idx < num_workers:
            for node_id in nodes:
                if worker_idx >= num_workers:
                    break

                node_cpus = list(self.nodes[node_id].cpus)
                if node_cpus:
                    # Assign workers across available NUMA nodes.
                    distribution.append([node_id])
                    worker_idx += 1

                    if worker_idx >= num_workers:
                        break

        return distribution

    def pin_thread_to_numa_node(self, node_id: int) -> bool:
        """Pin the current thread to a specific NUMA node."""
        try:
            sched_setaffinity = getattr(os, "sched_setaffinity", None)
            if not callable(sched_setaffinity):
                return False
            sched_setaffinity_fn = cast(Callable[[int, Set[int]], None], sched_setaffinity)
            sched_setaffinity_fn(0, self.nodes[node_id].cpus)  # pylint: disable=not-callable
            return True
        except (OSError, KeyError):
            logger.warning("Failed to pin thread to NUMA node %d", node_id)
            return False


# Global NUMA topology instance
_numa_topology: Optional[NUMATopology] = None
_numa_lock = threading.Lock()


def get_numa_topology() -> NUMATopology:
    """Get the global NUMA topology instance (singleton)."""
    # pylint: disable=global-statement
    global _numa_topology
    if _numa_topology is None:
        with _numa_lock:
            if _numa_topology is None:
                _numa_topology = NUMATopology()
    return _numa_topology


def is_numa_available() -> bool:
    """Check if NUMA is available on this system."""
    topology = get_numa_topology()
    return topology.get_node_count() > 1


def get_numa_node_count() -> int:
    """Get the number of NUMA nodes available."""
    return get_numa_topology().get_node_count()


def get_optimal_numa_distribution(num_workers: int) -> List[List[int]]:
    """Get optimal NUMA node distribution for tensor parallel workers."""
    return get_numa_topology().get_optimal_node_distribution(num_workers)


def pin_current_thread_to_numa_node(node_id: int) -> bool:
    """Pin the current thread to a specific NUMA node."""
    return get_numa_topology().pin_thread_to_numa_node(node_id)
