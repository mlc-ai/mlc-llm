"""NUMA-aware CPU tensor parallel execution engine for MLC LLM."""

import asyncio
import concurrent.futures
import multiprocessing
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import logging
import os

from mlc_llm.support.numa_utils import (
    get_numa_topology,
    is_numa_available,
    get_optimal_numa_distribution,
    pin_current_thread_to_numa_node,
    NUMATopology
)
from mlc_llm.support.tensor_parallel import (
    create_numa_tensor_parallel_manager,
    NUMATensorParallelManager,
    NUMATensorParallelConfig
)
from mlc_llm.serve.config import EngineConfig

logger = logging.getLogger(__name__)


class NUMAWorker:
    """A worker process/thread running on a specific NUMA node."""

    def __init__(self, worker_id: int, numa_node: int, worker_func: Callable,
                 numa_topology: NUMATopology):
        self.worker_id = worker_id
        self.numa_node = numa_node
        self.worker_func = worker_func
        self.numa_topology = numa_topology
        self.process: Optional[multiprocessing.Process] = None
        self._input_queue: Optional[multiprocessing.Queue] = None
        self._output_queue: Optional[multiprocessing.Queue] = None
        self._shutdown_event: Optional[multiprocessing.Event] = None

    def start(self) -> None:
        """Start the worker process."""
        self._input_queue = multiprocessing.Queue()
        self._output_queue = multiprocessing.Queue()
        self._shutdown_event = multiprocessing.Event()

        self.process = multiprocessing.Process(
            target=self._worker_main,
            args=(self.worker_id, self.numa_node, self._input_queue,
                  self._output_queue, self._shutdown_event)
        )
        self.process.start()
        logger.info(f"Started NUMA worker {self.worker_id} on NUMA node {self.numa_node}")

    def stop(self) -> None:
        """Stop the worker process."""
        if self._shutdown_event:
            self._shutdown_event.set()
        if self.process and self.process.is_alive():
            self.process.join(timeout=5.0)
            if self.process.is_alive():
                self.process.terminate()
        logger.info(f"Stopped NUMA worker {self.worker_id}")

    def send_task(self, task_data: Any) -> None:
        """Send a task to the worker."""
        if self._input_queue:
            self._input_queue.put(task_data)

    def receive_result(self, timeout: float = 1.0) -> Any:
        """Receive a result from the worker."""
        if self._output_queue:
            try:
                return self._output_queue.get(timeout=timeout)
            except multiprocessing.Queue.Empty:
                return None
        return None

    def is_alive(self) -> bool:
        """Check if the worker process is alive."""
        return self.process is not None and self.process.is_alive()

    def _worker_main(self, worker_id: int, numa_node: int,
                     input_queue: multiprocessing.Queue,
                     output_queue: multiprocessing.Queue,
                     shutdown_event: multiprocessing.Event) -> None:
        """Main function for the worker process."""
        try:
            # Pin this process to the assigned NUMA node
            if not pin_current_thread_to_numa_node(numa_node):
                logger.warning(f"Failed to pin worker {worker_id} to NUMA node {numa_node}")

            # Set process name for debugging
            if hasattr(os, 'setproctitle'):
                os.setproctitle(f"mlc_numa_worker_{worker_id}_node_{numa_node}")

            logger.info(f"NUMA worker {worker_id} running on node {numa_node}")

            while not shutdown_event.is_set():
                try:
                    # Wait for task with timeout
                    task_data = input_queue.get(timeout=0.1)

                    # Process the task
                    result = self.worker_func(worker_id, numa_node, task_data)

                    # Send result back
                    output_queue.put(result)

                except multiprocessing.Queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in NUMA worker {worker_id}: {e}")
                    output_queue.put({"error": str(e), "worker_id": worker_id})

        except Exception as e:
            logger.error(f"Fatal error in NUMA worker {worker_id}: {e}")
        finally:
            logger.info(f"NUMA worker {worker_id} shutting down")


class NUMACPUParallelEngine:
    """
    NUMA-aware CPU tensor parallel execution engine.

    This engine distributes tensor parallel workers across NUMA nodes to optimize
    bandwidth utilization and reduce inter-socket communication overhead.
    """

    def __init__(self, engine_config: EngineConfig, worker_func: Callable):
        self.engine_config = engine_config
        self.worker_func = worker_func
        self.numa_topology = get_numa_topology()
        self.workers: List[NUMAWorker] = []
        self.numa_manager: Optional[NUMATensorParallelManager] = None

        # Initialize NUMA tensor parallel manager if enabled
        if engine_config.numa_tensor_parallel and is_numa_available():
            numa_config = NUMATensorParallelConfig(
                enable_numa_tp=True,
                numa_nodes=engine_config.numa_nodes,
                inter_node_bandwidth_penalty=engine_config.numa_inter_node_penalty,
                prefer_local_memory=engine_config.numa_prefer_local_memory
            )
            self.numa_manager = create_numa_tensor_parallel_manager(
                enable_numa_tp=True,
                num_workers=engine_config.tensor_parallel_shards or 1,
                numa_nodes=engine_config.numa_nodes,
                inter_node_bandwidth_penalty=engine_config.numa_inter_node_penalty,
                prefer_local_memory=engine_config.numa_prefer_local_memory
            )
            logger.info("NUMA tensor parallel manager initialized")
        else:
            logger.info("NUMA tensor parallel not enabled or not available")

    def start_workers(self) -> None:
        """Start all NUMA workers."""
        if not self.engine_config.numa_tensor_parallel:
            logger.warning("NUMA tensor parallel not enabled, cannot start workers")
            return

        num_workers = self.engine_config.tensor_parallel_shards or 1
        numa_nodes = self._get_numa_nodes_for_workers(num_workers)

        for worker_id in range(num_workers):
            numa_node = numa_nodes[worker_id] if worker_id < len(numa_nodes) else 0
            worker = NUMAWorker(worker_id, numa_node, self.worker_func, self.numa_topology)
            worker.start()
            self.workers.append(worker)

        logger.info(f"Started {len(self.workers)} NUMA workers across {len(set(numa_nodes))} NUMA nodes")

    def stop_workers(self) -> None:
        """Stop all NUMA workers."""
        for worker in self.workers:
            worker.stop()
        self.workers.clear()
        logger.info("All NUMA workers stopped")

    def distribute_task(self, task_data: Any, target_worker: Optional[int] = None) -> Dict[int, Any]:
        """
        Distribute a task to workers, optionally optimizing placement based on NUMA topology.

        Parameters
        ----------
        task_data : Any
            The task data to distribute
        target_worker : Optional[int]
            Specific worker to target, or None for automatic placement

        Returns
        -------
        Dict[int, Any]
            Results from workers, keyed by worker ID
        """
        if not self.workers:
            raise RuntimeError("No workers available. Call start_workers() first.")

        if target_worker is not None:
            # Send to specific worker
            self.workers[target_worker].send_task(task_data)
            result = self.workers[target_worker].receive_result()
            return {target_worker: result} if result is not None else {}

        # Automatic placement based on NUMA topology
        if self.numa_manager:
            optimal_worker = self.numa_manager.optimize_tensor_placement(
                "task", [], 0  # Simplified placement decision
            )
            self.workers[optimal_worker].send_task(task_data)
            result = self.workers[optimal_worker].receive_result()
            return {optimal_worker: result} if result is not None else {}
        else:
            # Round-robin distribution
            results = {}
            for i, worker in enumerate(self.workers):
                worker.send_task(task_data)
                result = worker.receive_result()
                if result is not None:
                    results[i] = result
            return results

    def broadcast_task(self, task_data: Any) -> Dict[int, Any]:
        """
        Broadcast a task to all workers.

        Parameters
        ----------
        task_data : Any
            The task data to broadcast

        Returns
        -------
        Dict[int, Any]
            Results from all workers, keyed by worker ID
        """
        if not self.workers:
            raise RuntimeError("No workers available. Call start_workers() first.")

        results = {}
        for i, worker in enumerate(self.workers):
            worker.send_task(task_data)

        # Collect results from all workers
        for i, worker in enumerate(self.workers):
            result = worker.receive_result(timeout=5.0)
            if result is not None:
                results[i] = result

        return results

    def get_worker_stats(self) -> Dict[str, Any]:
        """Get statistics about NUMA workers."""
        stats = {
            "num_workers": len(self.workers),
            "workers_alive": sum(1 for w in self.workers if w.is_alive()),
            "numa_nodes_used": len(set(w.numa_node for w in self.workers)),
            "numa_distribution": {}
        }

        # Count workers per NUMA node
        for worker in self.workers:
            node = worker.numa_node
            stats["numa_distribution"][node] = stats["numa_distribution"].get(node, 0) + 1

        return stats

    def _get_numa_nodes_for_workers(self, num_workers: int) -> List[int]:
        """Get NUMA node assignment for workers."""
        if self.engine_config.numa_nodes:
            # Use explicitly specified NUMA nodes
            nodes = self.engine_config.numa_nodes
        else:
            # Auto-detect optimal distribution
            nodes = list(self.numa_topology.nodes.keys())

        # Distribute workers across available nodes
        numa_assignment = []
        for i in range(num_workers):
            node_id = nodes[i % len(nodes)]
            numa_assignment.append(node_id)

        return numa_assignment

    async def execute_async_task(self, task_data: Any) -> Dict[int, Any]:
        """Execute a task asynchronously."""
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = loop.run_in_executor(executor, self.distribute_task, task_data)
            return await future

    def __enter__(self):
        """Context manager entry."""
        self.start_workers()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_workers()


def create_numa_cpu_parallel_engine(
    engine_config: EngineConfig,
    worker_func: Callable[[int, int, Any], Any]
) -> NUMACPUParallelEngine:
    """
    Create a NUMA-aware CPU parallel execution engine.

    Parameters
    ----------
    engine_config : EngineConfig
        Engine configuration with NUMA settings
    worker_func : Callable[[int, int, Any], Any]
        Worker function that takes (worker_id, numa_node, task_data) and returns result

    Returns
    -------
    NUMACPUParallelEngine
        Configured NUMA CPU parallel engine
    """
    return NUMACPUParallelEngine(engine_config, worker_func)
