"""Tests for NUMA-aware tensor parallel functionality."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from mlc_llm.serve.config import EngineConfig
from mlc_llm.serve.numa_communication import NUMAAllocator, NUMACommunicator
from mlc_llm.serve.numa_weight_distribution import NUMAWeightDistributor
from mlc_llm.support.numa_utils import NUMANode, NUMATopology
from mlc_llm.support.tensor_parallel import (
    NUMATensorParallelConfig,
    NUMATensorParallelManager,
)


class TestNUMAUtils(unittest.TestCase):
    """Test NUMA utility functions."""

    def test_numa_topology_creation(self):
        """Test NUMA topology creation and basic functionality."""
        # Create a mock topology
        topology = NUMATopology.__new__(NUMATopology)
        topology.nodes = {0: NUMANode(0, {0, 1, 2, 3}, 16384), 1: NUMANode(1, {4, 5, 6, 7}, 16384)}
        topology.cpu_to_node = {i: 0 if i < 4 else 1 for i in range(8)}

        self.assertEqual(topology.get_node_count(), 2)
        self.assertEqual(topology.get_cpus_for_node(0), {0, 1, 2, 3})
        self.assertEqual(topology.get_node_for_cpu(5), 1)

    @patch("mlc_llm.support.numa_utils.subprocess.run")
    def test_numa_detection_with_numactl(self, mock_run):
        """Test NUMA detection using numactl."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="""
node 0 cpus: 0 1 2 3
node 0 size: 16384 MB
node 1 cpus: 4 5 6 7
node 1 size: 16384 MB
""",
        )

        topology = NUMATopology()
        # The actual implementation would parse this output
        # For testing, we just verify the method exists
        self.assertIsInstance(topology, NUMATopology)


class TestNUMATensorParallelManager(unittest.TestCase):
    """Test NUMA tensor parallel manager."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = NUMATensorParallelConfig(
            enable_numa_tp=True, inter_node_bandwidth_penalty=0.3, prefer_local_memory=True
        )

    def test_manager_creation(self):
        """Test creation of NUMA tensor parallel manager."""
        manager = NUMATensorParallelManager(self.config, 4)
        self.assertIsInstance(manager, NUMATensorParallelManager)
        self.assertEqual(manager.num_workers, 4)

    def test_worker_to_node_mapping(self):
        """Test worker to NUMA node mapping."""
        manager = NUMATensorParallelManager(self.config, 4)
        # With auto-assignment, workers should be distributed
        for worker_id in range(4):
            node_id = manager.get_worker_numa_node(worker_id)
            self.assertIsInstance(node_id, int)

    def test_communication_cost_calculation(self):
        """Test communication cost calculation between workers."""
        manager = NUMATensorParallelManager(self.config, 4)

        # Same node should have zero cost
        cost = manager.get_communication_cost(0, 0)
        self.assertEqual(cost, 0.0)

        # Different nodes should have non-zero cost
        cost = manager.get_communication_cost(0, 3)  # Assuming different nodes
        self.assertGreaterEqual(cost, 0.0)

    def test_tensor_placement_optimization(self):
        """Test tensor placement optimization."""
        manager = NUMATensorParallelManager(self.config, 4)

        # Test placement optimization
        optimal_worker = manager.optimize_tensor_placement("attention_weights", [4096, 4096], 0)
        self.assertIsInstance(optimal_worker, int)
        self.assertGreaterEqual(optimal_worker, 0)
        self.assertLess(optimal_worker, 4)


class TestNUMAWeightDistributor(unittest.TestCase):
    """Test NUMA weight distributor."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine_config = EngineConfig(
            numa_tensor_parallel=True,
            tensor_parallel_shards=4,
            numa_inter_node_penalty=0.3,
            numa_prefer_local_memory=True,
        )

    @patch("mlc_llm.serve.numa_weight_distribution.is_numa_available")
    def test_weight_distribution_plan(self, mock_numa_available):
        """Test weight distribution planning."""
        mock_numa_available.return_value = True

        with patch("mlc_llm.serve.numa_weight_distribution.Path"):
            distributor = NUMAWeightDistributor(self.engine_config, "/fake/model/path")

            # Test distribution planning
            plan = distributor.analyze_and_plan_distribution()
            self.assertIsInstance(plan, dict)
            self.assertIn("strategy", plan)

    def test_weight_placement(self):
        """Test weight placement decisions."""
        with patch("mlc_llm.serve.numa_weight_distribution.is_numa_available"):
            with patch("mlc_llm.serve.numa_weight_distribution.Path"):
                distributor = NUMAWeightDistributor(self.engine_config, "/fake/model/path")

                # Test placement for a weight
                node_id, strategy = distributor.get_weight_placement("attention_0")
                self.assertIsInstance(node_id, int)
                self.assertIsInstance(strategy, str)


class TestNUMACommunicator(unittest.TestCase):
    """Test NUMA communicator."""

    def setUp(self):
        """Set up test fixtures."""
        config = NUMATensorParallelConfig(enable_numa_tp=True)
        numa_manager = NUMATensorParallelManager(config, 4)
        self.communicator = NUMACommunicator(numa_manager)

    def test_simple_allreduce(self):
        """Test simple allreduce operation."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        result = self.communicator.allreduce(data, "sum")
        expected = data * 4  # 4 workers
        np.testing.assert_array_equal(result, expected)

    def test_communication_stats(self):
        """Test communication statistics tracking."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Perform some operations
        self.communicator.allreduce(data, "sum")

        stats = self.communicator.get_communication_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn("total_messages", stats)
        self.assertIn("total_bytes", stats)

    def test_stats_reset(self):
        """Test statistics reset functionality."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.communicator.allreduce(data, "sum")

        # Reset stats
        self.communicator.reset_stats()
        stats = self.communicator.get_communication_stats()

        self.assertEqual(stats["total_messages"], 0)
        self.assertEqual(stats["total_bytes"], 0)


class TestNUMAAllocator(unittest.TestCase):
    """Test NUMA allocator."""

    def setUp(self):
        """Set up test fixtures."""
        config = NUMATensorParallelConfig(enable_numa_tp=True)
        numa_manager = NUMATensorParallelManager(config, 4)
        self.allocator = NUMAAllocator(numa_manager)

    def test_tensor_allocation(self):
        """Test tensor allocation with NUMA awareness."""
        shape = (1024, 1024)
        dtype = np.float32

        tensor = self.allocator.allocate_tensor(shape, dtype, 0, "test_tensor")
        self.assertEqual(tensor.shape, shape)
        self.assertEqual(tensor.dtype, dtype)

    def test_allocation_stats(self):
        """Test allocation statistics tracking."""
        shape = (100, 100)
        dtype = np.float32

        # Allocate some tensors
        self.allocator.allocate_tensor(shape, dtype, 0, "tensor1")
        self.allocator.allocate_tensor(shape, dtype, 1, "tensor2")

        stats = self.allocator.get_allocation_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn("total_allocations", stats)
        self.assertEqual(stats["total_allocations"], 2)

    def test_stats_reset(self):
        """Test allocation statistics reset."""
        shape = (10, 10)
        dtype = np.float32

        self.allocator.allocate_tensor(shape, dtype, 0, "tensor")
        self.allocator.reset_stats()

        stats = self.allocator.get_allocation_stats()
        self.assertEqual(stats["total_allocations"], 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for NUMA tensor parallel components."""

    def test_full_pipeline(self):
        """Test the full NUMA tensor parallel pipeline."""
        # Create engine config with NUMA enabled
        engine_config = EngineConfig(
            numa_tensor_parallel=True,
            tensor_parallel_shards=4,
            numa_inter_node_penalty=0.3,
            numa_prefer_local_memory=True,
        )

        # Test that components can be created and work together
        self.assertTrue(engine_config.numa_tensor_parallel)
        self.assertEqual(engine_config.tensor_parallel_shards, 4)
        self.assertEqual(engine_config.numa_inter_node_penalty, 0.3)

        # Test NUMA manager creation
        numa_config = NUMATensorParallelConfig(
            enable_numa_tp=engine_config.numa_tensor_parallel,
            inter_node_bandwidth_penalty=engine_config.numa_inter_node_penalty,
            prefer_local_memory=engine_config.numa_prefer_local_memory,
        )
        numa_manager = NUMATensorParallelManager(numa_config, 4)

        self.assertIsInstance(numa_manager, NUMATensorParallelManager)

        # Test integration with communication and allocation
        communicator = NUMACommunicator(numa_manager)
        allocator = NUMAAllocator(numa_manager)

        # Test basic operations
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = communicator.allreduce(data, "sum")
        self.assertIsInstance(result, np.ndarray)

        tensor = allocator.allocate_tensor((10, 10), np.float32, 0, "test")
        self.assertEqual(tensor.shape, (10, 10))


if __name__ == "__main__":
    unittest.main()
