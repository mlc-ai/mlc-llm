"""Statistics of the loading process of parameter loaders"""

import dataclasses
import time
from contextlib import contextmanager

from mlc_llm.support import logging
from mlc_llm.support.style import green

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Stats:
    """Statistics of the loading process of parameter loaders.

    Attributes
    ----------
    load_time_sec : float
        Time used in loading the parameters.

    map_time_sec : float
        Time used in applying the mapping function, i.e. `ExternMapping.map_func`.

    quant_time_sec : float
        Time used in quantizing the parameters, i.e. `QuantizeMapping.quant_func`.

    current_memory_gb : float
        The current RAM usage in GB.

    total_memory_gb : float
        The total size data loaded from disk in GB.

    max_memory_gb : float
        The maximum RAM usage in GB.

    total_param_num: int
        Total number of parameters (original non-MLC model weights), excluding unused params.
    """

    load_time_sec: float = 0.0
    map_time_sec: float = 0.0
    quant_time_sec: float = 0.0

    current_memory_gb: float = 0.0
    total_memory_gb: float = 0.0
    max_memory_gb: float = 0.0

    total_param_num: int = 0

    def timer(self, attr):
        """A context manager to time the scope and add the time to the attribute."""

        @contextmanager
        def timed_scope():
            start_time = time.time()
            yield
            elapsed_time = time.time() - start_time
            setattr(self, attr, getattr(self, attr) + elapsed_time)

        return timed_scope()

    def mem_add(self, nbytes: int):
        """Add the memory usage by the given number of bytes."""
        mem_gb = float(nbytes) / float(1024**3)
        self.current_memory_gb += mem_gb
        self.total_memory_gb += mem_gb
        self.max_memory_gb = max(self.max_memory_gb, self.current_memory_gb)

    def mem_rm(self, nbytes: int):
        """Remove the memory usage by the given number of bytes."""
        mem_gb = float(nbytes) / float(1024**3)
        self.current_memory_gb -= mem_gb

    def log_time_info(self, weight_format: str):
        """Log the time used in loading, pre-quantization and quantization."""
        logger.info(
            "%s: "
            "%s loading: %.3f sec; "
            "Pre-quantization mapping: %.3f sec; "
            "Quantization: %.3f sec",
            green("Time usage"),
            weight_format,
            self.load_time_sec,
            self.map_time_sec,
            self.quant_time_sec,
        )

    def log_mem_usage(self):
        """Log the Memory usage information."""
        logger.info(
            "%s: Peak RAM: %.3f GB. Total bytes loaded from disk: %.3f GB",
            green("RAM usage"),
            self.max_memory_gb,
            self.total_memory_gb,
        )
