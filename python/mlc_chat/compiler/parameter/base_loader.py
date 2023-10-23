"""Base class for parameter loaders."""
import dataclasses
import gc
import logging
import time
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

import numpy as np
from tqdm import tqdm
from tvm.runtime import NDArray
from tvm.runtime.ndarray import array as as_ndarray

from .mapping import ExternMapping

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Stats:
    """Statistics of the loading process of loaders.

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
    """

    load_time_sec: float = 0.0
    map_time_sec: float = 0.0
    quant_time_sec: float = 0.0

    current_memory_gb: float = 0.0
    total_memory_gb: float = 0.0
    max_memory_gb: float = 0.0

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


class BaseLoader:  # pylint: disable=too-few-public-methods
    """A base loader loading parameters in other format and converts them to MLC's parameters.

    Attributes
    ----------
    stats : Stats
        Statistics of the loading process.

    extern_param_map : ExternMapping
        The parameter mapping from MLC to the specific source format.

    param_to_path : Dict[str, Path]
        A mapping from parameter name to the path of the file containing it, or the path
        meaning all parameters are stored in a single file.

    cached_files : Dict[Path, Dict[str, np.ndarray]]
        A cache of the loaded files. The key is the path of the file, and the value is a mapping
        from parameter name to the parameter value.

    weight_format : str
        The name of the weight format.
    """

    stats: Stats
    extern_param_map: ExternMapping
    cached_files: Dict[Path, Dict[str, np.ndarray]]
    param_to_path: Dict[str, Path]
    weight_format: str

    def load(self) -> Iterator[Tuple[str, NDArray]]:
        """Load the parameters and yield the MLC parameter and its value."""
        mlc_names = self._loading_order()
        for mlc_name in tqdm(mlc_names):
            param = self._load_mlc_param(mlc_name)
            yield mlc_name, param
        cached_files = list(self.cached_files.keys())
        for path in cached_files:
            self._unload_file(path)

        logger.info(
            "Time used: "
            "%s loading: %.3f sec; "
            "Pre-quantization mapping: %.3f sec; "
            "Quantization: %.3f sec",
            self.weight_format,
            self.stats.load_time_sec,
            self.stats.map_time_sec,
            self.stats.quant_time_sec,
        )
        logger.info(
            "Memory usage: Total size loaded from disk: %.3f GB; Peak memory usage: %.3f GB",
            self.stats.total_memory_gb,
            self.stats.max_memory_gb,
        )

    def _load_mlc_param(self, mlc_name: str) -> np.ndarray:
        param_names = self.extern_param_map.param_map[mlc_name]
        files_required = {self.param_to_path[p] for p in param_names}
        files_existing = set(self.cached_files.keys())
        files_to_load = files_required - files_existing
        files_to_unload = files_existing - files_required

        # Step 1. When there is some file to unloaded:
        # - If no pending file load: unloading is deferred as there is no gain in peak memory usage;
        # - Need to load files: unload immediately to save memory and make space for the new files.
        if files_to_load:
            for path in files_to_unload:
                self._unload_file(path)
        # Step 2. Load all the files needed
        for path in files_to_load:
            self._load_file(path)
        # Step 3. Collect all source parameters in order
        source_params = [self.cached_files[self.param_to_path[i]][i] for i in param_names]
        # Step 4. Apply the mapping function
        with self.stats.timer("map_time_sec"):
            param = self.extern_param_map.map_func[mlc_name](*source_params)
        logger.info('  Parameter: "%s", shape: %s, dtype: %s', mlc_name, param.shape, param.dtype)
        param = as_ndarray(param)
        return param

    def _load_file(self, path: Path) -> None:
        logger.info("Loading %s parameters from: %s", self.weight_format, path)
        with self.stats.timer("load_time_sec"):
            result = {}
            for name, param in self._load_shard(path):
                result[name] = param
                self.stats.mem_add(param.nbytes)
            self.cached_files[path] = result

    def _unload_file(self, path: Path) -> None:
        logger.info("Unloading %s weight file: %s", self.weight_format, path)
        with self.stats.timer("load_time_sec"):
            for _, param in self.cached_files[path].items():
                self.stats.mem_rm(param.nbytes)
            del self.cached_files[path]
            gc.collect()

    def _loading_order(self) -> List[str]:
        # Step 1. Build a map from path to a specific-format source parameters
        path_to_param: Dict[Path, List[str]] = defaultdict(list)
        for param_name, path in self.param_to_path.items():
            path_to_param[path].append(param_name)
        # Step 2. Build a map source parameters to MLC parameters
        param_to_mlc = defaultdict(list)
        for mlc_name, param_names in self.extern_param_map.param_map.items():
            for param_name in param_names:
                param_to_mlc[param_name].append(mlc_name)
        # Step 3. Construct the ordering that ensures file locality
        order = OrderedDict()
        for _, param_names in path_to_param.items():
            for param_name in param_names:
                for mlc_name in param_to_mlc[param_name]:
                    if mlc_name not in order:
                        order[mlc_name] = 1
        return list(order.keys())

    def _check_parameter_usage(self, source_weights: Set[str]):
        used_src_param_names = set(sum(self.extern_param_map.param_map.values(), []))
        # Check 1. All parameters in the source weight files are used unless explicitly specified
        unused_src_param_names = (
            source_weights - used_src_param_names - self.extern_param_map.unused_params
        )
        if unused_src_param_names:
            logger.warning(
                "Unused %s parameters: %s",
                self.weight_format,
                ", ".join(sorted(unused_src_param_names)),
            )
        # Check 2. All source parameters required are stored in the weight files
        nonexistent_src_param_names = used_src_param_names - source_weights
        if nonexistent_src_param_names:
            raise ValueError(
                f"The following {self.weight_format} parameters do not exist in the weight files:"
                + "\n  "
                + "\n  ".join(sorted(nonexistent_src_param_names)),
            )

    def _load_shard(self, path: Path):
        raise NotImplementedError
