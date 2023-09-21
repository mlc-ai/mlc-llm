"""A weight loader for HuggingFace's PyTorch format"""
import gc
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from .param_mapping import ParameterMapping

logger = logging.getLogger(__name__)


class HFTorchLoader:
    """A loader loading HuggingFace's PyTorch format and converts them to MLC's parameters.

    Attributes
    ----------
    param_map : ParameterMapping
        The parameter mapping from MLC to HuggingFace PyTorch.

    torch_to_path : Dict[str, Path]
        A mapping from PyTorch parameter name to the path of the file containing it.

    cached_files : Dict[Path, Dict[str, np.ndarray]]
        A cache of the loaded files. The key is the path of the file, and the value is a mapping
        from parameter name to the parameter value.

    stats_load_time_sec : float
        The time spent on loading the files in seconds.

    stats_load_data_gb : float
        The amount of data loaded in GB.
    """

    param_map: ParameterMapping
    torch_to_path: Dict[str, Path]
    cached_files: Dict[Path, Dict[str, np.ndarray]]
    stats_load_time_sec: float
    stats_load_data_gb: float

    def __init__(self, config_path: Path, param_map: ParameterMapping) -> None:
        """Create a parameter loader from HuggingFace PyTorch format.

        Parameters
        ----------
        config_path : pathlib.Path
            Path to the torch indexing file, usually `pytorch_model.bin.index.json` in the repo.
        param_map : ParameterMapping
            The parameter mapping from MLC to HuggingFace PyTorch.
        """
        with config_path.open("r", encoding="utf-8") as in_file:
            torch_weight_map = json.load(in_file)["weight_map"]
        self.param_map = param_map
        self.torch_to_path = {}
        for torch_name, path_str in torch_weight_map.items():
            path = config_path.parent / path_str
            self.torch_to_path[torch_name] = path
        self.cached_files = {}
        self.stats_load_time_sec = 0.0
        self.stats_load_data_gb = 0.0

        used_torch_names = sum(param_map.name_map.values(), ())
        # Check 1. All PyTorch parameters in the weight files are used unless explicitly specified
        unused_torch_names = set(torch_weight_map) - set(used_torch_names) - param_map.unused_params
        if unused_torch_names:
            logger.warning(
                "Unused torch parameters: %s",
                ", ".join(sorted(unused_torch_names)),
            )
        # Check 2. All PyTorch parameters required are stored in the weight files
        nonexistent_torch_names = set(used_torch_names) - set(torch_weight_map)
        if nonexistent_torch_names:
            raise ValueError(
                "The following torch parameters do not exist in the weight files:\n  "
                + "\n  ".join(sorted(nonexistent_torch_names)),
            )

    def suggest_loading_order(self) -> List[str]:
        """Suggest a loading order for MLC parameters.

        Returns
        -------
        order : List[str]
            A list of MLC parameters in the order that ensures file locality.
        """
        # Step 1. Build a map from path to torch parameters
        path_to_torch: Dict[Path, List[str]] = defaultdict(list)
        for torch_name, path in self.torch_to_path.items():
            path_to_torch[path].append(torch_name)
        # Step 2. Build a map from torch parameters to MLC parameters
        torch_to_mlc = defaultdict(list)
        for mlc_name, torch_names in self.param_map.name_map.items():
            for torch_name in torch_names:
                torch_to_mlc[torch_name].append(mlc_name)
        # Step 3. Construct the ordering that ensures file locality
        order = []
        for _, torch_names in path_to_torch.items():
            for torch_name in torch_names:
                for mlc_name in torch_to_mlc[torch_name]:
                    order.append(mlc_name)
        return order

    def load_param(self, name: str) -> np.ndarray:
        """Load a MLC parameter according to its name.

        Parameters
        ----------
        name : str
            The name of the MLC parameter.

        Returns
        -------
        param : np.ndarray
            The parameter value as a numpy array. Note that if the parameter is stored in bfloat16,
            it will be converted to float32.
        """
        mlc_name = name
        torch_names = self.param_map.name_map[mlc_name]
        files_required = {self.torch_to_path[p] for p in torch_names}
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
        # Step 3. Collect all torch parameters in order
        torch_names = [self._retrieve_torch_param_from_cache(name) for name in torch_names]
        # Step 4. Apply the mapping function
        map_func = self.param_map.map_func[mlc_name]
        return map_func(*torch_names)

    def __enter__(self) -> "HFTorchLoader":
        self.stats_load_time_sec = 0.0
        self.stats_load_data_gb = 0.0
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        cached_files = list(self.cached_files.keys())
        for path in cached_files:
            self._unload_file(path)
        logger.info(
            "Time used in PyTorch loading: %.3f sec. Total %.3f GB loaded",
            self.stats_load_time_sec,
            self.stats_load_data_gb,
        )

    def _load_file(self, path: Path) -> None:
        import torch  # pylint: disable=import-outside-toplevel

        logging.info("Loading PyTorch parameters from: %s", path)

        start_time = time.time()
        result = {}
        for name, param in torch.load(path, map_location=torch.device("cpu")).items():
            param = param.detach().cpu()
            dtype = str(param.dtype)
            if dtype == "torch.bfloat16":
                param = param.float()
            param = param.numpy()
            self.stats_load_data_gb += param.nbytes / (1024**3)
            result[name] = param
            logging.debug('    Parameter: "%s", shape: %s, dtype: %s', name, param.shape, dtype)
        self.cached_files[path] = result
        self.stats_load_time_sec += time.time() - start_time

    def _unload_file(self, path: Path) -> None:
        logging.debug("Unloading PyTorch weight file: %s", path)

        start_time = time.time()
        del self.cached_files[path]
        gc.collect()
        self.stats_load_time_sec += time.time() - start_time

    def _retrieve_torch_param_from_cache(self, name: str) -> np.ndarray:
        assert name in self.torch_to_path
        path = self.torch_to_path[name]
        assert path in self.cached_files
        cache = self.cached_files[path]
        assert name in cache
        return cache[name]
