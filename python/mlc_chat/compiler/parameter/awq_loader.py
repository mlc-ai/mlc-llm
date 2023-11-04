"""A weight loader for Activation-aware Weight Quantization(AWQ) format"""
import gc
import json
import logging
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
from tvm.runtime import NDArray
from tvm.runtime.ndarray import array as as_ndarray

from .mapping import ExternMapping
from .stats import Stats
from .utils import check_parameter_usage, load_safetensor_shard, load_torch_shard

logger = logging.getLogger(__name__)


class AWQLoader:  # pylint: disable=too-few-public-methods
    """A loader loading Activation-aware Weight Quantization(AWQ) format
    and converts them to MLC's parameters.

    Attributes
    ----------
    stats : Stats
        Statistics of the loading process.
    extern_param_map : ExternMapping
        The parameter mapping from MLC to AWQ format.
    param_to_path : Dict[str, Path]
        A mapping from AWQ parameter name to the path of the file containing it, or the path
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
    weight_format: str = "PyTorch"

    def __init__(
        self,
        path: Path,
        extern_param_map: ExternMapping,
    ) -> None:
        """Create a parameter loader from Activation-aware Weight Quantization(AWQ) format.
        Now support load quantized weights generated from official
        repo (https://github.com/mit-han-lab/llm-awq) and TheBloke/Llama-2-7B-AWQ
        repo (https://huggingface.co/TheBloke/Llama-2-7B-AWQ).

        Parameters
        ----------
        path : pathlib.Path
            Path to a Pytorch `pt` file, or a JSON indexing file, or a safetensor file.
            1) For Pytorch `pt` file, it is usually generated from the official repo,
            like `llama-2-7b-w4-g128-awq.pt`. 2) For JSON indexing file, it is usually
            `model.safetensors.index.json` in the huggingface repo, which contains a `weight_map`
            that maps each PyTorch parameter to the file containing the weight. 3) For SafeTensor
            file, it is usually `model.safetensors` in the repo,
            which contains all the parameters.
        extern_param_map : ExternMapping
            Maps an MLC parameter to a list of PyTorch parameters.
        """
        assert path.is_file()
        self.stats = Stats()
        self.extern_param_map = extern_param_map
        self.cached_files = {}
        self.param_to_path = {}
        if path.suffix in (".pt", ".safetensors"):
            self._load_file(path)
            for name in self.cached_files[path].keys():
                self.param_to_path[name] = path
        elif path.suffix == ".json":
            with path.open("r", encoding="utf-8") as in_file:
                weight_map = json.load(in_file)["weight_map"]
            for param_name, path_str in weight_map.items():
                self._load_file(path.parent / path_str)
                self.param_to_path[param_name] = path.parent / path_str
        else:
            raise FileNotFoundError(f"Unknown file suffix: {path}")
        check_parameter_usage(extern_param_map, set(self.param_to_path.keys()))

    def load(self) -> Iterator[Tuple[str, NDArray]]:
        """Load the AWQ parameters and yield the MLC parameter and its value."""
        for mlc_name, awq_names in self.extern_param_map.param_map.items():
            awq_params = [
                self.cached_files[self.param_to_path[awq_name]][awq_name] for awq_name in awq_names
            ]
            # Apply the mapping function
            with self.stats.timer("map_time_sec"):
                param = self.extern_param_map.map_func[mlc_name](*awq_params)
            logger.info(
                '  Parameter: "%s", shape: %s, dtype: %s',
                mlc_name,
                param.shape,
                param.dtype,
            )
            param = as_ndarray(param)
            yield mlc_name, param
        cached_files = list(self.cached_files.keys())
        for path in cached_files:
            self._unload_file(path)
        self.stats.log_time_info("HF")
        self.stats.log_mem_usage()

    def _load_file(self, path: Path):
        if path in self.cached_files:
            return
        logger.info("Loading AWQ parameters from: %s", path)
        load_func = load_safetensor_shard if path.suffix == ".safetensors" else load_torch_shard
        with self.stats.timer("load_time_sec"):
            result = {}
            for name, param in load_func(path):
                result[name] = param
                self.stats.mem_add(param.nbytes)
            self.cached_files[path] = result

    def _unload_file(self, path: Path) -> None:
        logger.info("Unloading AWQ weight file: %s", path)
        with self.stats.timer("load_time_sec"):
            for _, param in self.cached_files[path].items():
                self.stats.mem_rm(param.nbytes)
            del self.cached_files[path]
            gc.collect()


__all__ = ["AWQLoader"]
