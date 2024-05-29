"""A weight loader for HuggingFace's PyTorch format"""

import gc
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from tvm.runtime import Device, NDArray
from tvm.runtime.ndarray import array as as_ndarray

from mlc_llm.support import logging
from mlc_llm.support.preshard import _sharded_param_name
from mlc_llm.support.style import bold

from .mapping import ExternMapping, QuantizeMapping
from .stats import Stats
from .utils import check_parameter_usage, load_safetensor_shard, load_torch_shard

logger = logging.getLogger(__name__)


class HuggingFaceLoader:  # pylint: disable=too-few-public-methods
    """A loader loading HuggingFace's PyTorch/SafeTensor format and converts them
    to MLC's parameters.

    Attributes
    ----------
    stats : Stats
        Statistics of the loading process.

    extern_param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch/SafeTensor.

    torch_to_path : Dict[str, Path]
        A mapping from PyTorch/SafeTensor parameter name to the path of the file containing it,
        or the path meaning all parameters are stored in a single file.

    cached_files : Dict[Path, Dict[str, np.ndarray]]
        A cache of the loaded files. The key is the path of the file, and the value is a mapping
        from parameter name to the parameter value.

    quantize_param_map : Optional[QuantizeMapping]
        The quantization mapping from MLC to quantized MLC parameters.
    """

    stats: Stats
    cached_files: Dict[Path, Dict[str, np.ndarray]]
    torch_to_path: Dict[str, Path]
    extern_param_map: ExternMapping
    quantize_param_map: Optional[QuantizeMapping]

    def __init__(
        self,
        path: Path,
        extern_param_map: ExternMapping,
        quantize_param_map: Optional[QuantizeMapping] = None,
    ) -> None:
        """Create a parameter loader from HuggingFace PyTorch format.

        Parameters
        ----------
        path : pathlib.Path
            Path to either a JSON indexing file, or a PyTorch bin file.
            1) For JSON indexing file, it is usually `pytorch_model.bin.index.json`
            or `model.safetensors.index.json` in the repo, which contains a `weight_map` that
            maps each PyTorch parameter to the file containing the weight.
            2) For PyTorch bin file, it is usually `pytorch_model.bin` in the repo,
            which contains all the parameters.
            3) For safetensor file, it is usually `model.safetensors` in the repo,
            which contains all the parameters.

        extern_param_map : ExternMapping
            Maps an MLC parameter to a list of PyTorch/SafeTensor parameters.

        quantize_param_map: Optional[QuantizeMapping]
            The quantization mapping from MLC to quantized MLC parameters, default to None, which
            means no quantization.
        """
        assert path.is_file(), f"Path {path} is not a file"
        self.stats = Stats()
        self.extern_param_map = extern_param_map
        self.cached_files = {}
        self.torch_to_path = {}
        self.quantize_param_map = quantize_param_map
        if path.suffix in (".bin", ".safetensors", ".pt"):
            self._load_file(path)
            for name in self.cached_files[path].keys():
                self.torch_to_path[name] = path
        elif path.suffix == ".json":
            with path.open("r", encoding="utf-8") as in_file:
                torch_weight_map = json.load(in_file)["weight_map"]
            for torch_name, path_str in torch_weight_map.items():
                self.torch_to_path[torch_name] = path.parent / path_str
        else:
            raise FileNotFoundError(f"Unknown file suffix: {path}")
        check_parameter_usage(extern_param_map, set(self.torch_to_path.keys()))

    def load(
        self, device: Device, preshard_funcs: Dict[str, Callable] = None
    ) -> Iterator[Tuple[str, NDArray]]:
        """Load the parameters and yield the MLC parameter and its value.

        Parameters
        ----------
        device : Optional[Device]
            The device to store the parameter, default to None, which means using CPU.

        Yields
        ------
        Tuple[str, NDArray]
            The MLC parameter name and its value, quantized if quantization mapping is provided.
        """
        mlc_names = _loading_order(self.extern_param_map, self.torch_to_path)
        for mlc_name in tqdm(mlc_names):
            param = self._load_mlc_param(mlc_name, device=device)
            # Apply quantization if needed, in this case the original parameter may become
            # multiple quantized parameters.
            for name, loader_param in self._load_or_quantize(mlc_name, param, device):
                # Apply presharding if needed
                if preshard_funcs is not None and name in preshard_funcs:
                    for shard_id, shard_param in enumerate(preshard_funcs[name](loader_param)):
                        yield _sharded_param_name(name, shard_id), shard_param
                else:
                    yield name, loader_param

        cached_files = list(self.cached_files.keys())
        for path in cached_files:
            self._unload_file(path)
        self.stats.log_time_info("HF")
        self.stats.log_mem_usage()

    def _load_mlc_param(self, mlc_name: str, device: Optional[Device]) -> NDArray:
        torch_names = self.extern_param_map.param_map[mlc_name]
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
        torch_params = [self.cached_files[self.torch_to_path[i]][i] for i in torch_names]
        # Step 4. Apply the mapping function
        with self.stats.timer("map_time_sec"):
            param = self.extern_param_map.map_func[mlc_name](*torch_params)
        if device:
            return as_ndarray(param, device=device)
        return as_ndarray(param)

    def _load_or_quantize(self, mlc_name, param, device: Device):
        if self.quantize_param_map and mlc_name in self.quantize_param_map.param_map:
            with self.stats.timer("quant_time_sec"):
                q_names = self.quantize_param_map.param_map[mlc_name]
                q_params = self.quantize_param_map.map_func[mlc_name](param)
                device.sync()
            for q_name, q_param in zip(q_names, q_params):
                logger.info(
                    '[Quantized] Parameter: "%s", shape: %s, dtype: %s',
                    bold(q_name),
                    q_param.shape,
                    q_param.dtype,
                )
                yield q_name, q_param
        else:
            logger.info(
                '[Not quantized] Parameter: "%s", shape: %s, dtype: %s',
                bold(mlc_name),
                param.shape,
                param.dtype,
            )
            device.sync()
            yield mlc_name, param

    def _load_file(self, path: Path) -> None:
        logger.info("Loading HF parameters from: %s", path)
        load_func = load_safetensor_shard if path.suffix == ".safetensors" else load_torch_shard
        with self.stats.timer("load_time_sec"):
            result = {}
            for name, param in load_func(path):
                result[name] = param
                self.stats.mem_add(param.nbytes)
                if name not in self.extern_param_map.unused_params:
                    self.stats.total_param_num += param.size
            self.cached_files[path] = result

    def _unload_file(self, path: Path) -> None:
        logger.info("Unloading HF weight file: %s", path)
        with self.stats.timer("load_time_sec"):
            for _, param in self.cached_files[path].items():
                self.stats.mem_rm(param.nbytes)
            del self.cached_files[path]
            gc.collect()


def _loading_order(param_map: ExternMapping, torch_to_path: Dict[str, Path]) -> List[str]:
    # Step 1. Build a map from path to torch parameters
    path_to_torch: Dict[Path, List[str]] = defaultdict(list)
    for torch_name, path in torch_to_path.items():
        path_to_torch[path].append(torch_name)
    # Step 2. Build a map from torch parameters to MLC parameters
    torch_to_mlc = defaultdict(list)
    for mlc_name, torch_names in param_map.param_map.items():
        for torch_name in torch_names:
            torch_to_mlc[torch_name].append(mlc_name)
    # Step 3. Construct the ordering that ensures file locality
    order = OrderedDict()
    for _, torch_names in path_to_torch.items():
        for torch_name in torch_names:
            for mlc_name in torch_to_mlc[torch_name]:
                if mlc_name not in order:
                    order[mlc_name] = 1
    return list(order.keys())


__all__ = ["HuggingFaceLoader"]
