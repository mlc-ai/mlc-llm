"""A weight loader for HuggingFace's PyTorch format"""
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np

from .base_loader import BaseLoader, Stats
from .mapping import ExternMapping
from .utils import load_torch

logger = logging.getLogger(__name__)


class HFTorchLoader(BaseLoader):  # pylint: disable=too-few-public-methods
    """A loader loading HuggingFace's PyTorch format and converts them to MLC's parameters.

    Attributes
    ----------
    stats : Stats
        Statistics of the loading process.

    extern_param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.

    param_to_path : Dict[str, Path]
        A mapping from PyTorch parameter name to the path of the file containing it, or the path
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
        """Create a parameter loader from HuggingFace PyTorch format.

        Parameters
        ----------
        path : pathlib.Path
            Path to either a JSON indexing file, or a PyTorch bin file.
            1) For JSON indexing file, it is usually `pytorch_model.bin.index.json` in the repo,
            which contains a `weight_map` that maps each PyTorch parameter to the file containing
            the weight. 2) For PyTorch bin file, it is usually `pytorch_model.bin` in the repo,
            which contains all the parameters.

        extern_param_map : ExternMapping
            Maps an MLC parameter to a list of PyTorch parameters.
        """
        assert path.is_file()
        self.stats = Stats()
        self.extern_param_map = extern_param_map
        self.cached_files = {}
        self.param_to_path = {}
        if path.suffix == ".bin":
            self._load_file(path)
            for name in self.cached_files[path].keys():
                self.param_to_path[name] = path
        elif path.suffix == ".json":
            with path.open("r", encoding="utf-8") as in_file:
                torch_weight_map = json.load(in_file)["weight_map"]
            for torch_name, path_str in torch_weight_map.items():
                self.param_to_path[torch_name] = path.parent / path_str
        else:
            raise FileNotFoundError(f"Unknown file suffix: {path}")
        self._check_parameter_usage(set(self.param_to_path.keys()))

    def _load_shard(self, path: Path):
        return load_torch(path)


__all__ = ["HFTorchLoader"]
