"""Common utilities for loading parameters"""
# pylint: disable=too-few-public-methods
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional, Set, Tuple

import numpy as np

from .mapping import ExternMapping

if TYPE_CHECKING:
    from tvm.runtime import NDArray

    from ..parameter import QuantizeMapping

logger = logging.getLogger(__name__)


class ParamQuantizer:
    """A parameter quantizer that quantizes given mlc-llm parameters"""

    quantize_map: "QuantizeMapping"

    def __init__(self, quantize_map: "QuantizeMapping") -> None:
        self.quantize_map = quantize_map

    def quantize(self, name: str, param: "NDArray") -> Optional[Iterator[Tuple[str, "NDArray"]]]:
        """Apply quantization to the given parameters

        Parameters
        ----------
        name : str
            The name of the parameter
        param : NDArray
            The parameter to be quantized

        Returns
        -------
        Optional[Iterator[Tuple[str, "NDArray"]]]
            The quantized parameters, each with its name, returns None if the parameter is not
            quantized.
        """
        name = f".{name}"
        if name not in self.quantize_map.param_map:
            return None
        assert name in self.quantize_map.map_func, f"Quantization function for {name} not found."
        quantized_names = self.quantize_map.param_map[name]
        quantized_params = self.quantize_map.map_func[name](param)
        return zip(quantized_names, quantized_params)


def check_parameter_usage(param_map: ExternMapping, extern_weights: Set[str]):
    """Check that all external parameters have been used and are stored in the weights file."""
    used_extern_names = set(sum(param_map.param_map.values(), []))
    # Check 1. All extern parameters in the weight files are used unless explicitly specified
    unused_extern_names = extern_weights - used_extern_names - param_map.unused_params
    if unused_extern_names:
        logger.warning(
            "Unused extern parameters: %s",
            ", ".join(sorted(unused_extern_names)),
        )
    # Check 2. All extern parameters required are stored in the weight files
    nonexistent_extern_names = used_extern_names - extern_weights
    if nonexistent_extern_names:
        raise ValueError(
            "The following extern parameters do not exist in the weight files:\n  "
            + "\n  ".join(sorted(nonexistent_extern_names)),
        )


def load_torch_shard(path: Path) -> Iterator[Tuple[str, np.ndarray]]:
    """Load and yield PyTorch format parameters."""
    import torch  # pylint: disable=import-outside-toplevel

    for name, param in torch.load(path, map_location=torch.device("cpu")).items():
        param = param.detach().cpu()
        dtype = str(param.dtype)
        if dtype == "torch.bfloat16":
            param = param.float()
        param = param.numpy()
        yield name, param


def load_safetensor_shard(path: Path) -> Iterator[Tuple[str, np.ndarray]]:
    """Load and yield SafeTensor format parameters."""
    import safetensors  # pylint: disable=import-outside-toplevel,import-error

    with safetensors.safe_open(path, framework="numpy", device="cpu") as in_file:
        for name in in_file.keys():
            param = in_file.get_tensor(name)
            yield name, param
