"""Common utilities for loading parameters"""

# pylint: disable=too-few-public-methods
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Set, Tuple

import numpy as np

from mlc_llm.support import logging

if TYPE_CHECKING:
    from tvm.runtime import NDArray

    from .mapping import ExternMapping


logger = logging.getLogger(__name__)


def check_parameter_usage(param_map: "ExternMapping", extern_weights: Set[str]):
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
        if param is None:
            logger.warning("Encountered None param, skipping it: %s", name)
            continue
        param = param.detach().cpu()
        dtype = str(param.dtype)
        if dtype == "torch.bfloat16":
            param = param.float()
        param = param.numpy()
        yield name, param


def load_safetensor_shard(path: Path) -> Iterator[Tuple[str, np.ndarray]]:
    """Load and yield SafeTensor format parameters."""
    import safetensors  # pylint: disable=import-outside-toplevel,import-error
    import torch  # pylint: disable=import-outside-toplevel

    with safetensors.safe_open(path, framework="pt", device="cpu") as in_file:
        for name in in_file.keys():
            param = in_file.get_tensor(name)
            param = param.detach().cpu()
            dtype = str(param.dtype)
            if dtype == "torch.bfloat16":
                import ml_dtypes  # pylint: disable=import-outside-toplevel

                param = param.view(torch.float16).cpu().numpy().view(ml_dtypes.bfloat16)
            elif dtype == "torch.float8_e4m3fn":
                import ml_dtypes  # pylint: disable=import-outside-toplevel

                param = param.view(torch.uint8).cpu().numpy().view(ml_dtypes.float8_e4m3fn)
            else:
                param = param.numpy()
            yield name, param
