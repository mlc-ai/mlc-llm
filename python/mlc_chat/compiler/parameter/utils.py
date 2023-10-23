"""Comman paramter loading/quantization utilities"""
from pathlib import Path


def load_safetensor(path: Path):
    """Load SafeTensor format files.

    Parameters
    ----------
    path : pathlib.Path
        The Safetensor file path.
    """
    import safetensors  # pylint: disable=import-outside-toplevel

    with safetensors.safe_open(path, framework="numpy", device="cpu") as in_file:
        for name in in_file.keys():
            param = in_file.get_tensor(name)
            yield name, param


def load_torch(path: Path):
    """Load PyTorch format files.

    Parameters
    ----------
    path : pathlib.Path
        The Pytorch file path.
    """
    import torch  # pylint: disable=import-outside-toplevel

    for name, param in torch.load(path, map_location=torch.device("cpu")).items():
        param = param.detach().cpu()
        dtype = str(param.dtype)
        if dtype == "torch.bfloat16":
            param = param.float()
        param = param.numpy()
        yield name, param
