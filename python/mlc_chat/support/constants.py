"""Environment variables used by the MLC LLM."""
import os
import sys
from pathlib import Path


def _check():
    if MLC_JIT_POLICY not in ["ON", "OFF", "REDO", "READONLY"]:
        raise ValueError(
            'Invalid MLC_JIT_POLICY. It has to be one of "ON", "OFF", "REDO", "READONLY"'
            f"but got {MLC_JIT_POLICY}."
        )


def _get_cache_dir() -> Path:
    if "MLC_CACHE_DIR" in os.environ:
        result = Path(os.environ["MLC_CACHE_DIR"])
    elif sys.platform == "win32":
        result = Path(os.environ["LOCALAPPDATA"])
        result = result / "mlc_chat"
    elif os.getenv("XDG_CACHE_HOME", None) is not None:
        result = Path(os.getenv("XDG_CACHE_HOME"))
        result = result / "mlc_chat"
    else:
        result = Path(os.path.expanduser("~/.cache"))
        result = result / "mlc_chat"
    result.mkdir(parents=True, exist_ok=True)
    if not result.is_dir():
        raise ValueError(
            f"The default cache directory is not a directory: {result}. "
            "Use environment variable MLC_CACHE_DIR to specify a valid cache directory."
        )
    (result / "model_weights").mkdir(parents=True, exist_ok=True)
    (result / "model_lib").mkdir(parents=True, exist_ok=True)
    return result


MLC_TEMP_DIR = os.getenv("MLC_TEMP_DIR", None)
MLC_MULTI_ARCH = os.environ.get("MLC_MULTI_ARCH", None)
MLC_CACHE_DIR: Path = _get_cache_dir()
MLC_JIT_POLICY = os.environ.get("MLC_JIT_POLICY", "ON")


_check()
