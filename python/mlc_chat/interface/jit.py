"""Just-in-time compilation of MLC-Chat models."""
import dataclasses
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

from tvm.runtime import Device

from mlc_chat.model import MODELS
from mlc_chat.support import logging
from mlc_chat.support.auto_device import device2str
from mlc_chat.support.constants import MLC_CACHE_DIR, MLC_JIT_POLICY, MLC_TEMP_DIR
from mlc_chat.support.style import blue, bold

from .compiler_flags import ModelConfigOverride, OptimizationFlags

logger = logging.getLogger(__name__)


def jit(model_path: Path, chat_config: Dict[str, Any], device: Device) -> Path:
    """Just-in-time compile a MLC-Chat model."""
    if MLC_JIT_POLICY == "OFF":
        raise RuntimeError("JIT is disabled by MLC_JIT_POLICY=OFF")

    with open(model_path / "mlc-chat-config.json", "r", encoding="utf-8") as in_file:
        mlc_chat_config = json.load(in_file)
    model_type = mlc_chat_config.pop("model_type")
    quantization = mlc_chat_config.pop("quantization")

    def _get_optimization_flags() -> str:
        opt = chat_config.pop("opt", None)
        if opt is None:
            opt = "O2"
        return repr(OptimizationFlags.from_str(opt))

    def _get_overrides() -> str:
        forbid_list = ["context_window_size", "sliding_window_size", "attention_sink_size"]
        result = []
        for field in dataclasses.fields(ModelConfigOverride):
            value = chat_config.get(field.name, None)
            if value is not None:
                if field.name in forbid_list and value == -1:
                    continue
                result.append(f"{field.name}={value}")
        if not result:
            result = ["tensor_parallel_shards=1"]
        return ";".join(result)

    def _get_model_config() -> Dict[str, Any]:
        model_config = mlc_chat_config.pop("model_config")
        model_config.update(mlc_chat_config)
        for field in dataclasses.fields(ModelConfigOverride):
            value = chat_config.get(field.name, None)
            if value is not None:
                model_config[field.name] = value
        return MODELS[model_type].config.from_dict(model_config).asdict()

    def _run_jit(opt: str, overrides: str, device: str, dst: str):
        with tempfile.TemporaryDirectory(dir=MLC_TEMP_DIR) as tmp_dir:
            dso_path = os.path.join(tmp_dir, "lib.so")
            cmd = [
                sys.executable,
                "-m",
                "mlc_chat",
                "compile",
                str(model_path),
                "--opt",
                opt,
                "--overrides",
                overrides,
                "--device",
                device,
                "--output",
                dso_path,
            ]
            logger.info("Compiling using commands below:")
            logger.info("%s", blue(shlex.join(cmd)))
            subprocess.run(cmd, check=True)
            shutil.move(dso_path, dst)
            logger.info("Using compiled model lib: %s", bold(dst))

    hash_key = {
        "model_config": _get_model_config(),
        "overrides": _get_overrides(),
        "opt": _get_optimization_flags(),
        "device": device2str(device),
        "model_type": model_type,
        "quantization": quantization,
    }
    hash_value = hashlib.md5(
        json.dumps(
            hash_key,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
    ).hexdigest()
    dst = MLC_CACHE_DIR / "model_lib" / f"{hash_value}.so"
    if dst.is_file() and MLC_JIT_POLICY in ["ON", "READONLY"]:
        logger.info("Using cached model lib: %s", bold(str(dst)))
        return dst
    if MLC_JIT_POLICY == "READONLY":
        raise RuntimeError(
            "No cached model lib found, and JIT is disabled by MLC_JIT_POLICY=READONLY"
        )
    _run_jit(
        opt=hash_key["opt"],
        overrides=hash_key["overrides"],
        device=hash_key["device"],
        dst=str(dst),
    )
    return dst
