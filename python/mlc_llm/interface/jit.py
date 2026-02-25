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
from typing import Any, Dict, Optional, Union

from tvm.runtime import Device

from mlc_llm.model import MODELS
from mlc_llm.support import logging
from mlc_llm.support.auto_device import device2str
from mlc_llm.support.constants import (
    MLC_DSO_SUFFIX,
    MLC_JIT_POLICY,
    MLC_LLM_HOME,
    MLC_TEMP_DIR,
)
from mlc_llm.support.style import blue, bold

from .compiler_flags import ModelConfigOverride, OptimizationFlags

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class JITResult:
    """The jit compilation result class."""

    model_lib_path: str
    system_lib_prefix: Optional[str] = None


def log_jit_policy():
    """log current jit policy"""
    logger.info(
        "%s = %s. Can be one of: ON, OFF, REDO, READONLY",
        bold("MLC_JIT_POLICY"),
        MLC_JIT_POLICY,
    )


def jit(  # pylint: disable=too-many-locals,too-many-statements
    model_path: Path,
    overrides: Dict[str, Any],
    device: Union[Device, str],
    system_lib_prefix: Optional[str] = None,
    *,
    skip_log_jit_policy=False,
) -> JITResult:
    """Just-in-time compile a MLC-Chat model."""
    # skip logging jit policy since when outside can hint once
    if not skip_log_jit_policy:
        log_jit_policy()

    if MLC_JIT_POLICY == "OFF":
        raise RuntimeError("JIT is disabled by MLC_JIT_POLICY=OFF")

    with open(model_path / "mlc-chat-config.json", "r", encoding="utf-8") as in_file:
        mlc_chat_config = json.load(in_file)
    model_type = mlc_chat_config.pop("model_type")
    quantization = mlc_chat_config.pop("quantization")
    lib_suffix = MLC_DSO_SUFFIX if device not in ["iphone", "macabi", "android"] else "tar"

    def _get_optimization_flags() -> str:
        opt = overrides.pop("opt", None)
        if opt is None:
            opt = "O2"
        return repr(OptimizationFlags.from_str(opt))

    def _get_overrides() -> str:
        forbid_list = [
            "context_window_size",
            "sliding_window_size",
            "attention_sink_size",
        ]
        result = []
        for field in dataclasses.fields(ModelConfigOverride):
            value = overrides.get(field.name, None)
            if value is not None:
                if field.name in forbid_list and value == -1:
                    continue
                result.append(f"{field.name}={value}")
        return ";".join(result)

    def _get_model_config() -> Dict[str, Any]:
        model_config = mlc_chat_config.pop("model_config")
        model_config.update(mlc_chat_config)
        for field in dataclasses.fields(ModelConfigOverride):
            value = overrides.get(field.name, None)
            if value is not None:
                model_config[field.name] = value
        return MODELS[model_type].config.from_dict(model_config).asdict()

    def _run_jit(
        opt: str,
        overrides: str,
        device: str,
        system_lib_prefix: Optional[str],
        dst: str,
    ):
        with tempfile.TemporaryDirectory(dir=MLC_TEMP_DIR) as tmp_dir:
            dso_path = os.path.join(tmp_dir, f"lib.{lib_suffix}")
            cmd = [
                sys.executable,
                "-m",
                "mlc_llm",
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
            if system_lib_prefix:
                cmd += ["--system-lib-prefix", system_lib_prefix + "_"]
            logger.info("Compiling using commands below:")
            logger.info("%s", blue(shlex.join(cmd)))
            subprocess.run(cmd, check=False, env=os.environ)
            # note on windows: compilation can succeed but return code is still nonzero
            # check whether file exists instead
            if not os.path.isfile(dso_path):
                raise RuntimeError("Cannot find compilation output, compilation failed")
            shutil.move(dso_path, dst)
            logger.info("Using compiled model lib: %s", bold(dst))

    hash_key = {
        "model_config": _get_model_config(),
        "overrides": _get_overrides(),
        "opt": _get_optimization_flags(),
        "device": device2str(device) if isinstance(device, Device) else device,
        "model_type": model_type,
        "quantization": quantization,
    }
    if device in ["iphone", "macabi", "android"]:
        if system_lib_prefix is None:
            system_lib_hash_value = hashlib.md5(
                json.dumps(
                    hash_key,
                    sort_keys=True,
                    indent=2,
                ).encode("utf-8")
            ).hexdigest()
            system_lib_prefix = f"{model_type}_{quantization}_{system_lib_hash_value}".replace(
                "-", "_"
            )
        hash_key["system_lib_prefix"] = system_lib_prefix
    hash_value = hashlib.md5(
        json.dumps(
            hash_key,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
    ).hexdigest()
    dst = MLC_LLM_HOME / "model_lib" / f"{hash_value}.{lib_suffix}"
    if dst.is_file() and MLC_JIT_POLICY in ["ON", "READONLY"]:
        logger.info("Using cached model lib: %s", bold(str(dst)))
        return JITResult(str(dst), system_lib_prefix)
    if MLC_JIT_POLICY == "READONLY":
        raise RuntimeError(
            "No cached model lib found, and JIT is disabled by MLC_JIT_POLICY=READONLY"
        )
    _run_jit(
        opt=hash_key["opt"],
        overrides=hash_key["overrides"],
        device=hash_key["device"],
        system_lib_prefix=system_lib_prefix,
        dst=str(dst),
    )
    return JITResult(str(dst), system_lib_prefix)
