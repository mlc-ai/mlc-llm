"""Automatic detection of the device available on the local machine."""

import os
import subprocess
import sys
from typing import Dict, Optional

import tvm
from tvm.runtime import Device

from . import logging
from .style import bold, green, red

FOUND = green("Found")
NOT_FOUND = red("Not found")
AUTO_DETECT_DEVICES = ["cuda", "rocm", "metal", "vulkan", "opencl", "cpu"]
_RESULT_CACHE: Dict[str, bool] = {}


logger = logging.getLogger(__name__)


def detect_device(device_hint: str) -> Optional[Device]:
    """Detect locally available device from string hint."""
    if device_hint == "auto":
        device = None
        for device_type in AUTO_DETECT_DEVICES:
            cur_device = tvm.device(dev_type=device_type, dev_id=0)
            if _device_exists(cur_device):
                if device is None:
                    device = cur_device
        if device is None:
            logger.info("%s: No available device detected", NOT_FOUND)
            return None
        logger.info("Using device: %s", bold(device2str(device)))
        return device
    try:
        device = tvm.device(device_hint)
    except Exception as err:
        raise ValueError(f"Invalid device name: {device_hint}") from err
    if not _device_exists(device):
        raise ValueError(f"Device is not found on your local environment: {device_hint}")
    return device


def device2str(device: Device) -> str:
    """Convert a TVM device object to string."""
    return f"{tvm.runtime.Device.DEVICE_TYPE_TO_NAME[device.device_type]}:{device.device_id}"


def _device_exists(device: Device) -> bool:
    device_type = tvm.runtime.Device.DEVICE_TYPE_TO_NAME[device.device_type]
    device_str = device2str(device)
    if device_str in _RESULT_CACHE:
        return _RESULT_CACHE[device_str]
    cmd = [
        sys.executable,
        "-m",
        "mlc_llm.cli.check_device",
        device_type,
    ]
    prefix = "check_device:"
    subproc_outputs = [
        line[len(prefix) :].strip()
        for line in subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env=os.environ,
        )
        .stdout.strip()
        .splitlines()
        if line.startswith(prefix)
    ]
    if subproc_outputs:
        if subproc_outputs[0]:
            for i in subproc_outputs[0].split(","):
                logger.info("%s device: %s:%s", FOUND, device_type, i)
                _RESULT_CACHE[f"{device_type}:{i}"] = True
                if device.device_type == Device.kDLCPU:
                    break
    else:
        logger.error(
            "GPU device detection failed. Please report this issue with the output of command: %s",
            " ".join(cmd),
        )
    if device_str in _RESULT_CACHE:
        return _RESULT_CACHE[device_str]
    logger.info("%s device: %s", NOT_FOUND, device_str)
    _RESULT_CACHE[device_str] = False
    return False
