"""Automatic detection of the device available on the local machine."""
import logging
import subprocess
import sys
from typing import Dict

import tvm
from tvm.runtime import Device

from .style import bold, green, red

FOUND = green("Found")
NOT_FOUND = red("Not found")
AUTO_DETECT_DEVICES = ["cuda", "rocm", "metal", "vulkan", "opencl"]
_RESULT_CACHE: Dict[str, bool] = {}


logger = logging.getLogger(__name__)


def detect_device(device_hint: str) -> Device:
    """Detect locally available device from string hint."""
    if device_hint == "auto":
        device = None
        for device_type in AUTO_DETECT_DEVICES:
            cur_device = tvm.device(dev_type=device_type, dev_id=0)
            # NOTE: we do early exit at first detected device, so that we will not try
            # all the list of devices in auto detect to reduce VRAM
            # allocated for other environments
            if cur_device.exist:
                if device is None:
                    device = cur_device
                    break
        if device is None:
            logger.info("%s: No available device detected. Falling back to CPU", NOT_FOUND)
            return tvm.device("cpu:0")
        logger.info("Using device: %s. Use `--device` to override.", bold(_device_to_str(device)))
        return device
    try:
        device = tvm.device(device_hint)
    except Exception as err:
        raise ValueError(f"Invalid device name: {device_hint}") from err
    if not device.exist:
        raise ValueError(f"Device is not found on your local environment: {device_hint}")
    return device


def _device_to_str(device: Device) -> str:
    return f"{tvm.runtime.Device.MASK2STR[device.device_type]}:{device.device_id}"


def _device_exists_via_check_device(device: Device) -> bool:
    """Check whether device exists via mlc_chat.cli.check_device"""
    # NOTE: this function may have issues in return value checkings in windows
    # as of now we use the same process checking with early exit.
    device_str = _device_to_str(device)
    if device_str in _RESULT_CACHE:
        return _RESULT_CACHE[device_str]
    cmd = [
        sys.executable,
        "-m",
        "mlc_chat.cli.check_device",
        device_str,
    ]
    result = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8")
    result_bool = result.strip() == "1"
    if result_bool:
        logger.info("%s device: %s", FOUND, device_str)
    else:
        logger.info("%s device: %s", NOT_FOUND, device_str)
    _RESULT_CACHE[device_str] = result_bool
    return result_bool
