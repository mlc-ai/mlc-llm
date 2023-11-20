"""Automatic detection of the device available on the local machine."""
import logging

import tvm
from tvm.runtime import Device

from .style import bold, green, red

FOUND = green("Found")
NOT_FOUND = red("Not found")
AUTO_DETECT_DEVICES = ["cuda", "rocm", "metal", "vulkan", "opencl"]


logger = logging.getLogger(__name__)


def detect_device(device_hint: str) -> Device:
    """Detect locally available device from string hint."""
    if device_hint == "auto":
        device = None
        for device_type in AUTO_DETECT_DEVICES:
            cur_device = tvm.device(dev_type=device_type, dev_id=0)
            if cur_device.exist:
                logger.info("%s device: %s:0", FOUND, device_type)
                if device is None:
                    device = cur_device
            else:
                logger.info("%s device: %s:0", NOT_FOUND, device_type)
        if device is None:
            logger.info("%s: No available device detected. Falling back to CPU", NOT_FOUND)
            return tvm.device("cpu:0")
        device_str = f"{tvm.runtime.Device.MASK2STR[device.device_type]}:{device.device_id}"
        logger.info("Using device: %s. Use `--device` to override.", bold(device_str))
        return device
    try:
        device = tvm.device(device_hint)
    except Exception as err:
        raise ValueError(f"Invalid device name: {device_hint}") from err
    if not device.exist:
        raise ValueError(f"Device is not found on your local environment: {device_hint}")
    return device
