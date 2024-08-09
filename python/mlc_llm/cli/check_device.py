"""Check if a device exists."""

import os
import sys

from tvm.runtime import Device
from tvm.runtime import device as as_device


def _check_device(device: Device) -> bool:
    try:
        return bool(device.exist)
    except:  # pylint: disable=bare-except
        return False


def main():
    """Entrypoint for device check."""
    device_str = sys.argv[1]
    device_ids = []
    i = 0
    while True:
        if _check_device(as_device(device_str, i)):
            device_ids.append(i)
            i += 1
            if device_str in ["cpu", "llvm"] and i > os.cpu_count() / 2:
                break
        else:
            break
    print(f"check_device:{','.join(str(i) for i in device_ids)}")


if __name__ == "__main__":
    main()
