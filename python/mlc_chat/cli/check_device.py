"""Check if a device exists."""
import sys

import tvm


def main():
    """Entrypoint for device check."""
    device_str = sys.argv[1]
    try:
        device = tvm.runtime.device(device_str)
        if device.exist:
            print("1")
        else:
            print("0")
    except:  # pylint: disable=bare-except
        print("0")


if __name__ == "__main__":
    main()
