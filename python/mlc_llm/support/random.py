"""Utility functions for random number generation."""

import sys


def set_global_random_seed(seed):
    """Set global random seed for python, numpy, torch and tvm."""
    if "numpy" in sys.modules:
        sys.modules["numpy"].random.seed(seed)
    if "torch" in sys.modules:
        sys.modules["torch"].manual_seed(seed)
    if "random" in sys.modules:
        sys.modules["random"].seed(seed)  # pylint: disable=no-member
    if "tvm" in sys.modules:
        set_seed = sys.modules["tvm"].get_global_func("mlc.random.set_seed")
        if set_seed:
            set_seed(seed)
