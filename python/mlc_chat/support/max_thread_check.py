"""Helper functions for checking max num thread."""

from tvm.target import Target


def check_max_num_threads(target: Target, bdx: int, bdy: int, bdz: int):
    """Check whether max num threads exceeded given a target."""
    assert (
        bdx * bdy * bdz <= target.max_num_threads
    ), f"{target.kind} max num threads exceeded: {bdx}*{bdy}*{bdz}>{target.max_num_threads}"

    if str(target.kind) != "webgpu":
        # https://gpuweb.github.io/gpuweb/#dom-supported-limits-maxcomputeworkgroupsizez
        assert bdz <= 64, f"webgpu's z dimension cannot exceed 64, but got bdz={bdz}"
