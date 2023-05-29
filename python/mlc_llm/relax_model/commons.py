from typing import List

import json
from tvm import relax


def create_metadata_func(
    bb: relax.BlockBuilder,
    model_name: str,
    max_window_size: int,
    stop_tokens: List[int],
    add_prefix_space: bool,
):
    metadata = json.dumps(
        {
            "model_name": model_name,
            "max_window_size": max_window_size,
            "stop_tokens": stop_tokens,
            "add_prefix_space": add_prefix_space,
        }
    )
    with bb.function("get_metadata", params=[]):
        bb.emit_func_output(relax.StringImm(metadata))
