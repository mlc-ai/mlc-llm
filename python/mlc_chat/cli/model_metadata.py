"""A tool that inspects the metadata of a model lib."""
import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np

from mlc_chat.support import logging
from mlc_chat.support.argparse import ArgumentParser
from mlc_chat.support.style import green, red

logging.enable_logging()
logger = logging.getLogger(__name__)


def _extract_metadata(model_lib: Path) -> Dict[str, Any]:
    # pylint: disable=import-outside-toplevel
    from tvm.runtime import device, load_module
    from tvm.runtime.relax_vm import VirtualMachine

    # pylint: enable=import-outside-toplevel

    return json.loads(VirtualMachine(load_module(model_lib), device("cpu"))["_metadata"]())


def _report_all(metadata: Dict[str, Any]) -> None:
    # Print JSON with aesthetic values that packs each parameter into one line,
    # while keeping the rest indented.
    indent = 2
    indents = " " * indent
    params = metadata.pop("params")
    params = indents * 2 + (",\n" + indents * 2).join(json.dumps(p) for p in params)
    lines = json.dumps(
        metadata,
        sort_keys=True,
        indent=indent,
    ).splitlines()
    lines.insert(1, indents + '"params": [\n' + params + "\n" + indents + "],")
    beautified_json = "\n".join(lines)
    print(beautified_json)


def _report_memory_usage(metadata: Dict[str, Any]) -> None:
    params_bytes = 0.0
    for param in metadata["params"]:
        if all(v > 0 for v in param["shape"]):
            params_bytes += math.prod(param["shape"]) * np.dtype(param["dtype"]).itemsize
    temp_func_bytes = 0.0
    for _func_name, func_bytes in metadata["memory_usage"].items():
        temp_func_bytes = max(temp_func_bytes, func_bytes)
    kv_cache_bytes = metadata["kv_cache_bytes"]

    total_size = params_bytes + temp_func_bytes + kv_cache_bytes
    logger.info(
        "%s: %.2f MB (Parameters: %.2f MB. KVCache: %.2f MB. Temporary buffer: %.2f MB)",
        green("Total memory usage"),
        total_size / 1024 / 1024,
        params_bytes / 1024 / 1024,
        kv_cache_bytes / 1024 / 1024,
        temp_func_bytes / 1024 / 1024,
    )

    logger.info(
        "To reduce memory usage, "
        "tweak `prefill_chunk_size`, `context_window_size` and `sliding_window_size`"
    )


def main():
    """Entry point for the model metadata tool."""
    parser = ArgumentParser(description="A tool that inspects the metadata of a model lib.")
    parser.add_argument(
        "model_lib",
        type=Path,
        help="""The compiled model library. In MLC LLM, an LLM is compiled to a shared or static
        library (.so or .a), which contains GPU computation to efficiently run the LLM. MLC Chat,
        as the runtime of MLC LLM, depends on the compiled model library to generate tokens.
        """,
    )
    parser.add_argument(
        "--memory-only",
        action="store_true",
        help="""If set, only inspect the metadata in memory usage and print richer analysis.
        Otherwise, the tool will load all the metadata from the model library file but only print
        the basic information in JSON.
        """,
    )
    parsed = parser.parse_args()
    try:
        metadata = _extract_metadata(parsed.model_lib)
    except:  # pylint: disable=bare-except
        logger.exception("%s to read metadata section in legacy model lib.", red("FAILED"))
        return
    if parsed.memory_only:
        _report_memory_usage(metadata)
    else:
        _report_all(metadata)


if __name__ == "__main__":
    main()
