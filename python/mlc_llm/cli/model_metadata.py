"""A tool that inspects the metadata of a model lib."""

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Union

from tvm.runtime import DataType

from mlc_llm.support import logging
from mlc_llm.support.argparse import ArgumentParser
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import green, red

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


def _read_dynamic_shape(shape: List[Union[int, str]], config: Union[Dict, ConfigBase]) -> List[int]:
    if isinstance(config, ConfigBase):
        config = asdict(config)
    param_shape = []
    for s in shape:
        if isinstance(s, int):
            param_shape.append(s)
        else:
            if config is None:
                logger.error(
                    "%s: Encountered dynamic shape %s, need to specify `--mlc-chat-config` for "
                    + "memory usage calculation.",
                    red("FAILED"),
                    red(s),
                )
                raise AttributeError
            if not s in config:
                logger.error(
                    "%s to retrieve concrete %s for dynamic shape from %s.",
                    red("FAILED"),
                    red(s),
                    config,
                )
                raise KeyError
            param_shape.append(config[s])
    return param_shape


def _compute_memory_usage(metadata: Dict[str, Any], config: Union[Dict, ConfigBase]):
    params_bytes = 0.0
    for param in metadata["params"]:
        if all(isinstance(v, int) for v in param["shape"]):
            assert all(v > 0 for v in param["shape"]), "All shapes should be strictly positive."
            param_shape = param["shape"]
        else:
            # Contains dynamic shape; use config to look up concrete values
            param_shape = _read_dynamic_shape(param["shape"], config)
        params_bytes += math.prod(param_shape) * DataType(param["dtype"]).itemsize
    temp_func_bytes = 0.0
    for _func_name, func_bytes in metadata["memory_usage"].items():
        temp_func_bytes = max(temp_func_bytes, func_bytes)

    return params_bytes, temp_func_bytes


def _report_memory_usage(metadata: Dict[str, Any], config: Union[Dict, ConfigBase]) -> None:
    params_bytes, temp_func_bytes = _compute_memory_usage(metadata, config)
    total_size = params_bytes + temp_func_bytes
    logger.info(
        "%s: %.2f MB (Parameters: %.2f MB. Temporary buffer: %.2f MB)",
        green("Total memory usage without KV cache:"),
        total_size / 1024 / 1024,
        params_bytes / 1024 / 1024,
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
        "--mlc-chat-config",
        type=Path,
        help="""The `mlc-chat-config.json` file specific to a model variant. This is only required
        when `memory-only` is true and `model_lib` contains a dynamic parameter shape (i.e. using
        a variable to represent the shape). For instance, `model.embed_tokens.q_weight` can have
        shape `["vocab_size", 512]`. In these cases, we look up the concrete value in
        `mlc-chat-config.json`.
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
    # Load metadata from model lib
    try:
        metadata = _extract_metadata(parsed.model_lib)
    except:  # pylint: disable=bare-except
        logger.exception("%s to read metadata section in legacy model lib.", red("FAILED"))
        return
    # Load mlc_chat_config if provided
    cfg = None
    if parsed.mlc_chat_config:
        mlc_chat_config_path = Path(parsed.mlc_chat_config)
        if not mlc_chat_config_path.exists():
            raise ValueError(f"{mlc_chat_config_path} does not exist.")
        with open(mlc_chat_config_path, "r", encoding="utf-8") as config_file:
            cfg = json.load(config_file)
    # Main body
    if parsed.memory_only:
        _report_memory_usage(metadata, cfg)
    else:
        _report_all(metadata)


if __name__ == "__main__":
    main()
