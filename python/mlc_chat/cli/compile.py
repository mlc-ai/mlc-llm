"""Command line entrypoint of compilation."""
import argparse
import logging
from pathlib import Path
from typing import Union

from mlc_chat.compiler import (  # pylint: disable=redefined-builtin
    MODELS,
    QUANT,
    OptimizationFlags,
    compile,
)

from ..support.auto_config import detect_config, detect_model_type
from ..support.auto_target import detect_target_and_host

logging.basicConfig(
    level=logging.DEBUG,
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    format="[{asctime}] {levelname} {filename}:{lineno}: {message}",
)


def main():
    """Parse command line argumennts and call `mlc_llm.compiler.compile`."""

    def _parse_config(path: Union[str, Path]) -> Path:
        try:
            return detect_config(Path(path))
        except ValueError as err:
            raise argparse.ArgumentTypeError(f"No valid config.json in: {path}. Error: {err}")

    def _parse_output(path: Union[str, Path]) -> Path:
        path = Path(path)
        parent = path.parent
        if not parent.is_dir():
            raise argparse.ArgumentTypeError(f"Directory does not exist: {parent}")
        return path

    parser = argparse.ArgumentParser("MLC LLM Compiler")
    parser.add_argument(
        "--config",
        type=_parse_config,
        required=True,
        help="Path to config.json file or to the directory that contains config.json, which is "
        "a HuggingFace standard that defines model architecture, for example, "
        "https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf/blob/main/config.json",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        required=True,
        choices=list(QUANT.keys()),
        help="Quantization format.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto"] + list(MODELS.keys()),
        help="Model architecture, for example, llama. If not set, it is inferred "
        "from the config.json file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="The GPU device to compile the model to. If not set, it is inferred from locally "
        "available GPUs.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="auto",
        choices=[
            "auto",
            "arm",
            "arm64",
            "aarch64",
            "x86-64",
        ],
        help="The host CPU ISA to compile the model to. If not set, it is inferred from the "
        "local CPU.",
    )
    parser.add_argument(
        "--opt",
        type=OptimizationFlags.from_str,
        default="",
        help="Optimization flags. MLC LLM maintains a predefined set of optimization flags, "
        "denoted as O0, O1, O2, O3, where O0 means no optimization, O2 means majority of them, "
        "and O3 represents extreme optimization that could potentially break the system. "
        "Meanwhile, optimization flags could be explicitly specified via details knobs, e.g. "
        '--opt="cutlass_attn=1;cutlass_norm=0;cublas_gemm=0;cudagraph=0"',
    )
    parser.add_argument(
        "--prefix-symbols",
        type=str,
        default="",
        help='Adding a prefix to all symbols exported. Similar to "objcopy --prefix-symbols". '
        "This is useful when compiling multiple models into a single library to avoid symbol "
        "conflicts. Differet from objcopy, this takes no effect for shared library.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=_parse_output,
        required=True,
        help="The name of the output file. The suffix determines if the output file is a "
        "shared library or a static library. Available suffixes: "
        "1) Linux: .so (shared), .tar (static); "
        "2) macOS: .dylib (shared), .tar (static); "
        "3) Windows: .dll (shared), .tar (static); "
        "4) Android, iOS: .tar (static); "
        "5) Web: .wasm (web assembly)",
    )
    parsed = parser.parse_args()
    target, build_func = detect_target_and_host(parsed.device, parsed.host)
    parsed.model_type = detect_model_type(parsed.model_type, parsed.config)
    compile(
        config=parsed.config,
        quantization=parsed.quantization,
        model_type=parsed.model_type,
        target=target,
        opt=parsed.opt,
        build_func=build_func,
        prefix_symbols=parsed.prefix_symbols,
        output=parsed.output,
    )


if __name__ == "__main__":
    main()
