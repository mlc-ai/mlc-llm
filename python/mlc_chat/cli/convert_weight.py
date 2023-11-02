"""Command line entrypoint of weight conversion."""
import argparse
import logging
from pathlib import Path
from typing import Union

import tvm
from mlc_chat.compiler import MODELS, QUANTIZATION
from mlc_chat.compiler.parameter import HuggingFaceLoader
from mlc_chat.support import tqdm
from tvm.contrib import tvmjs

from ..support.auto_config import detect_config, detect_model_type
from ..support.auto_target import detect_target_and_host
from ..support.auto_weight import detect_weight

logging.basicConfig(
    level=logging.INFO,
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    format="[{asctime}] {levelname} {filename}:{lineno}: {message}",
)


def main():
    """Parse command line argumennts and apply quantization."""

    def _parse_config(path: Union[str, Path]) -> Path:
        try:
            return detect_config(path)
        except ValueError as err:
            raise argparse.ArgumentTypeError(f"No valid config.json in: {path}. Error: {err}")

    def _parse_source(path: Union[str, Path], config_path: Path) -> Path:
        if path == "auto":
            return config_path.parent
        path = Path(path)
        if not path.is_dir():
            raise argparse.ArgumentTypeError(f"Directory does not exist: {path}")
        return path

    def _parse_output(path: Union[str, Path]) -> Path:
        path = Path(path)
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
        return path

    parser = argparse.ArgumentParser("MLC AutoLLM Quantization Framework")
    parser.add_argument(
        "--config",
        type=_parse_config,
        required=True,
        help="Path to config.json file or to the directory that contains config.json, which is "
        "a HuggingFace standard that defines model architecture, for example, "
        "https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf/blob/main/config.json",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=False,
        default="auto",
        help="The path to original model weight, infer from `config` if missing",
    )
    parser.add_argument(
        "--source-format",
        type=str,
        required=False,
        choices=["auto", "huggingface-torch", "huggingface-safetensor"],
        default="auto",
        help="The format of source model weight, infer from `config` if missing",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        required=True,
        choices=list(QUANTIZATION.keys()),
        help="Quantization format, for example `q4f16_1`.",
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
        help="The device used to do quantization, \
              for example `auto` / `cuda:0` / `cuda --arch sm86`",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=_parse_output,
        required=True,
        help="The output directory to save the quantized model weight, "
        "will contain `params_shard_*.bin` and `ndarray-cache.json`.",
    )

    # parse arguments
    parsed = parser.parse_args()
    parsed.source = _parse_source(parsed.source, parsed.config)
    parsed.params, parsed.source_format = detect_weight(
        parsed.source, parsed.config, weight_format=parsed.source_format
    )
    model = detect_model_type(parsed.model_type, parsed.config)

    # detect quantization target
    quantization_target, _ = detect_target_and_host(parsed.device)
    if parsed.device != "auto":
        device = tvm.runtime.device(parsed.device.split(" ")[0])
    else:
        if quantization_target.kind.name == "cuda":
            device = tvm.cuda(0)
        else:
            device = tvm.cpu(0)

    # model config & quantization config
    model_config = model.config.from_file(parsed.config)
    quantization_config = QUANTIZATION[parsed.quantization]
    _, quantize_map = model.quantize[quantization_config.kind](model_config, quantization_config)

    # loader setup
    if parsed.source_format in ("huggingface-torch", "huggingface-safetensor"):
        loader = HuggingFaceLoader(
            path=parsed.params,
            extern_param_map=model.source[parsed.source_format](model_config, None),
            quantize_param_map=quantize_map,
        )
    else:
        raise ValueError(f"Unsupported loader source format: {parsed.source_format}")

    # load and quantize
    with quantization_target, tqdm.redirect():
        param_dict = dict(loader.load(device=device))

    # dump to output directory
    tvmjs.dump_ndarray_cache(
        param_dict,
        f"{parsed.output}/params",
        meta_data={"ParamSize": len(param_dict)},
        encode_format="raw",
    )


if __name__ == "__main__":
    main()
