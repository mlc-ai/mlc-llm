"""Continuous model delivery for MLC LLM models."""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from huggingface_hub import HfApi, snapshot_download  # pylint: disable=import-error
from huggingface_hub.utils import HfHubHTTPError  # pylint: disable=import-error

from mlc_llm.support import logging
from mlc_llm.support.argparse import ArgumentParser
from mlc_llm.support.style import bold, green, red

logging.enable_logging()
logger = logging.getLogger(__name__)

GEN_CONFIG_OPTIONAL_ARGS = [
    "context_window_size",
    "sliding_window_size",
    "prefill_chunk_size",
    "attention_sink_size",
    "tensor_parallel_shards",
]


@dataclasses.dataclass
class ModelInfo:  # pylint: disable=too-many-instance-attributes
    """Necessary information for the model delivery"""

    model_id: str
    model: Path
    conv_template: str
    quantization: str
    source_format: str = "auto"
    # If unspecified in CLI, remains to be None and will not be
    # passed to `gen_config` or `convert_weight`
    context_window_size: Optional[int] = None
    sliding_window_size: Optional[int] = None
    prefill_chunk_size: Optional[int] = None
    attention_sink_size: Optional[int] = None
    tensor_parallel_shards: Optional[int] = None


def _clone_repo(model: Union[str, Path], hf_local_dir: Optional[str]) -> Path:
    if isinstance(model, Path):
        if not model.exists():
            raise ValueError(f"Invalid model source: {model}")
        return model
    prefixes, mlc_prefix = ["HF://", "https://huggingface.co/"], ""
    mlc_prefix = next(p for p in prefixes if model.startswith(p))
    if mlc_prefix:
        repo_name = model[len(mlc_prefix) :]
        model_name = repo_name.split("/")[-1]
        if hf_local_dir:
            hf_local_dir = os.path.join(hf_local_dir, model_name)
            logger.info("[HF] Downloading model to %s", hf_local_dir)
        result = snapshot_download(repo_id=repo_name, local_dir=hf_local_dir)
        return Path(result)
    else:
        result = Path(model)
        if result.exists():
            return result
        raise ValueError(f"Invalid model source: {model}")


def _run_quantization(
    model_info: ModelInfo,
    repo: str,
    api: HfApi,
    output_dir: str,
) -> bool:
    logger.info("[HF] Creating repo https://huggingface.co/%s", repo)
    try:
        api.create_repo(repo_id=repo, private=False)
    except HfHubHTTPError as error:
        if error.response.status_code != 409:
            raise
        logger.info("[HF] Repo already exists. Recreating...")
        api.delete_repo(repo_id=repo)
        api.create_repo(repo_id=repo, private=False)
        logger.info("[HF] Repo recreated")
    succeeded = True
    log_path = Path(output_dir) / "logs.txt"
    with log_path.open("a", encoding="utf-8") as log_file:
        assert isinstance(model_info.model, Path)
        logger.info("[MLC] Processing in directory: %s", output_dir)
        # Required arguments
        cmd = [
            sys.executable,
            "-m",
            "mlc_llm",
            "gen_config",
            str(model_info.model),
            "--quantization",
            model_info.quantization,
            "--conv-template",
            model_info.conv_template,
            "--output",
            output_dir,
        ]
        # Optional arguments
        for optional_arg in GEN_CONFIG_OPTIONAL_ARGS:
            optional_arg_val = getattr(model_info, optional_arg, None)
            if optional_arg_val is not None:
                # e.g. --context-window-size 4096
                cmd += ["--" + optional_arg.replace("_", "-"), str(optional_arg_val)]

        print(" ".join(cmd), file=log_file, flush=True)
        subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT, env=os.environ)
        cmd = [
            sys.executable,
            "-m",
            "mlc_llm",
            "convert_weight",
            str(model_info.model),
            "--quantization",
            model_info.quantization,
            "--source-format",
            model_info.source_format,
            "--output",
            output_dir,
        ]
        print(" ".join(cmd), file=log_file, flush=True)
        subprocess.run(cmd, check=False, stdout=log_file, stderr=subprocess.STDOUT, env=os.environ)
        logger.info("[MLC] Complete!")
    if not (Path(output_dir) / "ndarray-cache.json").exists():
        logger.error(
            "[%s] Model %s. Quantization %s. No weights metadata found.",
            red("FAILED"),
            model_info.model_id,
            model_info.quantization,
        )
        succeeded = False
    logger.info("[HF] Uploading to: https://huggingface.co/%s", repo)
    for _retry in range(10):
        try:
            api.upload_folder(
                folder_path=output_dir,
                repo_id=repo,
                commit_message="Initial commit",
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("[%s] %s. Retrying...", red("FAILED"), exc)
        else:
            break
    else:
        raise RuntimeError("Failed to upload to HuggingFace Hub with 10 retries")
    return succeeded


def _main(  # pylint: disable=too-many-locals
    username: str,
    api: HfApi,
    spec: Dict[str, Any],
    hf_local_dir: Optional[str],
    output: str,
):
    failed_cases: List[Tuple[str, str]] = []
    for task_index, task in enumerate(spec["tasks"], 1):
        logger.info(
            bold("[{task_index}/{total_tasks}] Processing model: ").format(
                task_index=task_index,
                total_tasks=len(spec["tasks"]),
            )
            + green(task["model_id"])
        )
        model = _clone_repo(task["model"], hf_local_dir)
        for quantization in spec["default_quantization"] + task.get("quantization", []):
            model_info = {
                "model_id": task["model_id"],
                "model": model,
                "conv_template": task["conv_template"],
            }
            # Process optional arguments
            for optional_arg in GEN_CONFIG_OPTIONAL_ARGS:
                # e.g. "context_window_size": task.get("context_window_size", None)
                model_info[optional_arg] = task.get(optional_arg, None)
            if isinstance(quantization, str):
                model_info["quantization"] = quantization
            else:
                model_info["quantization"] = quantization.pop("format")
                model_info.update(quantization)
            repo = spec.get("destination", "{username}/{model_id}-{quantization}-MLC").format(
                username=username,
                model_id=model_info["model_id"],
                quantization=model_info["quantization"],
            )
            logger.info(
                "%s%s. %s%s. %s%s",
                bold("Model: "),
                green(task["model_id"]),
                bold("Quantization: "),
                green(model_info["quantization"]),
                bold("Repo: "),
                green(f"https://huggingface.co/{repo}"),
            )
            output_dir = os.path.join(
                output, f"{model_info['model_id']}-{model_info['quantization']}-MLC"
            )
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            result = _run_quantization(
                ModelInfo(**model_info),
                repo=spec["destination"].format(
                    username=username,
                    model_id=model_info["model_id"],
                    quantization=model_info["quantization"],
                ),
                api=api,
                output_dir=output_dir,
            )
            if not result:
                failed_cases.append(
                    (task["model_id"], model_info["quantization"]),
                )
    if failed_cases:
        logger.info("Total %s %s:", len(failed_cases), red("failures"))
        for model_id, quantization in failed_cases:
            logger.info("  Model %s. Quantization %s.", model_id, quantization)


def main():
    """Entry point."""

    def _load_spec(path_spec: str) -> Dict[str, Any]:
        path = Path(path_spec)
        if not path.exists():
            raise argparse.ArgumentTypeError(f"Spec file does not exist: {path}")
        with path.open("r", encoding="utf-8") as i_f:
            return json.load(i_f)

    parser = ArgumentParser("MLC LLM continuous model delivery")
    parser.add_argument(
        "--username",
        type=str,
        required=True,
        help="HuggingFace username",
    )
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="HuggingFace access token, obtained under https://huggingface.co/settings/tokens",
    )
    parser.add_argument(
        "--spec",
        type=_load_spec,
        required=True,
        help="Path to the spec file",
    )
    parser.add_argument(
        "--hf-local-dir",
        type=str,
        required=False,
        help="Local directory to store the HuggingFace model",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory",
    )
    parsed = parser.parse_args()
    _main(
        parsed.username,
        spec=parsed.spec,
        api=HfApi(token=parsed.token),
        hf_local_dir=parsed.hf_local_dir,
        output=parsed.output,
    )


if __name__ == "__main__":
    main()
