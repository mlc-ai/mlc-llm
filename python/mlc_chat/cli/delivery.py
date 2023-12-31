"""Continuous model delivery for MLC LLM models."""
import argparse
import dataclasses
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

from huggingface_hub import HfApi  # pylint: disable=import-error
from huggingface_hub.utils import HfHubHTTPError  # pylint: disable=import-error

from mlc_chat.support import logging
from mlc_chat.support.argparse import ArgumentParser
from mlc_chat.support.constants import MLC_TEMP_DIR
from mlc_chat.support.download import git_clone
from mlc_chat.support.style import bold, green, red

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
    context_window_size: int = None
    sliding_window_size: int = None
    prefill_chunk_size: int = None
    attention_sink_size: int = None
    tensor_parallel_shards: int = None


class DeferredScope:
    """A context manager that defers execution of functions until exiting the scope."""

    def __init__(self):
        self.deferred_functions = []

    def add(self, func: Callable[[], None]):
        """Add a function to be executed when exiting the scope."""
        self.deferred_functions.append(func)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for func in reversed(self.deferred_functions):
            func()
        return False

    def create_temp_dir(self) -> Path:
        """Create a temporary directory that will be deleted when exiting the scope."""
        temp_dir = tempfile.mkdtemp(dir=MLC_TEMP_DIR)
        self.add(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        return Path(temp_dir)


def _clone_repo(model: Union[str, Path], deferred: DeferredScope) -> Path:
    if isinstance(model, Path):
        if not model.exists():
            raise ValueError(f"Invalid model source: {model}")
        return model
    if model.startswith("https://") or model.startswith("git://"):
        result = deferred.create_temp_dir() / "repo"
        git_clone(model, result, ignore_lfs=False)
        return result
    result = Path(model)
    if result.exists():
        return result
    raise ValueError(f"Invalid model source: {model}")


def _run_quantization(
    model_info: ModelInfo,
    repo: str,
    api: HfApi,
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
    with tempfile.TemporaryDirectory(dir=MLC_TEMP_DIR) as output_dir:
        log_path = Path(output_dir) / "logs.txt"
        with log_path.open("a", encoding="utf-8") as log_file:
            assert isinstance(model_info.model, Path)
            logger.info("[MLC] Processing in directory: %s", output_dir)
            # Required arguments
            cmd = [
                sys.executable,
                "-m",
                "mlc_chat",
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
            subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT)
            cmd = [
                sys.executable,
                "-m",
                "mlc_chat",
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
            subprocess.run(cmd, check=False, stdout=log_file, stderr=subprocess.STDOUT)
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
):
    failed_cases: List[Tuple[str, str]] = []
    for task_index, task in enumerate(spec["tasks"], 1):
        with DeferredScope() as deferred:
            logger.info(
                bold("[{task_index}/{total_tasks}] Processing model: ").format(
                    task_index=task_index,
                    total_tasks=len(spec["tasks"]),
                )
                + green(task["model_id"])
            )
            model = _clone_repo(task["model"], deferred)
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
                with DeferredScope() as inner_deferred:
                    model_info["model"] = _clone_repo(model_info["model"], inner_deferred)
                    result = _run_quantization(
                        ModelInfo(**model_info),
                        repo=spec["destination"].format(
                            username=username,
                            model_id=model_info["model_id"],
                            quantization=model_info["quantization"],
                        ),
                        api=api,
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
    parsed = parser.parse_args()
    _main(
        parsed.username,
        spec=parsed.spec,
        api=HfApi(token=parsed.token),
    )


if __name__ == "__main__":
    main()
