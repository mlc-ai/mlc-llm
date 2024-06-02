"""Continuous model delivery for MLC LLM models."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from huggingface_hub import HfApi, snapshot_download  # pylint: disable=import-error
from huggingface_hub.utils import HfHubHTTPError  # pylint: disable=import-error
from pydantic import BaseModel, Field, ValidationError

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

T = TypeVar("T", bound="BaseModel")


class ModelDeliveryTask(BaseModel):
    """
    Example:
    {
        "model_id": "Phi-3-mini-128k-instruct",
        "model": "HF://microsoft/Phi-3-mini-128k-instruct",
        "conv_template": "phi-3",
        "quantization": ["q3f16_1"],
        "context_window_size": 4096
    }
    """

    model_id: str
    model: str
    conv_template: str
    quantization: Optional[Union[List[str], str]] = Field(default_factory=list)
    destination: Optional[str] = None

    context_window_size: Optional[int] = None
    sliding_window_size: Optional[int] = None
    prefill_chunk_size: Optional[int] = None
    attention_sink_size: Optional[int] = None
    tensor_parallel_shards: Optional[int] = None


class ModelDeliveryList(BaseModel):
    """
    The class that specifies the model delivery list.
    """

    tasks: List[ModelDeliveryTask]
    # For delivered log, the default destination and quantization fields are optional
    default_destination: Optional[str] = None
    default_quantization: Optional[List[str]] = None

    @classmethod
    def from_json(cls: Type[T], json_dict: Dict[str, Any]) -> T:
        """
        Convert from a json dictionary.
        """
        try:
            return ModelDeliveryList.model_validate(json_dict)
        except ValidationError as e:
            logger.error("Error validating ModelDeliveryList: %s", e)
            raise e

    def to_json(self) -> Dict[str, Any]:
        """
        Convert to a json dictionary.
        """
        return self.model_dump(exclude_none=True)


def _clone_repo(model: Union[str, Path], hf_local_dir: Optional[str]) -> str:
    if isinstance(model, Path):
        if not model.exists():
            raise ValueError(f"Invalid model source: {model}")
        return str(model)
    prefixes, mlc_prefix = ["HF://", "https://huggingface.co/"], ""
    mlc_prefix = next(p for p in prefixes if model.startswith(p))
    if mlc_prefix:
        repo_name = model[len(mlc_prefix) :]
        model_name = repo_name.split("/")[-1]
        if hf_local_dir:
            hf_local_dir = os.path.join(hf_local_dir, model_name)
            logger.info("[HF] Downloading model to %s", hf_local_dir)
        return snapshot_download(repo_id=repo_name, local_dir=hf_local_dir)
    result = Path(model)
    if result.exists():
        return model
    raise ValueError(f"Invalid model source: {model}")


def _run_quantization(
    model_info: ModelDeliveryTask,
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
        assert isinstance(model_info.quantization, str)
        logger.info("[MLC] Processing in directory: %s", output_dir)
        # Required arguments
        cmd = [
            sys.executable,
            "-m",
            "mlc_llm",
            "gen_config",
            model_info.model,
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


def _get_current_log(log: str) -> ModelDeliveryList:
    log_path = Path(log)
    if not log_path.exists():
        with log_path.open("w", encoding="utf-8") as o_f:
            current_log = ModelDeliveryList(tasks=[])
            json.dump(current_log.to_json(), o_f, indent=4)
    else:
        with log_path.open("r", encoding="utf-8") as i_f:
            current_log = ModelDeliveryList.from_json(json.load(i_f))
    return current_log


def _main(  # pylint: disable=too-many-locals, too-many-arguments
    username: str,
    api: HfApi,
    spec: ModelDeliveryList,
    log: str,
    hf_local_dir: Optional[str],
    output: str,
):
    failed_cases: List[Tuple[str, str]] = []
    delivered_log = _get_current_log(log)
    for task_index, task in enumerate(spec.tasks, 1):
        logger.info(
            bold("[{task_index}/{total_tasks}] Processing model: ").format(
                task_index=task_index,
                total_tasks=len(spec.tasks),
            )
            + green(task.model_id)
        )
        model = _clone_repo(task.model, hf_local_dir)

        quantizations = []

        if spec.default_quantization:
            quantizations += spec.default_quantization

        if task.quantization:
            if isinstance(task.quantization, str):
                quantizations.append(task.quantization)
            else:
                quantizations += task.quantization

        default_destination = spec.default_destination or "{username}/{model_id}-{quantization}-MLC"
        for quantization in quantizations:
            repo = default_destination.format(
                username=username,
                model_id=task.model_id,
                quantization=quantization,
            )
            model_info = ModelDeliveryTask(
                model=model,
                quantization=quantization,
                destination=repo,
                **task.model_dump(exclude_none=True, exclude={"model", "quantization"}),
            )
            logger.info("Model info: %s", model_info.model_dump_json(indent=4))
            output_dir = os.path.join(
                output, f"{model_info.model_id}-{model_info.quantization}-MLC"
            )
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            result = _run_quantization(
                model_info=model_info,
                repo=repo,
                api=api,
                output_dir=output_dir,
            )
            if not result:
                failed_cases.append(
                    (task.model_id, quantization),
                )
            else:
                delivered_log.tasks.append(model_info)
    if failed_cases:
        logger.info("Total %s %s:", len(failed_cases), red("failures"))
        for model_id, quantization in failed_cases:
            logger.info("  Model %s. Quantization %s.", model_id, quantization)

    logger.info("Writing log to %s", log)
    with open(log, "w", encoding="utf-8") as o_f:
        json.dump(delivered_log.to_json(), o_f, indent=4)


def main():
    """Entry point."""

    def _load_spec(path_spec: str) -> ModelDeliveryList:
        path = Path(path_spec)
        if not path.exists():
            raise argparse.ArgumentTypeError(f"Spec file does not exist: {path}")
        with path.open("r", encoding="utf-8") as i_f:
            return ModelDeliveryList.from_json(json.load(i_f))

    def _get_default_hf_token() -> str:
        # Try to get the token from the environment variable
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            logger.info("HF token found in environment variable HF_TOKEN")
            return hf_token

        # If not found, look for the token in the default cache folder
        token_file_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(token_file_path):
            with open(token_file_path, "r", encoding="utf-8") as token_file:
                hf_token = token_file.read().strip()
                if hf_token:
                    logger.info("HF token found in ~/.cache/huggingface/token")
                    return hf_token

        raise EnvironmentError("HF token not found")

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
        default=_get_default_hf_token(),
        help="HuggingFace access token, obtained under https://huggingface.co/settings/tokens",
    )
    parser.add_argument(
        "--spec",
        type=_load_spec,
        default="model-delivery-config.json",
        help="Path to the model delivery file" + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--log",
        type=str,
        default="model-delivered-log.json",
        help="Path to the output log file" + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to store the output MLC models",
    )
    parser.add_argument(
        "--hf-local-dir",
        type=str,
        required=False,
        help="Local directory to store the downloaded HuggingFace model",
    )
    parsed = parser.parse_args()
    _main(
        parsed.username,
        spec=parsed.spec,
        log=parsed.log,
        api=HfApi(token=parsed.token),
        hf_local_dir=parsed.hf_local_dir,
        output=parsed.output,
    )


if __name__ == "__main__":
    main()
