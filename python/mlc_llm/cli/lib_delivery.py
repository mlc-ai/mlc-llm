"""Continuous model delivery for MLC LLM models."""

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List

from mlc_llm.support import logging
from mlc_llm.support.argparse import ArgumentParser
from mlc_llm.support.constants import MLC_TEMP_DIR
from mlc_llm.support.style import bold, green, red

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ModelInfo:  # pylint: disable=too-many-instance-attributes
    """Necessary information for the model delivery"""

    model_id: str
    model: Path
    quantization: str
    device: str
    # overrides the `context_window_size`, `prefill_chunk_size`,
    # `sliding_window_size`, `attention_sink_size`, `max_batch_size`
    # and `tensor_parallel_shards in mlc-chat-config.json
    overrides: Dict[str, int]


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


def _run_compilation(model_info: ModelInfo, repo_dir: Path) -> bool:
    """Run the compilation of the model library."""

    def get_lib_ext(device: str) -> str:
        if device in ["cuda", "vulkan", "metal"]:
            return ".so"
        if device in ["android", "ios"]:
            return ".tar"
        if device in ["webgpu"]:
            return ".wasm"

        return ""

    succeeded = True
    with tempfile.TemporaryDirectory(dir=MLC_TEMP_DIR) as temp_dir:
        log_path = Path(temp_dir) / "logs.txt"
        model_lib_name = f"{model_info.model_id}-{model_info.quantization}-{model_info.device}"
        lib_ext = get_lib_ext(model_info.device)
        if lib_ext == "":
            raise ValueError(f"Unsupported device: {model_info.device}")
        model_lib_name += lib_ext
        with log_path.open("a", encoding="utf-8") as log_file:
            overrides = ";".join(f"{key}={value}" for key, value in model_info.overrides.items())
            cmd = [
                sys.executable,
                "-m",
                "mlc_llm",
                "compile",
                str(model_info.model),
                "--device",
                model_info.device,
                "--quantization",
                model_info.quantization,
                "--overrides",
                overrides,
                "--output",
                os.path.join(temp_dir, model_lib_name),
            ]
            print(" ".join(cmd), file=log_file, flush=True)
            subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT)
            logger.info("[MLC] Compilation Complete!")
        if not (Path(temp_dir) / model_lib_name).exists():
            logger.error(
                "[%s] Model %s. Device %s. No compiled library found.",
                red("FAILED"),
                model_info.model_id,
                model_info.device,
            )
            succeeded = False
            return succeeded

        # overwrite git repo file with the compiled library
        repo_filepath = repo_dir / model_info.model_id / model_lib_name
        if not repo_filepath.parent.exists():
            repo_filepath.parent.mkdir(parents=True, exist_ok=True)
        # copy lib from Path(temp_dir) / model_lib_name to repo_filepath
        shutil.copy(Path(temp_dir) / model_lib_name, repo_filepath)
        logger.info("Saved library %s at %s", model_lib_name, repo_filepath)
    return succeeded


def _main(  # pylint: disable=too-many-locals
    spec: Dict[str, Any],
):
    """Compile the model libs in the spec and save them to the binary_libs_dir."""
    failed_cases: List[Any] = []
    for task_index, task in enumerate(spec["tasks"], 1):
        logger.info(  # pylint: disable=logging-not-lazy
            bold("[{task_index}/{total_tasks}] Processing model: ").format(
                task_index=task_index,
                total_tasks=len(spec["tasks"]),
            )
            + green(task["model_id"])
        )
        model_info = {
            "model_id": task["model_id"],
            "model": task["model"],
        }
        for compile_opt in spec["default_compile_options"] + task.get("compile_options", []):
            for quantization in spec["default_quantization"] + task.get("quantization", []):
                model_info["quantization"] = quantization
                model_info["device"] = compile_opt["device"]
                model_info["overrides"] = compile_opt.get("overrides", {})
                logger.info(
                    "[Config] "
                    + bold("model_id: ")
                    + model_info["model_id"]
                    + bold(", quantization: ")
                    + model_info["quantization"]
                    + bold(", device: ")
                    + model_info["device"]
                    + bold(", overrides: ")
                    + json.dumps(model_info["overrides"])
                )

                result = _run_compilation(
                    ModelInfo(**model_info),
                    repo_dir=Path(spec["binary_libs_dir"]),
                )
                if not result:
                    failed_cases.append(model_info)

    if failed_cases:
        logger.info("Total %s %s:", len(failed_cases), red("failures"))
        for case in failed_cases:
            logger.info(
                "model_id %s, quantization %s, device %s, overrides %s",
                case["model_id"],
                case["quantization"],
                case["device"],
                json.dumps(case["overrides"]),
            )


def main():
    """Entry point."""

    def _load_spec(path_spec: str) -> Dict[str, Any]:
        path = Path(path_spec)
        if not path.exists():
            raise argparse.ArgumentTypeError(f"Spec file does not exist: {path}")
        with path.open("r", encoding="utf-8") as i_f:
            return json.load(i_f)

    parser = ArgumentParser("MLC LLM continuous library delivery")
    parser.add_argument(
        "--spec",
        type=_load_spec,
        required=True,
        help="Path to the spec file",
    )
    parsed = parser.parse_args()
    _main(
        spec=parsed.spec,
    )


if __name__ == "__main__":
    main()
