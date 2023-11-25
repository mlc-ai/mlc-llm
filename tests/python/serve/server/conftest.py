# pylint: disable=missing-module-docstring,missing-function-docstring
# pylint: disable=redefined-outer-name,consider-using-with
import os
import subprocess
import time
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def served_model() -> str:
    model = os.environ.get("MLC_SERVE_MODEL")
    if model is None:
        raise ValueError(
            'Environment variable "MLC_SERVE_MODEL" not found. '
            "Please set it to model compiled by MLC LLM (e.g., Llama-2-7b-chat-hf-q0f16)."
        )
    return model


@pytest.fixture(scope="session")
def launch_server(served_model):
    """A pytest session-level fixture which launches the server in a subprocess."""

    # Start your subprocess here
    cmd = ["python"]
    cmd += ["-m", "mlc_chat.serve.server"]
    cmd += ["--model", f"{served_model}"]
    cmd += ["--max-total-seq-length", "5120"]
    process_path = str(Path(__file__).resolve().parents[4])
    process = subprocess.Popen(cmd, cwd=process_path)
    # NOTE: DO NOT USE `stdout=subprocess.PIPE, stderr=subprocess.PIPE`
    # in subprocess.Popen here. PIPE may conflict with logging in TVM
    # and cause subprocess hangs forever.

    # Wait for the subprocess to be ready. This may take a while.
    time.sleep(20)

    process_return_code = process.poll()
    if process_return_code is not None:
        raise RuntimeError(
            "The server fails to launch. "
            f"Please check if {served_model} is a valid model compiled by MLC LLM."
        )
    yield

    # Fixture teardown code.
    process.terminate()
    process.wait()
