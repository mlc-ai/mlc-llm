# pylint: disable=missing-module-docstring,missing-function-docstring
import os
from typing import Tuple

import pytest

from mlc_chat.serve import PopenServer


@pytest.fixture(scope="session")
def served_model() -> Tuple[str, str]:
    model_lib_path = os.environ.get("MLC_SERVE_MODEL_LIB")
    if model_lib_path is None:
        raise ValueError(
            'Environment variable "MLC_SERVE_MODEL_LIB" not found. '
            "Please set it to model lib compiled by MLC LLM "
            "(e.g., `dist/Llama-2-7b-chat-hf-q0f16-MLC/Llama-2-7b-chat-hf-q0f16-MLC-cuda.so`)."
        )
    model = os.path.dirname(model_lib_path)
    return model, model_lib_path


@pytest.fixture(scope="session")
def launch_server(served_model):  # pylint: disable=redefined-outer-name
    """A pytest session-level fixture which launches the server in a subprocess."""
    server = PopenServer(
        model=served_model[0],
        model_lib_path=served_model[1],
        enable_tracing=True,
    )
    server.start()
    yield

    # Fixture teardown code.
    server.terminate()
