# pylint: disable=missing-module-docstring,missing-function-docstring
import os

import pytest

from mlc_chat.serve import PopenServer


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
def launch_server(served_model):  # pylint: disable=redefined-outer-name
    """A pytest session-level fixture which launches the server in a subprocess."""
    server = PopenServer(
        served_model,
        max_total_sequence_length=5120,
        use_threaded_engine=True,
        enable_tracing=True,
    )
    server.start()
    yield

    # Fixture teardown code.
    server.terminate()
