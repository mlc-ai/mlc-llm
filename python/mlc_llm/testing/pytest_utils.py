"""Extra utilities to mark tests"""
import functools
from typing import Callable

import pytest

from mlc_llm.support.constants import MLC_TEST_MODEL_PATH


def require_test_model(model: str):
    """Testcase decorator to require a model

    Examples
    --------
    .. code::

        @require_test_model("Llama-2-7b-chat-hf-q4f16_1-MLC")
        def test_reload_reset_unload(model):
            # model now points to the right path
            # specified by MLC_TEST_MODEL_PATH
            engine = mlc_llm.MLCEngine(model)
            # test code follows

    Parameters
    ----------
    model : str
        The model dir name
    """
    model_path = None
    for base_path in MLC_TEST_MODEL_PATH:
        if (base_path / model / "mlc-chat-config.json").is_file():
            model_path = base_path / model
    missing_model = model_path is None
    message = (
        f"Model {model} does not exist in candidate paths {[str(p) for p in MLC_TEST_MODEL_PATH]},"
        " if you set MLC_TEST_MODEL_PATH, please ensure model paths are in the right location,"
        " by default we reuse cache, try to run mlc_llm chat to download right set of models."
    )

    def _decorator(func: Callable[[str], None]):
        wrapped = functools.partial(func, str(model_path))
        wrapped.__name__ = func.__name__  # type: ignore

        @functools.wraps(wrapped)
        def wrapper(*args, **kwargs):
            if missing_model:
                print(f"{message} skipping...")
                return
            wrapped(*args, **kwargs)

        return pytest.mark.skipif(missing_model, reason=message)(wrapper)

    return _decorator
