"""Unit tests for mlc_llm.cli.serve – single-task serve CLI validation.

No module-level import of mlc_llm — everything is loaded via the ``serve_mod``
fixture.  When tvm is unavailable, the fixture installs minimal stubs scoped
to this module's lifetime and removes them on teardown, so other test files
in the same pytest session are never affected.
"""

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub lists (only used when tvm is absent)
# ---------------------------------------------------------------------------
_TVM_NAMES = (
    "tvm",
    "tvm.relax",
    "tvm.relax.frontend",
    "tvm.relax.frontend.nn",
    "tvm.runtime",
    "tvm.runtime.disco",
)
_MOCK_NAMES = (
    "mlc_llm.serve",
    "mlc_llm.serve.engine",
    "mlc_llm.serve.engine_base",
    "mlc_llm.serve.engine_utils",
    "mlc_llm.serve.embedding_engine",
    "mlc_llm.serve.entrypoints",
    "mlc_llm.serve.entrypoints.debug_entrypoints",
    "mlc_llm.serve.entrypoints.metrics_entrypoints",
    "mlc_llm.serve.entrypoints.microserving_entrypoints",
    "mlc_llm.serve.entrypoints.openai_entrypoints",
    "mlc_llm.serve.server",
    "mlc_llm.serve.config",
    "mlc_llm.protocol",
    "mlc_llm.protocol.error_protocol",
    "mlc_llm.libinfo",
    "fastapi",
    "fastapi.middleware",
    "fastapi.middleware.cors",
    "uvicorn",
)

_MISSING = object()


@pytest.fixture(scope="module")
def serve_mod():
    """Import ``mlc_llm.cli.serve``, installing tvm stubs only when needed.

    Stubs are removed on teardown so they never leak to other modules.
    """
    installed: dict = {}  # name → previous sys.modules value or _MISSING

    try:
        import mlc_llm.cli.serve as mod  # noqa: F811

        yield mod
        return  # no stubs, nothing to clean up
    except ImportError:
        pass

    # --- tvm unavailable: install stubs ---
    tvm_stub = types.ModuleType("tvm")
    tvm_stub.register_global_func = lambda *a, **kw: (lambda fn: fn)

    for name in _TVM_NAMES:
        installed[name] = sys.modules.get(name, _MISSING)
        sys.modules[name] = tvm_stub if name == "tvm" else MagicMock()
    for name in _MOCK_NAMES:
        installed[name] = sys.modules.get(name, _MISSING)
        sys.modules[name] = MagicMock()
    if "mlc_llm.libinfo" in sys.modules:
        sys.modules["mlc_llm.libinfo"].__version__ = "0.0.0-test"

    import mlc_llm.cli.serve as mod  # noqa: F811

    yield mod

    # --- teardown: remove every stub we inserted ---
    # Also remove the module-under-test itself so a later real import
    # is not served the stale, stub-backed module object.
    for key in list(sys.modules):
        if key == "mlc_llm.cli.serve" or key.startswith("mlc_llm.cli.serve."):
            sys.modules.pop(key, None)
    for name, prev in installed.items():
        if prev is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_config(tmp_path, model_task=None, **extra):
    config = {"model_type": "test", **extra}
    if model_task is not None:
        config["model_task"] = model_task
    (tmp_path / "mlc-chat-config.json").write_text(json.dumps(config))
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def embedding_model_dir(tmp_path):
    return _write_config(tmp_path, model_task="embedding")


@pytest.fixture()
def chat_model_dir(tmp_path):
    return _write_config(tmp_path, model_task="chat")


@pytest.fixture()
def default_model_dir(tmp_path):
    return _write_config(tmp_path)


# ---------------------------------------------------------------------------
# _argv_has_option
# ---------------------------------------------------------------------------


class TestArgvHasOption:
    def test_exact_match(self, serve_mod):
        assert serve_mod._argv_has_option(["--mode", "local"], "--mode") is True

    def test_equals_syntax(self, serve_mod):
        assert serve_mod._argv_has_option(["--mode=local"], "--mode") is True

    def test_absent(self, serve_mod):
        assert serve_mod._argv_has_option(["--host", "0.0.0.0"], "--mode") is False

    def test_prefix_does_not_false_positive(self, serve_mod):
        assert serve_mod._argv_has_option(["--model-lib", "foo.so"], "--model") is False


# ---------------------------------------------------------------------------
# _detect_model_task
# ---------------------------------------------------------------------------


class TestDetectModelTask:
    @staticmethod
    def _mock_detect(model_dir):
        return lambda model: Path(model_dir) / "mlc-chat-config.json"

    def test_embedding(self, serve_mod, embedding_model_dir):
        with patch(
            "mlc_llm.support.auto_config.detect_mlc_chat_config",
            side_effect=self._mock_detect(embedding_model_dir),
        ):
            assert serve_mod._detect_model_task(embedding_model_dir) == "embedding"

    def test_chat(self, serve_mod, chat_model_dir):
        with patch(
            "mlc_llm.support.auto_config.detect_mlc_chat_config",
            side_effect=self._mock_detect(chat_model_dir),
        ):
            assert serve_mod._detect_model_task(chat_model_dir) == "chat"

    def test_default_is_chat(self, serve_mod, default_model_dir):
        with patch(
            "mlc_llm.support.auto_config.detect_mlc_chat_config",
            side_effect=self._mock_detect(default_model_dir),
        ):
            assert serve_mod._detect_model_task(default_model_dir) == "chat"


# ---------------------------------------------------------------------------
# Removed args: --embedding-model, --embedding-model-lib
# ---------------------------------------------------------------------------


class TestRemovedArgs:
    def test_embedding_model_rejected(self, serve_mod, chat_model_dir):
        with pytest.raises(SystemExit):
            serve_mod.main([chat_model_dir, "--embedding-model", "/some/path"])

    def test_embedding_model_lib_rejected(self, serve_mod, chat_model_dir):
        with pytest.raises(SystemExit):
            serve_mod.main([chat_model_dir, "--embedding-model-lib", "/some/lib.so"])


# ---------------------------------------------------------------------------
# Embedding model: --model-lib required
# ---------------------------------------------------------------------------


class TestEmbeddingModelLibRequired:
    def test_missing_model_lib_errors(self, serve_mod, embedding_model_dir):
        with patch("mlc_llm.cli.serve._detect_model_task", return_value="embedding"):
            with pytest.raises(SystemExit) as exc_info:
                serve_mod.main([embedding_model_dir])
            assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# Embedding model: forbidden options
# ---------------------------------------------------------------------------


class TestEmbeddingForbiddenOptions:
    @pytest.mark.parametrize(
        "extra_argv",
        [
            ["--mode", "local"],
            ["--speculative-mode", "disable"],
            ["--prefix-cache-mode", "radix"],
            ["--prefill-mode", "hybrid"],
            ["--overrides", "max_num_sequence=1"],
            ["--enable-tracing"],
            ["--enable-debug"],
            ["--additional-models", "/some/model"],
        ],
        ids=[
            "mode",
            "speculative-mode",
            "prefix-cache-mode",
            "prefill-mode",
            "overrides",
            "enable-tracing",
            "enable-debug",
            "additional-models",
        ],
    )
    def test_forbidden_option_rejected(self, serve_mod, embedding_model_dir, extra_argv):
        argv = [embedding_model_dir, "--model-lib", "/fake/lib.so"] + extra_argv
        with patch("mlc_llm.cli.serve._detect_model_task", return_value="embedding"):
            with pytest.raises(SystemExit) as exc_info:
                serve_mod.main(argv)
            assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# Chat model: normal path calls serve()
# ---------------------------------------------------------------------------


class TestChatModelServe:
    @patch("mlc_llm.cli.serve.serve")
    @patch("mlc_llm.cli.serve._detect_model_task", return_value="chat")
    def test_chat_model_calls_serve(self, _mock_task, mock_serve, serve_mod, chat_model_dir):
        serve_mod.main([chat_model_dir])
        mock_serve.assert_called_once()
        kw = mock_serve.call_args.kwargs
        assert kw["model"] == chat_model_dir
        # embedding_model / embedding_model_lib no longer passed (single-task serve)
        assert "embedding_model" not in kw
        assert "embedding_model_lib" not in kw

    @patch("mlc_llm.cli.serve.serve")
    @patch("mlc_llm.cli.serve._detect_model_task", return_value="chat")
    def test_chat_model_with_mode(self, _mock_task, mock_serve, serve_mod, chat_model_dir):
        serve_mod.main([chat_model_dir, "--mode", "server"])
        kw = mock_serve.call_args.kwargs
        assert kw["mode"] == "server"


# ---------------------------------------------------------------------------
# Embedding model: valid args accepted
# ---------------------------------------------------------------------------


class TestEmbeddingValidArgs:
    @patch("mlc_llm.cli.serve.serve")
    @patch("mlc_llm.cli.serve._detect_model_task", return_value="embedding")
    def test_embedding_valid_args_calls_serve(
        self, _mock_task, mock_serve, serve_mod, embedding_model_dir
    ):
        serve_mod.main(
            [
                embedding_model_dir,
                "--model-lib",
                "/fake/lib.so",
                "--device",
                "cuda",
                "--host",
                "0.0.0.0",
                "--port",
                "9000",
                "--api-key",
                "secret",
            ]
        )
        mock_serve.assert_called_once()
        kw = mock_serve.call_args.kwargs
        assert kw["model"] == embedding_model_dir
        assert kw["model_lib"] == "/fake/lib.so"
        assert kw["device"] == "cuda"
        assert kw["host"] == "0.0.0.0"
        assert kw["port"] == 9000
        assert kw["api_key"] == "secret"
