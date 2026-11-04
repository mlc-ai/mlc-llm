"""Unit tests for single-task embedding serve dispatch."""

# pylint: disable=missing-function-docstring,protected-access
import json
from pathlib import Path

import pytest

from mlc_llm.interface import serve as serve_interface
from mlc_llm.serve.entrypoints import openai_entrypoints
from mlc_llm.support.auto_config import detect_model_task, detect_model_task_and_config


def _write_model_dir(tmp_path: Path, config: dict) -> Path:
    (tmp_path / "mlc-chat-config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


def _serve_kwargs(model: str, **overrides):
    kwargs = dict(
        model=model,
        device="auto",
        model_lib="dummy-lib.so",
        mode="local",
        enable_debug=False,
        additional_models=[],
        embedding_model=None,
        embedding_model_lib=None,
        tensor_parallel_shards=None,
        pipeline_parallel_stages=None,
        opt=None,
        max_num_sequence=None,
        max_total_sequence_length=None,
        max_single_sequence_length=None,
        prefill_chunk_size=None,
        sliding_window_size=None,
        attention_sink_size=None,
        max_history_size=None,
        gpu_memory_utilization=None,
        speculative_mode="disable",
        spec_draft_length=None,
        spec_tree_width=None,
        prefix_cache_mode="radix",
        prefix_cache_max_num_recycling_seqs=None,
        prefill_mode="hybrid",
        enable_tracing=False,
        host="127.0.0.1",
        port=8000,
        allow_credentials=True,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        api_key=None,
    )
    kwargs.update(overrides)
    return kwargs


def test_detect_model_task_embedding(tmp_path):
    model_dir = _write_model_dir(tmp_path, {"model_task": "embedding"})
    assert detect_model_task(str(model_dir)) == "embedding"


def test_detect_model_task_defaults_to_chat(tmp_path):
    model_dir = _write_model_dir(tmp_path, {"model_type": "llama"})
    assert detect_model_task(str(model_dir)) == "chat"


def test_detect_model_task_and_config_returns_config_path(tmp_path):
    model_dir = _write_model_dir(tmp_path, {"model_task": "embedding"})
    task, config_path = detect_model_task_and_config(str(model_dir))
    assert task == "embedding"
    assert config_path == model_dir / "mlc-chat-config.json"


def test_embedding_app_exposes_only_embedding_endpoints():
    routes = {
        (method, route.path)
        for route in openai_entrypoints.embedding_app.routes
        for method in route.methods
    }
    assert routes == {("GET", "/v1/models"), ("POST", "/v1/embeddings")}


def test_chat_serve_router_keeps_all_endpoints():
    routes = {
        (method, route.path) for route in openai_entrypoints.app.routes for method in route.methods
    }
    assert ("POST", "/v1/embeddings") in routes
    assert ("POST", "/v1/completions") in routes
    assert ("POST", "/v1/chat/completions") in routes
    assert ("GET", "/v1/models") in routes


def test_serve_dispatches_embedding_model(tmp_path, monkeypatch):
    model_dir = _write_model_dir(tmp_path, {"model_task": "embedding"})
    called = {}

    monkeypatch.setattr(serve_interface, "_serve_embedding", called.update)
    monkeypatch.setattr(
        serve_interface.engine,
        "AsyncMLCEngine",
        lambda *a, **k: pytest.fail("chat engine must not be created for embedding models"),
    )

    serve_interface.serve(**_serve_kwargs(str(model_dir)))

    assert called["model"] == str(model_dir)
    assert called["model_path"] == str(model_dir)
    assert called["model_lib"] == "dummy-lib.so"


def test_serve_rejects_sidecar_flags_with_embedding_primary(tmp_path, monkeypatch):
    model_dir = _write_model_dir(tmp_path, {"model_task": "embedding"})
    monkeypatch.setattr(
        serve_interface.engine,
        "AsyncMLCEngine",
        lambda *a, **k: pytest.fail("chat engine must not be created"),
    )
    with pytest.raises(ValueError, match="--embedding-model"):
        serve_interface.serve(**_serve_kwargs(str(model_dir), embedding_model="some-model"))


def test_serve_embedding_requires_model_lib(tmp_path):
    model_dir = _write_model_dir(tmp_path, {"model_task": "embedding"})
    with pytest.raises(ValueError, match="--model-lib"):
        serve_interface.serve(**_serve_kwargs(str(model_dir), model_lib=None))


class _ChatPathTaken(Exception):
    """Sentinel raised by the fake chat engine to prove the chat path was taken."""


def test_serve_dispatches_chat_model(tmp_path, monkeypatch):
    model_dir = _write_model_dir(tmp_path, {"model_type": "llama"})

    def _fake_engine(*_args, **_kwargs):
        raise _ChatPathTaken

    monkeypatch.setattr(serve_interface.engine, "AsyncMLCEngine", _fake_engine)
    monkeypatch.setattr(
        serve_interface,
        "_serve_embedding",
        lambda **k: pytest.fail("embedding path must not be taken for chat models"),
    )
    with pytest.raises(_ChatPathTaken):
        serve_interface.serve(**_serve_kwargs(str(model_dir)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
