"""Embedding server endpoint tests in MLC LLM.

Tests the /v1/embeddings endpoint via HTTP using the OpenAI client,
following the same patterns as test_server.py.

Reuses MLC LLM test infrastructure:
  - Pytest markers (endpoint)
  - expect_error() response validation pattern from test_server.py
  - OpenAI client usage pattern from test_server.py
  - Session-scoped server fixture pattern from conftest.py

Run (launches its own embedding-only server via ``mlc_llm serve``):
  MLC_SERVE_EMBEDDING_MODEL_LIB="path/to/model.dylib" \
    pytest -m endpoint tests/python/serve/server/test_embedding_server.py -v

Environment variables:
  MLC_SERVE_EMBEDDING_MODEL_LIB  Path to compiled embedding model library (required)
  MLC_SERVE_EMBEDDING_MODEL      Path to embedding model weight directory
                                  (optional, defaults to dirname of model lib)
"""

# pylint: disable=redefined-outer-name

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import psutil
import pytest
import requests
from openai import OpenAI

# Reuse MLC LLM marker system
pytestmark = [pytest.mark.endpoint]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_LIB = os.environ.get("MLC_SERVE_EMBEDDING_MODEL_LIB")
EMBEDDING_MODEL_DIR = os.environ.get(
    "MLC_SERVE_EMBEDDING_MODEL",
    os.path.dirname(EMBEDDING_MODEL_LIB) if EMBEDDING_MODEL_LIB else None,
)
EMBEDDING_SERVER_HOST = "127.0.0.1"
EMBEDDING_SERVER_PORT = 8321
EMBEDDING_BASE_URL = f"http://{EMBEDDING_SERVER_HOST}:{EMBEDDING_SERVER_PORT}/v1"
EMBEDDING_MODEL_NAME = "embedding"


def _skip_if_no_model():
    if EMBEDDING_MODEL_LIB is None:
        pytest.skip(
            'Environment variable "MLC_SERVE_EMBEDDING_MODEL_LIB" not found. '
            "Set it to a compiled embedding model library."
        )
    if not os.path.isfile(EMBEDDING_MODEL_LIB):
        pytest.skip(f"Embedding model library not found at: {EMBEDDING_MODEL_LIB}")
    if EMBEDDING_MODEL_DIR is None or not os.path.isdir(EMBEDDING_MODEL_DIR):
        pytest.skip(f"Embedding model directory not found at: {EMBEDDING_MODEL_DIR}")


# ---------------------------------------------------------------------------
# Response validation helpers — adapted from test_server.py patterns
# ---------------------------------------------------------------------------


def check_embedding_response(
    response: Dict,
    *,
    model: str,
    num_embeddings: int,
    expected_dim: Optional[int] = None,
    check_unit_norm: bool = True,
):
    """Validate an OpenAI-compatible embedding response.

    Adapted from check_openai_nonstream_response() in test_server.py,
    specialized for embedding responses.
    """
    assert response["object"] == "list"
    assert response["model"] == model

    data = response["data"]
    assert isinstance(data, list)
    assert len(data) == num_embeddings

    for item in data:
        assert item["object"] == "embedding"
        assert isinstance(item["index"], int)
        emb = item["embedding"]
        assert isinstance(emb, list)
        assert len(emb) > 0

        if expected_dim is not None:
            assert len(emb) == expected_dim, f"Expected dim={expected_dim}, got {len(emb)}"

        if check_unit_norm:
            norm = float(np.linalg.norm(emb))
            assert abs(norm - 1.0) < 1e-3, f"Expected unit norm, got {norm}"

    # Usage validation — same pattern as test_server.py
    usage = response["usage"]
    assert isinstance(usage, dict)
    assert usage["prompt_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"]


def expect_error(response_str: str, msg_prefix: Optional[str] = None):
    """Validate error response — reused directly from test_server.py."""
    response = json.loads(response_str)
    assert response["object"] == "error"
    assert isinstance(response["message"], str)
    if msg_prefix is not None:
        assert response["message"].startswith(msg_prefix)


# ---------------------------------------------------------------------------
# Server fixture — uses the proper ``mlc_llm serve`` CLI path
# ---------------------------------------------------------------------------


def _terminate_proc(proc):
    """Terminate a subprocess and all its children (same pattern as PopenServer)."""
    try:
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


@pytest.fixture(scope="module")
def launch_embedding_server():
    """Launch an embedding-only server via ``mlc_llm serve``.

    Uses the proper serve path so that model_task detection in
    interface/serve.py routes to the embedding-only app automatically.
    Follows the same lifecycle pattern as PopenServer.
    """
    _skip_if_no_model()

    cmd = [
        sys.executable,
        "-m",
        "mlc_llm",
        "serve",
        EMBEDDING_MODEL_DIR,
        "--model-lib",
        EMBEDDING_MODEL_LIB,
        "--device",
        "auto",
        "--host",
        EMBEDDING_SERVER_HOST,
        "--port",
        str(EMBEDDING_SERVER_PORT),
    ]

    process_path = str(Path(__file__).resolve().parents[4])
    proc = subprocess.Popen(cmd, cwd=process_path)  # pylint: disable=consider-using-with

    # Wait for server readiness — same polling pattern as PopenServer.start()
    timeout = 120
    attempts = 0.0
    ready = False
    while attempts < timeout:
        try:
            response = requests.get(f"{EMBEDDING_BASE_URL}/models", timeout=2)
            if response.status_code == 200:
                ready = True
                break
        except requests.RequestException:
            pass
        attempts += 0.5
        time.sleep(0.5)

    if not ready:
        _terminate_proc(proc)
        raise RuntimeError(f"Embedding server failed to start in {timeout}s.")

    yield proc

    _terminate_proc(proc)


@pytest.fixture(scope="module")
def client(launch_embedding_server):
    """OpenAI client connected to the embedding server."""
    assert launch_embedding_server is not None
    return OpenAI(base_url=EMBEDDING_BASE_URL, api_key="none")


# ===================================================================
# /v1/models
# ===================================================================


@pytest.mark.usefixtures("client")
def test_models_endpoint():
    """The /v1/models endpoint lists the embedding model."""
    resp = requests.get(f"{EMBEDDING_BASE_URL}/models", timeout=5)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["data"], list)


# ===================================================================
# Single input
# ===================================================================


def test_single_string_input(client):
    """Single string input returns one embedding."""
    resp = client.embeddings.create(input="What is machine learning?", model=EMBEDDING_MODEL_NAME)
    raw = resp.model_dump()
    check_embedding_response(raw, model=EMBEDDING_MODEL_NAME, num_embeddings=1)


# ===================================================================
# Batch input
# ===================================================================

BATCH_INPUTS = [
    "What is machine learning?",
    "How to brew coffee?",
    "ML is a subset of AI.",
]


def test_batch_string_input(client):
    """List of strings returns one embedding per input."""
    resp = client.embeddings.create(input=BATCH_INPUTS, model=EMBEDDING_MODEL_NAME)
    raw = resp.model_dump()
    check_embedding_response(raw, model=EMBEDDING_MODEL_NAME, num_embeddings=len(BATCH_INPUTS))


def test_batch_index_ordering(client):
    """Embedding indices are sequential."""
    resp = client.embeddings.create(input=BATCH_INPUTS, model=EMBEDDING_MODEL_NAME)
    indices = [d.index for d in resp.data]
    assert indices == list(range(len(BATCH_INPUTS)))


# ===================================================================
# Cosine similarity — semantic quality via endpoint
# ===================================================================


def test_cosine_similarity_via_endpoint(client):
    """Related texts have higher similarity than unrelated (end-to-end)."""
    resp = client.embeddings.create(
        input=[
            "What is machine learning?",
            "Explain deep learning",
            "Order a pizza",
        ],
        model=EMBEDDING_MODEL_NAME,
    )
    e0, e1, e2 = [np.array(d.embedding) for d in resp.data]
    sim_related = float(np.dot(e0, e1))
    sim_unrelated = float(np.dot(e0, e2))
    assert (
        sim_related > sim_unrelated
    ), f"Related ({sim_related:.4f}) should > unrelated ({sim_unrelated:.4f})"


# ===================================================================
# Dimension truncation (Matryoshka)
# ===================================================================


def test_dimension_truncation(client):
    """dimensions parameter truncates and re-normalizes output."""
    target_dim = 256
    resp = client.embeddings.create(
        input="Hello world", model=EMBEDDING_MODEL_NAME, dimensions=target_dim
    )
    raw = resp.model_dump()
    check_embedding_response(
        raw,
        model=EMBEDDING_MODEL_NAME,
        num_embeddings=1,
        expected_dim=target_dim,
    )


# ===================================================================
# Encoding format
# ===================================================================


@pytest.mark.usefixtures("launch_embedding_server")
def test_base64_encoding():
    """base64 encoding format returns base64-encoded embeddings."""
    resp = requests.post(
        f"{EMBEDDING_BASE_URL}/embeddings",
        json={
            "input": "Hello world",
            "model": EMBEDDING_MODEL_NAME,
            "encoding_format": "base64",
        },
        timeout=5,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["data"][0]["object"] == "embedding"
    # base64 string should be a non-empty string (not a list)
    emb = data["data"][0]["embedding"]
    assert isinstance(emb, str) and len(emb) > 0


# ===================================================================
# Error handling — reuses expect_error() pattern from test_server.py
# ===================================================================


@pytest.mark.usefixtures("launch_embedding_server")
def test_any_model_name_works_with_single_engine():
    """When only one embedding engine is served, any model name works.

    This mirrors ServerContext.get_engine() behavior: a single served
    model is returned regardless of the requested model name.
    """
    resp = requests.post(
        f"{EMBEDDING_BASE_URL}/embeddings",
        json={"input": "test", "model": "any-name-works"},
        timeout=5,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["data"]) == 1


# ===================================================================
# Chat-only endpoints must NOT be registered (404)
# ===================================================================

_BASE = f"http://{EMBEDDING_SERVER_HOST}:{EMBEDDING_SERVER_PORT}"


@pytest.mark.usefixtures("launch_embedding_server")
def test_metrics_not_registered():
    """/metrics must return 404 for embedding-only serve."""
    resp = requests.get(f"{_BASE}/metrics", timeout=5)
    assert resp.status_code == 404


@pytest.mark.usefixtures("launch_embedding_server")
def test_debug_dump_event_trace_not_registered():
    """/debug/dump_event_trace must return 404 for embedding-only serve."""
    resp = requests.post(f"{_BASE}/debug/dump_event_trace", timeout=5)
    assert resp.status_code == 404


@pytest.mark.usefixtures("launch_embedding_server")
def test_completions_not_registered():
    """/v1/completions must return 404 for embedding-only serve."""
    resp = requests.post(
        f"{EMBEDDING_BASE_URL}/completions",
        json={"model": "x", "prompt": "hi"},
        timeout=5,
    )
    assert resp.status_code == 404


@pytest.mark.usefixtures("launch_embedding_server")
def test_chat_completions_not_registered():
    """/v1/chat/completions must return 404 for embedding-only serve."""
    resp = requests.post(
        f"{EMBEDDING_BASE_URL}/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        timeout=5,
    )
    assert resp.status_code == 404


# ===================================================================
# Standalone runner (same pattern as test_server.py __main__)
# ===================================================================

if __name__ == "__main__":
    _skip_if_no_model()

    print(f"Using model: {EMBEDDING_MODEL_DIR}")
    print(f"Using model lib: {EMBEDDING_MODEL_LIB}")
    print(f"Server URL: {EMBEDDING_BASE_URL}")
    print(
        "\nMake sure the embedding server is running, or set env vars "
        "and use pytest to auto-launch."
    )

    # Allow running against an already-running server
    c = OpenAI(base_url=EMBEDDING_BASE_URL, api_key="none")
    test_models_endpoint()
    test_single_string_input(c)
    test_batch_string_input(c)
    test_batch_index_ordering(c)
    test_cosine_similarity_via_endpoint(c)
    test_dimension_truncation(c)
    test_base64_encoding()
    test_any_model_name_works_with_single_engine()
    test_metrics_not_registered()
    test_debug_dump_event_trace_not_registered()
    test_completions_not_registered()
    test_chat_completions_not_registered()
    print("\nAll embedding server tests passed!")
