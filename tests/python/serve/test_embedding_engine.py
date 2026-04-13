"""Embedding engine tests in MLC LLM.

Tests AsyncEmbeddingEngine for both direct (sync) and async embedding inference.
Reuses MLC LLM test infrastructure: markers, require_test_model pattern,
and conventions from test_serve_engine.py.

Run with real model (requires GPU + compiled embedding model):
  MLC_SERVE_EMBEDDING_MODEL_LIB="path/to/model.dylib" \
    pytest -m engine tests/python/serve/test_embedding_engine.py -v

Environment variables:
  MLC_SERVE_EMBEDDING_MODEL_LIB  Path to compiled embedding model library (required)
  MLC_SERVE_EMBEDDING_MODEL      Path to embedding model weight directory
                                  (optional, defaults to dirname of model lib)
"""

# pylint: disable=import-outside-toplevel,protected-access,redefined-outer-name,possibly-used-before-assignment

import asyncio
import os

import numpy as np
import pytest

# Reuse MLC LLM marker system (registered in tests/python/conftest.py)
pytestmark = [pytest.mark.engine]

# ---------------------------------------------------------------------------
# Fixtures — follows pattern from serve/server/conftest.py (served_model)
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_LIB = os.environ.get("MLC_SERVE_EMBEDDING_MODEL_LIB")
EMBEDDING_MODEL_DIR = os.environ.get(
    "MLC_SERVE_EMBEDDING_MODEL",
    os.path.dirname(EMBEDDING_MODEL_LIB) if EMBEDDING_MODEL_LIB else None,
)


def _skip_if_no_model():
    if EMBEDDING_MODEL_LIB is None:
        pytest.skip(
            'Environment variable "MLC_SERVE_EMBEDDING_MODEL_LIB" not found. '
            "Set it to a compiled embedding model library "
            "(e.g., Qwen3-Embedding-0.6B-q0f32-MLC.dylib)."
        )
    if not os.path.isfile(EMBEDDING_MODEL_LIB):
        pytest.skip(f"Embedding model library not found at: {EMBEDDING_MODEL_LIB}")
    if EMBEDDING_MODEL_DIR is None or not os.path.isdir(EMBEDDING_MODEL_DIR):
        pytest.skip(f"Embedding model directory not found at: {EMBEDDING_MODEL_DIR}")


@pytest.fixture(scope="module")
def embedding_engine():
    """Module-scoped AsyncEmbeddingEngine — loaded once, shared across tests."""
    _skip_if_no_model()
    from mlc_llm.serve.embedding_engine import AsyncEmbeddingEngine

    engine = AsyncEmbeddingEngine(
        model=EMBEDDING_MODEL_DIR,
        model_lib=EMBEDDING_MODEL_LIB,
        device="auto",
    )
    yield engine
    engine.terminate()


# ---------------------------------------------------------------------------
# Helpers — reuse cosine_similarity pattern from test_serve_engine.py
# ---------------------------------------------------------------------------


def cosine_similarity(a, b):
    """Return cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ===================================================================
# Engine initialization tests
# ===================================================================


def test_engine_metadata(embedding_engine):
    """Engine reports valid model type and matching pooling strategy."""
    assert embedding_engine.model_type in ("encoder", "decoder")
    if embedding_engine.model_type == "encoder":
        assert embedding_engine.pooling_strategy == "cls"
    else:
        assert embedding_engine.pooling_strategy == "last"


# ===================================================================
# Single-text embedding
# ===================================================================


def test_single_text_smoke(embedding_engine):
    """Single text returns one unit-norm embedding vector."""
    embeddings, tokens = embedding_engine.embed(["Hello world"])
    assert len(embeddings) == 1
    assert len(embeddings[0]) > 0
    assert tokens > 0
    norm = float(np.linalg.norm(embeddings[0]))
    assert abs(norm - 1.0) < 1e-4, f"Expected unit norm, got {norm}"


# ===================================================================
# Batch embedding
# ===================================================================


def test_batch_mixed_lengths(embedding_engine):
    """Mixed-length batch returns correct count, consistent dim, unit norms, tokens>0."""
    texts = [
        "a",
        "a b c d e f g h i j",
        "a " * 100,
        "Machine learning is fascinating",
        "I love pizza",
        "Deep learning uses neural networks",
    ] + [f"item {i}" for i in range(10)]
    embeddings, tokens = embedding_engine.embed(texts)

    # count
    assert len(embeddings) == len(texts)
    assert tokens > 0

    # consistent dimension
    dims = {len(emb) for emb in embeddings}
    assert len(dims) == 1, f"Inconsistent dimensions: {dims}"

    # all unit-norm
    for i, emb in enumerate(embeddings):
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-3, f"Embedding [{i}] norm={norm}"


def test_item_order_preserved_in_batch(embedding_engine):
    """Output order matches input order, not sorted by length."""
    texts = ["short", "a much longer sentence for testing", "mid"]
    emb1, _ = embedding_engine.embed(texts)
    # Verify by re-embedding individually and matching
    for i, t in enumerate(texts):
        single_emb, _ = embedding_engine.embed([t])
        cos = cosine_similarity(emb1[i], single_emb[0])
        assert cos > 0.999, f"Item {i} order mismatch: cosine={cos}"


# ===================================================================
# Semantic quality — cosine similarity ranking
# ===================================================================

SIMILARITY_TEXTS = [
    "What is machine learning?",
    "Explain deep learning algorithms",
    "I want to order pizza",
]


def test_cosine_similarity_ranking(embedding_engine):
    """Related texts have higher cosine similarity than unrelated texts."""
    embeddings, _ = embedding_engine.embed(SIMILARITY_TEXTS)
    e_ml, e_dl, e_pizza = [np.array(e) for e in embeddings]
    sim_related = float(np.dot(e_ml, e_dl))
    sim_unrelated = float(np.dot(e_ml, e_pizza))
    assert (
        sim_related > sim_unrelated
    ), f"Related sim ({sim_related:.4f}) should > unrelated sim ({sim_unrelated:.4f})"


# ===================================================================
# Async embedding
# ===================================================================


def test_async_embed(embedding_engine):
    """async_embed produces same result as sync embed."""
    text = ["Async test"]
    sync_emb, sync_tokens = embedding_engine.embed(text)

    loop = asyncio.new_event_loop()
    try:
        async_emb, async_tokens = loop.run_until_complete(embedding_engine.async_embed(text))
    finally:
        loop.close()

    assert sync_tokens == async_tokens
    cos = cosine_similarity(sync_emb[0], async_emb[0])
    assert cos > 0.9999, f"Async vs sync mismatch, cosine={cos}"


# ===================================================================
# Edge cases
# ===================================================================


def test_empty_string(embedding_engine):
    """Empty string should still produce a valid embedding for supported models."""
    embeddings, tokens = embedding_engine.embed([""])
    assert len(embeddings) == 1
    assert len(embeddings[0]) > 0
    assert tokens > 0


def test_unicode_text(embedding_engine):
    """Unicode input is handled correctly."""
    texts = ["Привет мир", "你好世界", "こんにちは世界"]
    embeddings, _ = embedding_engine.embed(texts)
    assert len(embeddings) == 3
    for emb in embeddings:
        assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-4


# ===================================================================
# Long text handling (model-type dependent)
# ===================================================================


def test_long_text_decoder_chunked_prefill(embedding_engine):
    """[Decoder only] Text >prefill_chunk_size triggers chunked prefill.
    ~5000 tokens processed in 3 chunks. Result is unit-norm embedding."""
    if embedding_engine.model_type != "decoder":
        pytest.skip("Chunked prefill is decoder-only")
    long_text = "word " * 5000
    embeddings, tokens = embedding_engine.embed([long_text])
    assert tokens > 2048, f"Expected >2048 tokens to trigger chunking, got {tokens}"
    norm = float(np.linalg.norm(embeddings[0]))
    assert abs(norm - 1.0) < 1e-3


def _get_encoder_tokens(embedding_engine, text):
    """Replicate encoder preprocessing: tokenize and add [CLS]/[SEP]."""
    tokens = list(embedding_engine.tokenizer.encode(text))
    if embedding_engine._cls_token_id is not None and (
        len(tokens) == 0 or tokens[0] != embedding_engine._cls_token_id
    ):
        tokens = [embedding_engine._cls_token_id] + tokens
    if embedding_engine._sep_token_id is not None and (
        len(tokens) == 0 or tokens[-1] != embedding_engine._sep_token_id
    ):
        tokens = tokens + [embedding_engine._sep_token_id]
    return tokens


def test_long_text_encoder_truncation(embedding_engine):  # pylint: disable=too-many-locals
    """[Encoder only] Text exceeding prefill_chunk_size is truncated.
    Two texts with the same shared prefix but different suffixes beyond the
    limit should produce identical embeddings, since the suffix is truncated
    and the retained token prefixes are verified to be identical."""
    if embedding_engine.model_type != "encoder":
        pytest.skip("Truncation test is encoder-only")
    prefill_chunk = embedding_engine._metadata.get("prefill_chunk_size", 512)

    # Dynamically construct input that exceeds prefill_chunk_size.
    unit = "machine learning is great "
    suffix_a = " alpha beta gamma " * 200
    suffix_b = " totally different ending " * 200
    unit_tokens = len(list(embedding_engine.tokenizer.encode(unit)))
    repeats = max(1, prefill_chunk // max(unit_tokens, 1) + 64)

    # Increase prefix length until both inputs exceed prefill_chunk_size
    # and their truncated token prefixes are identical.
    while True:
        shared_prefix = unit * repeats
        full_tokens_a = _get_encoder_tokens(embedding_engine, shared_prefix + suffix_a)
        full_tokens_b = _get_encoder_tokens(embedding_engine, shared_prefix + suffix_b)
        if (
            len(full_tokens_a) > prefill_chunk
            and len(full_tokens_b) > prefill_chunk
            and full_tokens_a[:prefill_chunk] == full_tokens_b[:prefill_chunk]
        ):
            break
        repeats += 64
        assert repeats < 200000, "Failed to construct truncation test inputs"

    text_a = shared_prefix + suffix_a
    text_b = shared_prefix + suffix_b

    emb_a, tokens_a = embedding_engine.embed([text_a])
    emb_b, tokens_b = embedding_engine.embed([text_b])

    # Verify truncation happened
    assert (
        tokens_a <= prefill_chunk
    ), f"Encoder should truncate to {prefill_chunk}, got {tokens_a} tokens"
    assert tokens_b <= prefill_chunk
    # Both should be valid unit-norm embeddings
    assert abs(float(np.linalg.norm(emb_a[0])) - 1.0) < 1e-3
    assert abs(float(np.linalg.norm(emb_b[0])) - 1.0) < 1e-3

    # Both truncated to identical token sequences → embeddings must match
    cos = cosine_similarity(emb_a[0], emb_b[0])
    assert cos > 0.999, f"Same truncated tokens should match, cosine={cos:.6f}"


def test_long_vs_short_semantic_quality(embedding_engine):
    """Long text should still capture semantic meaning correctly.
    Decoder: chunked prefill preserves full context.
    Encoder: truncation keeps most relevant prefix."""
    short_ml = "Machine learning enables systems to learn from data"
    long_ml = (
        "Machine learning is a fascinating field of study. " * 200
        + "It enables systems to learn from data."
    )
    pizza = "I want to order a pepperoni pizza for dinner"

    embs, _ = embedding_engine.embed([short_ml, long_ml, pizza])
    e_short, e_long, e_pizza = [np.array(e) for e in embs]

    sim_same_topic = float(np.dot(e_short, e_long))
    sim_different = float(np.dot(e_short, e_pizza))
    assert (
        sim_same_topic > sim_different
    ), f"Same topic ({sim_same_topic:.4f}) should > different ({sim_different:.4f})"


# ===================================================================
# Repeated-call determinism (real model required)
# ===================================================================


def test_repeated_calls_no_workspace_pollution(embedding_engine):
    """Repeated calls with same input produce identical results.

    This catches workspace buffer reuse bugs where a previous call's
    hidden states leak into the next call's output.
    """
    text = ["workspace pollution test"]
    results = [embedding_engine.embed(text) for _ in range(5)]
    for i in range(1, 5):
        cos = cosine_similarity(results[0][0][0], results[i][0][0])
        assert cos > 0.9999, f"Call {i} diverged from call 0: cosine={cos}"
        assert results[0][1] == results[i][1], "Token counts should be identical"


# ===================================================================
# Unit tests — no real model required
# ===================================================================

from unittest import mock

try:
    from mlc_llm.serve import engine_utils
    from mlc_llm.serve.embedding_engine import AsyncEmbeddingEngine, EmbeddingRuntime

    _HAS_UNIT_DEPS = isinstance(EmbeddingRuntime, type) and isinstance(AsyncEmbeddingEngine, type)
except ImportError:
    _HAS_UNIT_DEPS = False

_skip_no_tvm = pytest.mark.skipif(not _HAS_UNIT_DEPS, reason="tvm not installed")

if _HAS_UNIT_DEPS:

    class _FakeRuntime(EmbeddingRuntime):
        """Minimal fake runtime for unit testing AsyncEmbeddingEngine delegation."""

        def __init__(self, model_type="decoder", pooling="last", norm=True, emb_meta=None):
            self._tokenizer_obj = mock.MagicMock(name="fake_tokenizer")
            self._model_type = model_type
            self._pooling = pooling
            self._norm = norm
            self._raw_metadata = {
                "prefill_chunk_size": 512,
                "context_window_size": 4096,
            }
            self._emb_meta = emb_meta
            self._cls_token_id = None
            self._sep_token_id = None
            self.embed_calls = []

        @property
        def device(self):
            return None

        @property
        def tokenizer(self):
            return self._tokenizer_obj

        @property
        def model_type(self):
            return self._model_type

        @property
        def pooling_strategy(self):
            return self._pooling

        @property
        def normalize(self):
            return self._norm

        @property
        def metadata(self):
            return self._raw_metadata

        @property
        def embedding_metadata(self):
            return self._emb_meta

        def embed(self, inputs):
            self.embed_calls.append(inputs)
            dim = 4
            return [[0.5] * dim for _ in inputs], sum(len(s.split()) for s in inputs)

    def _make_fake_mod(raw_metadata_dict):
        """Build a minimal fake TVM module whose ["_metadata"]() returns JSON."""
        import json as _json

        metadata_json = _json.dumps(raw_metadata_dict)
        fake_mod = mock.MagicMock()
        fake_mod.__getitem__ = lambda self, key: (
            (lambda: metadata_json) if key == "_metadata" else mock.MagicMock()
        )
        fake_mod.implements_function.return_value = True
        return fake_mod

    def _patch_tvm_deps(monkeypatch, raw_metadata_dict):
        """Monkeypatch TVM/tokenizer deps so TVMNativeEmbeddingRuntime.__init__ can run."""
        import mlc_llm.serve.embedding_engine as _eng_mod

        fake_mod = _make_fake_mod(raw_metadata_dict)
        fake_vm = mock.MagicMock()
        fake_vm.module = fake_mod

        monkeypatch.setattr(_eng_mod, "detect_device", lambda d: mock.MagicMock())
        monkeypatch.setattr(_eng_mod, "Tokenizer", lambda path: mock.MagicMock())
        monkeypatch.setattr(_eng_mod.tvm.runtime, "load_module", lambda path: mock.MagicMock())
        monkeypatch.setattr(_eng_mod.relax, "VirtualMachine", lambda ex, device: fake_vm)
        monkeypatch.setattr(_eng_mod.engine_utils, "load_embedding_params", lambda *a: [])


@_skip_no_tvm
def test_unit_engine_delegates_embed():
    """AsyncEmbeddingEngine.embed delegates to runtime.embed."""
    rt = _FakeRuntime()
    eng = AsyncEmbeddingEngine("", "", _runtime=rt)
    result = eng.embed(["hello world", "foo"])
    assert result == ([[0.5, 0.5, 0.5, 0.5]] * 2, 3)
    assert rt.embed_calls == [["hello world", "foo"]]
    eng.terminate()


@_skip_no_tvm
def test_unit_async_embed_matches_sync():
    """async_embed returns same result as sync embed."""
    rt = _FakeRuntime()
    eng = AsyncEmbeddingEngine("", "", _runtime=rt)
    sync_result = eng.embed(["test input"])
    loop = asyncio.new_event_loop()
    try:
        async_result = loop.run_until_complete(eng.async_embed(["test input"]))
    finally:
        loop.close()
    assert sync_result == async_result
    eng.terminate()


@_skip_no_tvm
def test_unit_attributes_mirrored():
    """Backward-compat attributes are mirrored from runtime."""
    rt = _FakeRuntime(model_type="encoder", pooling="cls", norm=False)
    rt._cls_token_id = 101  # pylint: disable=protected-access
    rt._sep_token_id = 102  # pylint: disable=protected-access
    eng = AsyncEmbeddingEngine("", "", _runtime=rt)
    assert eng.model_type == "encoder"
    assert eng.pooling_strategy == "cls"
    assert eng.normalize is False
    assert eng.tokenizer is rt._tokenizer_obj  # pylint: disable=protected-access
    assert eng._metadata is rt.metadata  # pylint: disable=protected-access
    assert eng._cls_token_id == 101  # pylint: disable=protected-access
    assert eng._sep_token_id == 102  # pylint: disable=protected-access
    assert eng.embedding_metadata is None
    eng.terminate()


@_skip_no_tvm
def test_unit_attributes_with_embedding_metadata():
    """embedding_metadata is mirrored when present."""
    emb_meta = {
        "model_type": "decoder",
        "pooling_strategy": "last",
        "normalize": True,
    }
    rt = _FakeRuntime(emb_meta=emb_meta)
    eng = AsyncEmbeddingEngine("", "", _runtime=rt)
    assert eng.embedding_metadata == emb_meta
    eng.terminate()


@_skip_no_tvm
def test_unit_terminate_idempotent():
    """Calling terminate multiple times does not raise."""
    rt = _FakeRuntime()
    eng = AsyncEmbeddingEngine("", "", _runtime=rt)
    eng.terminate()
    eng.terminate()
    assert eng._terminated is True  # pylint: disable=protected-access


@_skip_no_tvm
def test_unit_terminate_shuts_down_executor():
    """terminate shuts down the thread pool executor."""
    rt = _FakeRuntime()
    eng = AsyncEmbeddingEngine("", "", _runtime=rt)
    executor = eng._executor  # pylint: disable=protected-access
    eng.terminate()
    with pytest.raises(RuntimeError):
        executor.submit(lambda: None)


@_skip_no_tvm
@pytest.mark.parametrize(
    "config, expected",
    [
        pytest.param(
            {
                "model_task": "embedding",
                "embedding_metadata": {
                    "model_type": "decoder",
                    "pooling_strategy": "last",
                    "normalize": True,
                },
            },
            {"model_type": "decoder", "pooling_strategy": "last", "normalize": True},
            id="embedding_task_returns_metadata",
        ),
        pytest.param(
            {"model_task": "chat"},
            None,
            id="chat_task_returns_none",
        ),
        pytest.param(
            {},
            None,
            id="missing_task_returns_none",
        ),
    ],
)
def test_unit_get_embedding_metadata(config, expected):
    """get_embedding_metadata returns metadata only for embedding models."""
    result = engine_utils.get_embedding_metadata(config)
    assert result == expected


@_skip_no_tvm
@pytest.mark.parametrize(
    "raw_meta, match_msg",
    [
        pytest.param(
            {"model_task": "chat", "params": []},
            "Embedding metadata is missing or incomplete",
            id="missing_metadata",
        ),
        pytest.param(
            {
                "model_task": "embedding",
                "embedding_metadata": {"model_type": "decoder"},  # missing pooling_strategy, normalize
                "params": [],
            },
            "Embedding metadata is missing or incomplete",
            id="incomplete_metadata",
        ),
    ],
)
def test_unit_runtime_rejects_bad_metadata(monkeypatch, raw_meta, match_msg):
    """TVMNativeEmbeddingRuntime hard-fails on missing or incomplete embedding_metadata."""
    from mlc_llm.serve.embedding_engine import TVMNativeEmbeddingRuntime

    _patch_tvm_deps(monkeypatch, raw_meta)

    with pytest.raises(ValueError, match=match_msg):
        TVMNativeEmbeddingRuntime("fake_model", "fake_lib.so", device="cpu")


# ===================================================================
# Phase 3 — Canonicalization parity tests (unit, no model needed)
# ===================================================================

try:
    from mlc_llm.serve.embedding_engine import (
        _canonicalize_encoder_inputs,
        _get_encoder_special_token_ids,
    )

    _HAS_CANON_DEPS = True
except ImportError:
    _HAS_CANON_DEPS = False

_skip_no_canon = pytest.mark.skipif(not _HAS_CANON_DEPS, reason="tvm not installed")


class _MockTokenizer:
    """Deterministic mock tokenizer: each char -> its ordinal."""

    def encode(self, text):
        return [ord(c) for c in text]


@_skip_no_canon
class TestCanonicalizationParity:
    """Verify _canonicalize_encoder_inputs matches Phase 2 behavior exactly."""

    def test_cls_sep_injection(self):
        """CLS prepended, SEP appended when not already present."""
        tok = _MockTokenizer()
        result = _canonicalize_encoder_inputs(["abc"], tok, cls_token_id=101, sep_token_id=102, prefill_chunk_size=512)
        assert result[0][0] == 101
        assert result[0][-1] == 102
        assert result[0][1:-1] == [97, 98, 99]

    def test_no_duplicate_cls_sep(self):
        """No double-injection when first/last tokens already match CLS/SEP."""
        tok = _MockTokenizer()
        # 'a'=97 is CLS, 'c'=99 is SEP
        result = _canonicalize_encoder_inputs(["abc"], tok, cls_token_id=97, sep_token_id=99, prefill_chunk_size=512)
        assert result[0] == [97, 98, 99]

    @pytest.mark.parametrize(
        "cls_id, sep_id, expected_len, check_first, check_last",
        [
            pytest.param(101, 102, 5, 101, 102, id="truncation_with_sep_forced"),
            pytest.param(None, None, 5, ord("a"), ord("a"), id="truncation_without_sep"),
        ],
    )
    def test_truncation(self, cls_id, sep_id, expected_len, check_first, check_last):
        """Truncation respects CLS/SEP placement."""
        tok = _MockTokenizer()
        result = _canonicalize_encoder_inputs(
            ["a" * 10], tok, cls_token_id=cls_id, sep_token_id=sep_id, prefill_chunk_size=5
        )
        assert len(result[0]) == expected_len
        assert result[0][0] == check_first
        assert result[0][-1] == check_last

    def test_no_special_tokens(self):
        """No CLS/SEP when both are None."""
        tok = _MockTokenizer()
        result = _canonicalize_encoder_inputs(["xy"], tok, cls_token_id=None, sep_token_id=None, prefill_chunk_size=512)
        assert result[0] == [120, 121]

    def test_empty_string(self):
        """Empty string gets CLS+SEP if configured."""
        tok = _MockTokenizer()
        result = _canonicalize_encoder_inputs([""], tok, cls_token_id=101, sep_token_id=102, prefill_chunk_size=512)
        assert result[0] == [101, 102]

    def test_item_order_preserved(self):
        """Multiple inputs preserve order."""
        tok = _MockTokenizer()
        result = _canonicalize_encoder_inputs(
            ["a", "b", "c"], tok, cls_token_id=None, sep_token_id=None, prefill_chunk_size=512
        )
        assert len(result) == 3
        assert result[0] == [97]
        assert result[1] == [98]
        assert result[2] == [99]

    def test_total_tokens_consistent_with_phase2(self):
        """Sum of canonicalized lengths matches Phase 2 total_tokens semantics."""
        tok = _MockTokenizer()
        inputs = ["hello", "world"]
        result = _canonicalize_encoder_inputs(
            inputs, tok, cls_token_id=101, sep_token_id=102, prefill_chunk_size=512
        )
        total = sum(len(t) for t in result)
        # "hello" = 5 chars + CLS + SEP = 7, "world" = 5 chars + CLS + SEP = 7
        assert total == 14


# ===================================================================
# Phase 3 — Backend selector tests (unit, no model needed)
# ===================================================================


@_skip_no_tvm
class TestBackendSelector:
    """Verify AsyncEmbeddingEngine / AsyncMLCEmbeddingEngine backend selection."""

    def test_tvm_native_forced(self, monkeypatch, tmp_path):
        """MLC_SERVE_EMBEDDING_BACKEND=tvm_native calls _select_backend → TVMNativeEmbeddingRuntime."""
        monkeypatch.setenv("MLC_SERVE_EMBEDDING_BACKEND", "tvm_native")

        config_dir = tmp_path / "model"
        config_dir.mkdir()
        import json as _json

        (config_dir / "mlc-chat-config.json").write_text(
            _json.dumps({"embedding_metadata": {"model_type": "encoder"}})
        )

        import mlc_llm.serve.embedding_engine as _eng_mod

        raw_meta = {
            "model_task": "embedding",
            "embedding_metadata": {
                "model_type": "encoder",
                "pooling_strategy": "cls",
                "normalize": True,
            },
            "params": [],
        }
        _patch_tvm_deps(monkeypatch, raw_meta)

        rt = _eng_mod.AsyncMLCEmbeddingEngine._select_backend(
            str(config_dir), "fake.so", "cpu", None
        )
        assert isinstance(rt, _eng_mod.TVMNativeEmbeddingRuntime)

    def test_cpp_forced_rejects_decoder(self, monkeypatch, tmp_path):
        """MLC_SERVE_EMBEDDING_BACKEND=cpp rejects decoder models."""
        monkeypatch.setenv("MLC_SERVE_EMBEDDING_BACKEND", "cpp")
        # Write a minimal mlc-chat-config.json with decoder model type
        config_dir = tmp_path / "model"
        config_dir.mkdir()
        import json as _json

        (config_dir / "mlc-chat-config.json").write_text(
            _json.dumps({"embedding_metadata": {"model_type": "decoder"}})
        )
        from mlc_llm.serve.embedding_engine import AsyncMLCEmbeddingEngine

        with pytest.raises(ValueError, match="requires encoder model"):
            AsyncMLCEmbeddingEngine._select_backend(
                str(config_dir), "fake.so", "cpu", None
            )

    def test_auto_decoder_skips_cpp(self, monkeypatch, tmp_path):
        """auto mode for decoder models goes straight to TVMNativeEmbeddingRuntime."""
        monkeypatch.setenv("MLC_SERVE_EMBEDDING_BACKEND", "auto")
        config_dir = tmp_path / "model"
        config_dir.mkdir()
        import json as _json

        (config_dir / "mlc-chat-config.json").write_text(
            _json.dumps({"embedding_metadata": {"model_type": "decoder"}})
        )
        import mlc_llm.serve.embedding_engine as _eng_mod

        raw_meta = {
            "model_task": "embedding",
            "embedding_metadata": {"model_type": "decoder", "pooling_strategy": "last", "normalize": True},
            "params": [],
        }
        _patch_tvm_deps(monkeypatch, raw_meta)

        # Should not raise — decoder goes to TVMNativeEmbeddingRuntime directly
        rt = _eng_mod.AsyncMLCEmbeddingEngine._select_backend(
            str(config_dir), "fake.so", "cpu", None
        )
        assert isinstance(rt, _eng_mod.TVMNativeEmbeddingRuntime)


# ===================================================================
# Phase 3 — Negative tests (unit, no model needed)
# ===================================================================


@_skip_no_tvm
class TestNegativeCases:
    """Embedding lane should reject chat-only features."""

    def test_thread_encoder_runtime_rejects_decoder(self, monkeypatch, tmp_path):
        """ThreadEncoderRuntime rejects decoder model type."""
        config_dir = tmp_path / "model"
        config_dir.mkdir()
        import json as _json

        (config_dir / "mlc-chat-config.json").write_text(
            _json.dumps({
                "model_task": "embedding",
                "embedding_metadata": {"model_type": "decoder", "pooling_strategy": "last", "normalize": True},
            })
        )
        import mlc_llm.serve.embedding_engine as _eng_mod

        monkeypatch.setattr(_eng_mod, "Tokenizer", lambda path: mock.MagicMock())
        from mlc_llm.serve.embedding_engine import ThreadEncoderRuntime

        with pytest.raises(ValueError, match="only supports encoder models"):
            ThreadEncoderRuntime(str(config_dir), "fake.so", "cpu")

    def test_thread_encoder_runtime_rejects_missing_metadata(self, monkeypatch, tmp_path):
        """ThreadEncoderRuntime rejects model without embedding_metadata."""
        config_dir = tmp_path / "model"
        config_dir.mkdir()
        import json as _json

        (config_dir / "mlc-chat-config.json").write_text(_json.dumps({"model_task": "chat"}))
        import mlc_llm.serve.embedding_engine as _eng_mod

        monkeypatch.setattr(_eng_mod, "Tokenizer", lambda path: mock.MagicMock())
        from mlc_llm.serve.embedding_engine import ThreadEncoderRuntime

        with pytest.raises(ValueError, match="embedding_metadata"):
            ThreadEncoderRuntime(str(config_dir), "fake.so", "cpu")


# ===================================================================
# Phase 3 — Parity: C++ vs Phase 2 runtime (real model required)
# ===================================================================


def test_parity_cpp_vs_tvm_native(embedding_engine):
    """If using C++ backend, compare with TVMNativeEmbeddingRuntime output.

    This test only runs when the engine is backed by ThreadEncoderRuntime
    AND the model is available. It constructs a TVMNativeEmbeddingRuntime
    as golden reference and compares outputs.
    """
    try:
        from mlc_llm.serve.embedding_engine import ThreadEncoderRuntime
    except ImportError:
        pytest.skip("ThreadEncoderRuntime not available")

    # Only run this parity test if the engine is using the C++ backend
    if not isinstance(embedding_engine._runtime, ThreadEncoderRuntime):
        pytest.skip("Engine is not using C++ ThreadEncoderRuntime backend")

    from mlc_llm.serve.embedding_engine import TVMNativeEmbeddingRuntime

    # Construct the Phase 2 runtime for parity comparison
    tvm_rt = TVMNativeEmbeddingRuntime(
        model=EMBEDDING_MODEL_DIR,
        model_lib=EMBEDDING_MODEL_LIB,
        device="auto",
    )

    texts = [
        "Hello world",
        "Machine learning is great",
        "",  # empty string edge case
    ]

    cpp_emb, cpp_tokens = embedding_engine.embed(texts)
    tvm_emb, tvm_tokens = tvm_rt.embed(texts)

    # Item count must match
    assert len(cpp_emb) == len(tvm_emb) == len(texts)
    # Dimension must match
    assert len(cpp_emb[0]) == len(tvm_emb[0])
    # Token counts should be consistent
    assert cpp_tokens == tvm_tokens, f"Token count mismatch: C++={cpp_tokens}, TVM={tvm_tokens}"

    # Embeddings should be very close (cosine > 0.99)
    for i in range(len(texts)):
        cos = cosine_similarity(cpp_emb[i], tvm_emb[i])
        assert cos > 0.99, (
            f"Parity mismatch for item {i} ({texts[i]!r}): cosine={cos:.6f}"
        )


# ===================================================================
# Standalone runner (like test_serve_engine.py)
# ===================================================================

if __name__ == "__main__":
    _skip_if_no_model()
    from mlc_llm.serve.embedding_engine import AsyncEmbeddingEngine

    engine = AsyncEmbeddingEngine(
        model=EMBEDDING_MODEL_DIR,
        model_lib=EMBEDDING_MODEL_LIB,
        device="auto",
    )
    try:
        test_engine_metadata(engine)
        test_single_text_smoke(engine)
        test_batch_mixed_lengths(engine)
        test_item_order_preserved_in_batch(engine)
        test_cosine_similarity_ranking(engine)
        test_async_embed(engine)
        test_empty_string(engine)
        test_unicode_text(engine)
        test_long_text_decoder_chunked_prefill(engine)
        test_long_text_encoder_truncation(engine)
        test_long_vs_short_semantic_quality(engine)
        test_repeated_calls_no_workspace_pollution(engine)
        print("\nAll embedding engine tests passed!")
    finally:
        engine.terminate()
