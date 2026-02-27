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
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ===================================================================
# Engine initialization tests
# ===================================================================


def test_engine_model_type(embedding_engine):
    """Engine reports a valid model type."""
    assert embedding_engine.model_type in ("encoder", "decoder")


def test_engine_pooling_strategy(embedding_engine):
    """Engine selects appropriate default pooling strategy."""
    if embedding_engine.model_type == "encoder":
        assert embedding_engine.pooling_strategy == "cls"
    else:
        assert embedding_engine.pooling_strategy == "last"


# ===================================================================
# Single-text embedding
# ===================================================================


def test_single_text_shape(embedding_engine):
    """Single text returns exactly one embedding vector."""
    embeddings, tokens = embedding_engine.embed(["Hello world"])
    assert len(embeddings) == 1
    assert len(embeddings[0]) > 0
    assert tokens > 0


def test_single_text_unit_norm(embedding_engine):
    """Embedding output is L2-normalized."""
    embeddings, _ = embedding_engine.embed(["Hello world"])
    norm = float(np.linalg.norm(embeddings[0]))
    assert abs(norm - 1.0) < 1e-4, f"Expected unit norm, got {norm}"


# ===================================================================
# Batch embedding
# ===================================================================

BATCH_TEXTS = [
    "Machine learning is fascinating",
    "I love pizza",
    "Deep learning uses neural networks",
]


def test_batch_count(embedding_engine):
    """Batch embedding returns one vector per input."""
    embeddings, tokens = embedding_engine.embed(BATCH_TEXTS)
    assert len(embeddings) == len(BATCH_TEXTS)
    assert tokens > 0


def test_batch_all_normalized(embedding_engine):
    """Every vector in a batch is L2-normalized."""
    embeddings, _ = embedding_engine.embed(BATCH_TEXTS)
    for i, emb in enumerate(embeddings):
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-4, f"Embedding [{i}] norm={norm}"


def test_batch_consistent_dimension(embedding_engine):
    """All embeddings in a batch have the same dimension."""
    embeddings, _ = embedding_engine.embed(BATCH_TEXTS)
    dims = {len(emb) for emb in embeddings}
    assert len(dims) == 1, f"Inconsistent dimensions: {dims}"


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
# Determinism
# ===================================================================


def test_deterministic_output(embedding_engine):
    """Same input produces identical output across calls."""
    text = ["Deterministic test"]
    emb1, _ = embedding_engine.embed(text)
    emb2, _ = embedding_engine.embed(text)
    cos = cosine_similarity(emb1[0], emb2[0])
    assert cos > 0.9999, f"Expected deterministic output, cosine={cos}"


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
    """Empty string behavior depends on model type.
    Encoder: [CLS]+[SEP] → valid embedding. Decoder: zero tokens → error."""
    if embedding_engine.model_type == "encoder":
        # Encoder adds [CLS]/[SEP], so empty string still produces valid embedding
        embeddings, _ = embedding_engine.embed([""])
        assert len(embeddings) == 1
        assert len(embeddings[0]) > 0
    else:
        # Decoder has no special tokens, zero tokens → skipped, empty result
        embeddings, tokens = embedding_engine.embed([""])
        assert len(embeddings) == 0
        assert tokens == 0


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


def test_long_text_encoder_truncation(embedding_engine):
    """[Encoder only] Text exceeding prefill_chunk_size is truncated.
    Two texts with the same prefix but different suffixes beyond the limit
    should produce identical embeddings, since the suffix gets truncated."""
    if embedding_engine.model_type != "encoder":
        pytest.skip("Truncation test is encoder-only")
    prefill_chunk = embedding_engine._metadata.get("prefill_chunk_size", 512)

    # Same prefix, different suffixes — both exceed the limit
    shared_prefix = "machine learning is great " * 500  # ~2500 tokens
    text_a = shared_prefix + " alpha beta gamma " * 500
    text_b = shared_prefix + " totally different ending " * 500

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

    # Both truncated to same first N tokens → identical embeddings
    cos = cosine_similarity(emb_a[0], emb_b[0])
    assert cos > 0.99, f"Same-prefix texts after truncation should match, cosine={cos:.6f}"


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


def test_unicode_text(embedding_engine):
    """Unicode input is handled correctly."""
    texts = ["Привет мир", "你好世界", "こんにちは世界"]
    embeddings, _ = embedding_engine.embed(texts)
    assert len(embeddings) == 3
    for emb in embeddings:
        assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-4


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
        test_engine_model_type(engine)
        test_engine_pooling_strategy(engine)
        test_single_text_shape(engine)
        test_single_text_unit_norm(engine)
        test_batch_count(engine)
        test_batch_all_normalized(engine)
        test_batch_consistent_dimension(engine)
        test_cosine_similarity_ranking(engine)
        test_deterministic_output(engine)
        test_async_embed(engine)
        test_empty_string(engine)
        test_long_text_decoder_chunked_prefill(engine)
        test_long_text_encoder_truncation(engine)
        test_long_vs_short_semantic_quality(engine)
        test_unicode_text(engine)
        print("\nAll embedding engine tests passed!")
    finally:
        engine.terminate()
