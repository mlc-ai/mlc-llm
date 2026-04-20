"""Asynchronous embedding inference engine for encoder and decoder models."""

import abc
import asyncio
import concurrent.futures
import json
import logging
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import tvm
from tvm import relax
from tvm.runtime import Device, ShapeTuple

from mlc_llm.serve import engine_utils
from mlc_llm.serve.config import EngineConfig
from mlc_llm.support.auto_device import detect_device
from mlc_llm.tokenizers import Tokenizer

logger = logging.getLogger(__name__)

# ====================================================================
# Shared canonicalization helpers (encoder + decoder lanes)
# ====================================================================


def _get_encoder_special_token_ids(
    tokenizer: Tokenizer, model_path: str
) -> Tuple[Optional[int], Optional[int]]:
    """Read CLS and SEP token IDs for an encoder model.

    Mirrors the Phase 2 TVMNativeEmbeddingRuntime._init_encoder logic exactly.

    Returns
    -------
    cls_token_id : Optional[int]
    sep_token_id : Optional[int]
    """
    cls_token_id: Optional[int] = None
    sep_token_id: Optional[int] = None

    tok_config_path = os.path.join(model_path, "tokenizer_config.json")
    if os.path.exists(tok_config_path):
        with open(tok_config_path, encoding="utf-8") as f:
            tok_config = json.load(f)
        # Try added_tokens_decoder first (newer HF format)
        added = tok_config.get("added_tokens_decoder", {})
        for tid, info in added.items():
            if info.get("content") == tok_config.get("cls_token"):
                cls_token_id = int(tid)
            if info.get("content") == tok_config.get("sep_token"):
                sep_token_id = int(tid)
        # Fallback: encode the special token strings via tokenizer
        if cls_token_id is None and tok_config.get("cls_token"):
            ids = list(tokenizer.encode(tok_config["cls_token"]))
            if len(ids) == 1:
                cls_token_id = ids[0]
        if sep_token_id is None and tok_config.get("sep_token"):
            ids = list(tokenizer.encode(tok_config["sep_token"]))
            if len(ids) == 1:
                sep_token_id = ids[0]

    return cls_token_id, sep_token_id


def _canonicalize_encoder_inputs(
    inputs: List[str],
    tokenizer: Tokenizer,
    cls_token_id: Optional[int],
    sep_token_id: Optional[int],
    prefill_chunk_size: int,
) -> List[List[int]]:
    """Canonicalize encoder inputs: tokenize, add CLS/SEP, truncate.

    This is the single source of truth for encoder preprocessing, shared by
    TVMNativeEmbeddingRuntime and ThreadEncoderRuntime.

    Bug-for-bug compatible with Phase 2 TVMNativeEmbeddingRuntime._embed_encoder.

    Parameters
    ----------
    inputs : List[str]
        Raw text inputs.
    tokenizer : Tokenizer
        The tokenizer instance.
    cls_token_id : Optional[int]
        CLS token ID, or None if not applicable.
    sep_token_id : Optional[int]
        SEP token ID, or None if not applicable.
    prefill_chunk_size : int
        Maximum sequence length; inputs are truncated to this length.

    Returns
    -------
    List[List[int]]
        Canonicalized token ID sequences (one per input).
    """
    result: List[List[int]] = []
    for text in inputs:
        tokens = list(tokenizer.encode(text))
        # Add [CLS] if needed
        if cls_token_id is not None and (len(tokens) == 0 or tokens[0] != cls_token_id):
            tokens = [cls_token_id] + tokens
        # Add [SEP] if needed
        if sep_token_id is not None and (len(tokens) == 0 or tokens[-1] != sep_token_id):
            tokens = tokens + [sep_token_id]
        # Truncate to compiled buffer limit (keep [CLS] at start, force [SEP] at end)
        if len(tokens) > prefill_chunk_size:
            tokens = tokens[:prefill_chunk_size]
            if sep_token_id is not None:
                tokens[-1] = sep_token_id
        result.append(tokens)
    return result


def _get_decoder_special_tokens(
    tokenizer: Tokenizer, model_path: str
) -> Tuple[bool, Optional[int]]:
    """Read decoder-embedding EOS handling for a Qwen3-Embedding-style model.

    Mirrors the Phase 2 TVMNativeEmbeddingRuntime._init_decoder logic exactly.
    Extracted so both the legacy TVM-native runtime and the Phase-4 threaded
    runtime agree on when to append EOS during canonicalization.

    Returns
    -------
    tokenizer_appends_eos : bool
        ``True`` when the HF ``tokenizer.json`` already appends a special token
        (e.g. via ``TemplateProcessing`` with ``$A <|endoftext|>``). If true, the
        caller must NOT append EOS manually.
    eos_token_id : Optional[int]
        Model's EOS token id read from ``mlc-chat-config.json``'s
        ``eos_token_id`` field. ``None`` if absent; caller should skip manual
        EOS append in that case.
    """
    tokenizer_appends_eos = False
    tokenizer_json_path = os.path.join(model_path, "tokenizer.json")
    if os.path.exists(tokenizer_json_path):
        with open(tokenizer_json_path, encoding="utf-8") as f:
            tokenizer_json = json.load(f)
        post_proc = tokenizer_json.get("post_processor")
        if post_proc is not None:
            test_tokens = list(tokenizer.encode("test"))
            if len(test_tokens) > 0:
                vocab = tokenizer_json.get("added_tokens", [])
                special_ids = {t["id"] for t in vocab if t.get("special", False)}
                if test_tokens[-1] in special_ids:
                    tokenizer_appends_eos = True

    eos_token_id: Optional[int] = None
    config_path = os.path.join(model_path, "mlc-chat-config.json")
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            chat_config = json.load(f)
        eos = chat_config.get("eos_token_id")
        if isinstance(eos, list):
            eos_token_id = eos[0]
        elif isinstance(eos, int):
            eos_token_id = eos

    return tokenizer_appends_eos, eos_token_id


def _canonicalize_decoder_inputs(
    inputs: List[str],
    tokenizer: Tokenizer,
    tokenizer_appends_eos: bool,
    eos_token_id: Optional[int],
    max_seq_len: int,
) -> List[List[int]]:
    """Canonicalize decoder-embedding inputs: tokenize, conditionally append EOS, truncate.

    Shared source of truth for decoder preprocessing between
    :class:`TVMNativeEmbeddingRuntime._embed_decoder` and the Phase-4 threaded
    runtime decoder lane.

    Parameters
    ----------
    inputs : List[str]
        Raw text inputs.
    tokenizer : Tokenizer
        The tokenizer instance.
    tokenizer_appends_eos : bool
        If ``True``, the tokenizer already appends EOS via its post-processor,
        so this helper skips manual append.
    eos_token_id : Optional[int]
        EOS token id. ``None`` disables manual append.
    max_seq_len : int
        Maximum sequence length; inputs are truncated to this length.

    Returns
    -------
    List[List[int]]
        Canonicalized token ID sequences (one per input).
    """
    result: List[List[int]] = []
    for text in inputs:
        tokens = list(tokenizer.encode(text))
        if (
            not tokenizer_appends_eos
            and eos_token_id is not None
            and (len(tokens) == 0 or tokens[-1] != eos_token_id)
        ):
            tokens.append(eos_token_id)
        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]
        result.append(tokens)
    return result


# ====================================================================
# EmbeddingRuntime — abstract interface
# ====================================================================


class EmbeddingRuntime(abc.ABC):
    """Abstract base class for embedding runtime implementations.

    Minimal responsibilities:
    - Initialize tokenizer, module, params, metadata
    - Provide embed(inputs) -> (embeddings, total_tokens)
    - Expose read-only state: tokenizer, model_type, pooling_strategy, normalize, metadata
    """

    @property
    @abc.abstractmethod
    def tokenizer(self) -> Tokenizer:
        """The tokenizer instance."""

    @property
    @abc.abstractmethod
    def model_type(self) -> str:
        """Model type: 'encoder' or 'decoder'."""

    @property
    @abc.abstractmethod
    def pooling_strategy(self) -> str:
        """Pooling strategy: 'cls', 'mean', or 'last'."""

    @property
    @abc.abstractmethod
    def normalize(self) -> bool:
        """Whether to L2-normalize embeddings."""

    @property
    @abc.abstractmethod
    def metadata(self) -> dict:
        """Model metadata dictionary."""

    @abc.abstractmethod
    def embed(self, inputs: List[str]) -> Tuple[List[List[float]], int]:
        """Compute embeddings for input strings.

        Returns
        -------
        embeddings : List[List[float]]
            The embedding vectors.
        total_tokens : int
            Total number of tokens processed.
        """


# ====================================================================
# TVMNativeEmbeddingRuntime — TVM-native implementation
# ====================================================================


class TVMNativeEmbeddingRuntime(EmbeddingRuntime):  # pylint: disable=too-many-instance-attributes
    """TVM-native embedding runtime for encoder and decoder models.

    Handles TVM module loading, tokenization, and embedding inference
    for both encoder (BERT-style) and decoder-only (Qwen3-Embeddings) models.

    Parameters
    ----------
    model : str
        Path to the model weight directory.

    model_lib : str
        Path to the compiled model library (.so/.dylib file).

    device : Union[str, Device]
        Device string, e.g. "auto", "cuda:0", "metal".

    pooling_strategy : Optional[str]
        Pooling strategy override: "cls", "mean", or "last".
    """

    def __init__(  # pylint: disable=too-many-branches
        self,
        model: str,
        model_lib: str,
        device: Union[str, Device] = "auto",
        *,
        pooling_strategy: Optional[str] = None,
    ) -> None:
        # Reuse existing utility: device detection
        self._device = detect_device(device) if isinstance(device, str) else device
        # Reuse existing utility: tokenizer
        self._tokenizer = Tokenizer(model)

        # Load TVM module, metadata, and params via engine_utils helpers
        ex = tvm.runtime.load_module(model_lib)
        vm = relax.VirtualMachine(ex, device=self._device)
        self._mod = vm.module
        self._raw_metadata = json.loads(self._mod["_metadata"]())
        self._params = engine_utils.load_embedding_params(model, self._device, self._raw_metadata)

        # Read embedding metadata — required since phase-1 metadata path
        self._embedding_metadata = engine_utils.get_embedding_metadata(self._raw_metadata)
        _REQUIRED_FIELDS = ("model_type", "pooling_strategy", "normalize")
        if self._embedding_metadata is None or not all(
            k in self._embedding_metadata for k in _REQUIRED_FIELDS
        ):
            missing = (
                "all"
                if self._embedding_metadata is None
                else ", ".join(k for k in _REQUIRED_FIELDS if k not in self._embedding_metadata)
            )
            raise ValueError(
                f"Embedding metadata is missing or incomplete in the model library metadata "
                f"(missing: {missing}). "
                "Embedding serving requires models compiled with embedding_metadata "
                "(phase-1 metadata path). Please regenerate/recompile the model artifacts "
                "with the current MLC LLM toolchain."
            )
        self._model_type = self._embedding_metadata["model_type"]
        self._pooling_strategy = self._embedding_metadata["pooling_strategy"]
        self._normalize = self._embedding_metadata["normalize"]
        # Allow caller to override pooling strategy
        if pooling_strategy:
            self._pooling_strategy = pooling_strategy

        # Special token ids (set by _init_encoder; None for decoder models)
        self._cls_token_id: Optional[int] = None
        self._sep_token_id: Optional[int] = None

        # Initialize model-type-specific functions
        if self._model_type == "encoder":
            self._init_encoder(model)
        else:
            self._init_decoder(model)

    # ---- Read-only properties (abstract interface) ----

    @property
    def device(self) -> Device:
        """The target device."""
        return self._device

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def pooling_strategy(self) -> str:
        return self._pooling_strategy

    @property
    def normalize(self) -> bool:
        return self._normalize

    @property
    def metadata(self) -> dict:
        return self._raw_metadata

    @property
    def embedding_metadata(self) -> Optional[dict]:
        """Embedding-specific metadata, or None for legacy models."""
        return self._embedding_metadata

    # ---- Initialization helpers ----

    def _init_encoder(self, model: str) -> None:
        """Initialize encoder (BERT-style) model functions and special tokens."""
        self._prefill_func = self._mod["prefill"]
        self._cls_token_id, self._sep_token_id = _get_encoder_special_token_ids(
            self._tokenizer, model
        )

    def _init_decoder(self, model: str) -> None:
        """Initialize decoder (Qwen3-Embeddings style) model functions."""
        # Prefer tokenizer post-processing (HF-style) for terminal/pooling token handling.
        # Only fall back to manual EOS append when tokenizer does not define a post-processor
        # that actually appends a token at the end of the sequence.
        self._decoder_tokenizer_appends_eos = False
        tokenizer_json_path = os.path.join(model, "tokenizer.json")
        if os.path.exists(tokenizer_json_path):
            with open(tokenizer_json_path, encoding="utf-8") as f:
                tokenizer_json = json.load(f)
            post_proc = tokenizer_json.get("post_processor")
            if post_proc is not None:
                # Check if the post-processor actually appends a special token at the end
                # (e.g. TemplateProcessing with "$A <|endoftext|>"). We verify by encoding
                # a test string and checking if the last token is a known special token.
                test_tokens = list(self._tokenizer.encode("test"))
                if len(test_tokens) > 0:
                    vocab = tokenizer_json.get("added_tokens", [])
                    special_ids = {t["id"] for t in vocab if t.get("special", False)}
                    if test_tokens[-1] in special_ids:
                        self._decoder_tokenizer_appends_eos = True

        # Read EOS token from config — fallback only when tokenizer does not auto-append.
        self._decoder_eos_token_id: Optional[int] = None
        config_path = os.path.join(model, "mlc-chat-config.json")
        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                chat_config = json.load(f)
            eos = chat_config.get("eos_token_id")
            if isinstance(eos, list):
                self._decoder_eos_token_id = eos[0]
            elif isinstance(eos, int):
                self._decoder_eos_token_id = eos

        self._embed_func = self._mod["embed"]
        self._prefill_to_hidden_func = self._mod["prefill_to_last_hidden_states"]
        self._batch_prefill_to_hidden_func = self._mod["batch_prefill_to_last_hidden_states"]
        if self._mod.implements_function("create_tir_paged_kv_cache"):
            self._create_kv_cache_func = self._mod["create_tir_paged_kv_cache"]
        elif self._mod.implements_function("create_flashinfer_paged_kv_cache"):
            self._create_kv_cache_func = self._mod["create_flashinfer_paged_kv_cache"]
        else:
            raise RuntimeError("Cannot find KV cache creation function in model library.")
        self._kv_state_add_sequence = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
        self._kv_state_remove_sequence = tvm.get_global_func("vm.builtin.kv_state_remove_sequence")
        self._kv_state_begin_forward = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
        self._kv_state_end_forward = tvm.get_global_func("vm.builtin.kv_state_end_forward")
        self._nd_reshape = tvm.get_global_func("vm.builtin.reshape")

    # ---- Embedding methods ----

    def embed(self, inputs: List[str]) -> Tuple[List[List[float]], int]:
        """Compute embeddings for a list of input strings (synchronous)."""
        if self._model_type == "encoder":
            return self._embed_encoder(inputs)
        return self._embed_decoder(inputs)

    def _embed_encoder(  # pylint: disable=too-many-locals
        self, inputs: List[str]
    ) -> Tuple[List[List[float]], int]:
        """Encoder model embedding (BERT-style)."""
        embeddings: List[List[float]] = []
        total_tokens = 0
        prefill_chunk = self._raw_metadata.get("prefill_chunk_size", 512)

        # Use shared canonicalization helper
        token_lists = _canonicalize_encoder_inputs(
            inputs, self._tokenizer, self._cls_token_id, self._sep_token_id, prefill_chunk
        )

        for tokens in token_lists:
            seq_len = len(tokens)
            total_tokens += seq_len

            token_ids = np.array([tokens], dtype=np.int32)  # [1, seq_len]
            attention_mask: np.ndarray = np.ones((1, seq_len), dtype=np.int32)  # [1, seq_len]

            tokens_tvm = tvm.runtime.tensor(token_ids, device=self._device)
            mask_tvm = tvm.runtime.tensor(attention_mask, device=self._device)

            output = self._prefill_func(tokens_tvm, mask_tvm, self._params)
            output_np = output.numpy()  # [1, seq_len, hidden_size]

            # Pooling
            if self._pooling_strategy == "cls":
                pooled = output_np[0, 0, :]
            elif self._pooling_strategy == "mean":
                pooled = output_np[0].mean(axis=0)
            else:  # "last"
                pooled = output_np[0, -1, :]

            # L2 normalize
            pooled = pooled.astype(np.float32)
            if self._normalize:
                norm = np.linalg.norm(pooled)
                if norm > 1e-12:
                    pooled = pooled / norm

            embeddings.append(pooled.tolist())

        return embeddings, total_tokens

    def _embed_decoder(self, inputs: List[str]) -> Tuple[List[List[float]], int]:
        """Decoder model embedding with batch prefill optimization."""
        prefill_chunk = self._raw_metadata.get("prefill_chunk_size", 2048)
        max_seq_len = self._raw_metadata.get("context_window_size", 32768)
        if max_seq_len == -1:
            max_seq_len = self._raw_metadata.get("sliding_window_size", -1)
        assert max_seq_len > 0, f"max_seq_len must be positive, got {max_seq_len}"
        support_sliding = int(self._raw_metadata.get("sliding_window_size", -1) != -1)

        token_lists: List[List[int]] = []
        for text in inputs:
            tokens = list(self._tokenizer.encode(text))
            if (
                not self._decoder_tokenizer_appends_eos
                and self._decoder_eos_token_id is not None
                and (len(tokens) == 0 or tokens[-1] != self._decoder_eos_token_id)
            ):
                tokens.append(self._decoder_eos_token_id)
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]
            token_lists.append(tokens)

        total_tokens = sum(len(t) for t in token_lists)

        if total_tokens <= prefill_chunk and all(len(t) > 0 for t in token_lists):
            return self._batch_embed_decoder(
                token_lists, total_tokens, max_seq_len, prefill_chunk, support_sliding
            )

        sub_batches = self._build_sub_batches(token_lists, prefill_chunk)
        all_embeddings: List[List[float]] = []
        for batch_type, batch, batch_total in sub_batches:
            if batch_type == "batch":
                embs, _ = self._batch_embed_decoder(
                    batch, batch_total, max_seq_len, prefill_chunk, support_sliding
                )
            else:
                embs, _ = self._sequential_embed_decoder(
                    batch, batch_total, max_seq_len, prefill_chunk, support_sliding
                )
            all_embeddings.extend(embs)

        return all_embeddings, total_tokens

    @staticmethod
    def _build_sub_batches(
        token_lists: List[List[int]], prefill_chunk: int
    ) -> List[Tuple[Literal["batch", "sequential"], List[List[int]], int]]:
        """Partition token lists into sub-batches that fit within prefill_chunk."""
        sub_batches: List[Tuple[Literal["batch", "sequential"], List[List[int]], int]] = []
        current_batch: List[List[int]] = []
        current_tokens = 0

        for tokens in token_lists:
            if not tokens:
                continue
            token_len = len(tokens)
            is_oversized = token_len > prefill_chunk
            if current_batch and (is_oversized or current_tokens + token_len > prefill_chunk):
                sub_batches.append(("batch", current_batch, current_tokens))
                current_batch, current_tokens = [], 0
            if is_oversized:
                sub_batches.append(("sequential", [tokens], token_len))
            else:
                current_batch.append(tokens)
                current_tokens += token_len
        if current_batch:
            sub_batches.append(("batch", current_batch, current_tokens))

        return sub_batches

    def _batch_embed_decoder(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        token_lists: List[List[int]],
        total_tokens: int,
        max_seq_len: int,
        prefill_chunk: int,
        support_sliding: int,
    ) -> Tuple[List[List[float]], int]:
        """Batch prefill: process all inputs in a single forward pass."""
        batch_size = len(token_lists)

        kv_cache = self._create_kv_cache_func(
            ShapeTuple([batch_size]),
            ShapeTuple([max_seq_len]),
            ShapeTuple([prefill_chunk]),
            ShapeTuple([16]),
            ShapeTuple([support_sliding]),
        )

        seq_ids = list(range(batch_size))
        seq_lens = [len(t) for t in token_lists]
        for sid in seq_ids:
            self._kv_state_add_sequence(kv_cache, sid)

        self._kv_state_begin_forward(kv_cache, ShapeTuple(seq_ids), ShapeTuple(seq_lens))

        all_tokens = []
        for tokens in token_lists:
            all_tokens.extend(tokens)
        token_ids = tvm.runtime.tensor(np.array(all_tokens, dtype=np.int32), device=self._device)
        all_embed = self._embed_func(token_ids, self._params)
        all_embed = self._nd_reshape(all_embed, ShapeTuple([1, total_tokens, all_embed.shape[-1]]))

        hidden_states, _ = self._batch_prefill_to_hidden_func(all_embed, kv_cache, self._params)
        hidden_np = hidden_states.numpy()
        self._kv_state_end_forward(kv_cache)
        for sid in seq_ids:
            self._kv_state_remove_sequence(kv_cache, sid)

        embeddings: List[List[float]] = []
        offset = 0
        for tokens in token_lists:
            last_pos = offset + len(tokens) - 1
            pooled = hidden_np[0, last_pos, :].astype(np.float32)
            if self._normalize:
                norm = np.linalg.norm(pooled)
                if norm > 1e-12:
                    pooled = pooled / norm
            embeddings.append(pooled.tolist())
            offset += len(tokens)

        return embeddings, total_tokens

    def _sequential_embed_decoder(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        token_lists: List[List[int]],
        total_tokens: int,
        max_seq_len: int,
        prefill_chunk: int,
        support_sliding: int,
    ) -> Tuple[List[List[float]], int]:
        """Sequential chunked prefill: process each input independently."""
        embeddings: List[List[float]] = []

        for tokens in token_lists:
            if len(tokens) == 0:
                continue

            kv_cache = self._create_kv_cache_func(
                ShapeTuple([1]),
                ShapeTuple([max_seq_len]),
                ShapeTuple([prefill_chunk]),
                ShapeTuple([16]),
                ShapeTuple([support_sliding]),
            )
            self._kv_state_add_sequence(kv_cache, 0)

            hidden = None
            for chunk_start in range(0, len(tokens), prefill_chunk):
                chunk_end = min(chunk_start + prefill_chunk, len(tokens))
                chunk_tokens = tokens[chunk_start:chunk_end]
                chunk_len = len(chunk_tokens)

                token_ids = tvm.runtime.tensor(
                    np.array(chunk_tokens, dtype=np.int32), device=self._device
                )
                chunk_embed = self._embed_func(token_ids, self._params)
                chunk_embed = self._nd_reshape(
                    chunk_embed, ShapeTuple([1, chunk_len, chunk_embed.shape[-1]])
                )
                self._kv_state_begin_forward(kv_cache, ShapeTuple([0]), ShapeTuple([chunk_len]))
                hidden, kv_cache = self._prefill_to_hidden_func(chunk_embed, kv_cache, self._params)
                hidden_np = hidden.numpy()
                self._kv_state_end_forward(kv_cache)

            self._kv_state_remove_sequence(kv_cache, 0)

            pooled = hidden_np[0, -1, :] if hidden_np.ndim == 3 else hidden_np[-1, :]
            pooled = pooled.astype(np.float32)
            if self._normalize:
                norm = np.linalg.norm(pooled)
                if norm > 1e-12:
                    pooled = pooled / norm
            embeddings.append(pooled.tolist())

        return embeddings, total_tokens


# ====================================================================
# ThreadedEmbeddingRuntime — C++ threaded embedding backend (encoder + decoder)
# ====================================================================

# Pooling strategy name -> C++ int mapping
_POOLING_STRATEGY_MAP = {"cls": 0, "mean": 1, "last": 2}
_THREAD_WAIT_POLL_SEC = 0.1


class ThreadedEmbeddingRuntime(EmbeddingRuntime):
    """C++ threaded embedding runtime using the first-class embedding lane.

    Covers both encoder models (Phase 3, right-pad bidirectional) and
    decoder-only models (Phase 4, left-pad causal). Lane selection is driven
    by ``embedding_metadata.model_type`` in ``mlc-chat-config.json``; the C++
    engine's ``InitEmbeddingLane`` picks the matching
    ``BatchEmbeddingPrefillAction`` / ``BatchDecoderEmbeddingPrefillAction``.

    The threaded runtime itself uses the same two-thread pattern as
    ``AsyncMLCEngine`` for chat. Only the token canonicalization step differs
    per model_type (CLS/SEP append for encoder, EOS append for decoder).
    """

    def __init__(
        self,
        model: str,
        model_lib: str,
        device: Union[str, Device] = "auto",
        *,
        pooling_strategy: Optional[str] = None,
    ) -> None:
        # Set _terminated early so __del__ is safe even if __init__ fails partway.
        self._terminated = True
        self._module = None
        self._ffi: Dict[str, Any] = {}
        self._bg_loop_thread: Optional[threading.Thread] = None
        self._bg_stream_thread: Optional[threading.Thread] = None
        self._pending_lock = threading.Lock()
        self._pending: Dict[str, "_PendingEntry"] = {}

        self._device = detect_device(device) if isinstance(device, str) else device
        self._tokenizer = Tokenizer(model)
        self._model_path = model
        self._model_lib = model_lib

        # Read metadata from mlc-chat-config.json
        config_path = os.path.join(model, "mlc-chat-config.json")
        with open(config_path, encoding="utf-8") as f:
            self._raw_metadata = json.load(f)

        self._embedding_metadata = engine_utils.get_embedding_metadata(self._raw_metadata)
        if self._embedding_metadata is None:
            raise ValueError("Model does not have embedding_metadata in mlc-chat-config.json")

        self._model_type = self._embedding_metadata["model_type"]
        if self._model_type not in ("encoder", "decoder"):
            raise ValueError(
                f"ThreadedEmbeddingRuntime supports model_type in "
                f"{{'encoder', 'decoder'}}; got {self._model_type!r}"
            )

        self._pooling_strategy = self._embedding_metadata["pooling_strategy"]
        self._normalize = self._embedding_metadata["normalize"]
        if pooling_strategy:
            self._pooling_strategy = pooling_strategy

        # Lane-specific canonicalization state.
        # Encoder lane: CLS/SEP pre/post append, truncate to prefill_chunk_size.
        # Decoder lane: conditional EOS append, truncate to max_seq_len.
        self._cls_token_id: Optional[int] = None
        self._sep_token_id: Optional[int] = None
        self._decoder_tokenizer_appends_eos: bool = False
        self._decoder_eos_token_id: Optional[int] = None
        self._prefill_chunk_size = self._raw_metadata.get("prefill_chunk_size", 512)
        self._max_seq_len = 0  # Decoder only; see below.

        if self._model_type == "encoder":
            self._cls_token_id, self._sep_token_id = _get_encoder_special_token_ids(
                self._tokenizer, model
            )
        else:
            self._decoder_tokenizer_appends_eos, self._decoder_eos_token_id = (
                _get_decoder_special_tokens(self._tokenizer, model)
            )
            # Decoder lane truncates to the model's context window rather than the
            # compiled prefill chunk (the C++ action re-checks batch*seq vs chunk).
            self._max_seq_len = self._raw_metadata.get("context_window_size", 32768)
            if self._max_seq_len == -1:
                self._max_seq_len = self._raw_metadata.get("sliding_window_size", -1)
            if self._max_seq_len <= 0:
                raise ValueError(
                    f"Decoder embedding model must have a positive context/sliding window; "
                    f"got max_seq_len={self._max_seq_len}"
                )

        # FFI functions
        self._f_embedding_request = tvm.get_global_func("mlc.serve.EmbeddingRequest")
        self._f_embedding_result_unpack = tvm.get_global_func("mlc.serve.EmbeddingResultUnpack")

        # Threaded engine initialization — wrapped in try so that if ANY step
        # fails after threads are started, we clean up before re-raising.
        try:
            self._module = tvm.get_global_func(
                "mlc.serve.create_threaded_engine", allow_missing=False
            )()
            for key in [
                "init_threaded_engine",
                "run_background_loop",
                "run_background_stream_back_loop",
                "reload",
                "exit_background_loop",
                "add_embedding_request",
                "set_embedding_request_callback",
            ]:
                self._ffi[key] = self._module[key]

            # Init threaded engine with a no-op chat callback (required by C++ init).
            def _noop_chat_callback(delta_outputs):
                pass

            self._ffi["init_threaded_engine"](self._device, _noop_chat_callback, None)

            # Register embedding callback BEFORE reload (per C++ contract).
            self._ffi["set_embedding_request_callback"](self._embedding_callback)

            # Start background threads (same pattern as chat)
            self._bg_loop_thread = threading.Thread(
                target=self._ffi["run_background_loop"], daemon=True
            )
            self._bg_stream_thread = threading.Thread(
                target=self._ffi["run_background_stream_back_loop"], daemon=True
            )
            self._bg_loop_thread.start()
            self._bg_stream_thread.start()

            # Reload with embedding-safe config defaults.
            engine_config = EngineConfig(
                model=model,
                model_lib=model_lib,
                additional_models=[],
                mode="local",
                prefix_cache_mode="disable",
                prefill_mode="chunked",
            )
            self._ffi["reload"](engine_config.asjson())
            # Init succeeded — mark as alive.
            self._terminated = False
        except Exception:
            self._cleanup_init_failure()
            raise

    def _cleanup_init_failure(self) -> None:
        """Best-effort cleanup when __init__ fails after threads may have started."""
        # Signal C++ to exit loops
        exit_fn = self._ffi.get("exit_background_loop")
        if exit_fn is not None:
            try:
                exit_fn()
            except Exception:  # pylint: disable=broad-except
                pass
        # Join any started threads
        for t in [self._bg_loop_thread, self._bg_stream_thread]:
            if t is not None and t.is_alive():
                try:
                    t.join(timeout=5.0)
                except Exception:  # pylint: disable=broad-except
                    pass
        # Clear pending map
        with self._pending_lock:
            self._pending.clear()
        self._terminated = True

    # ---- Read-only properties ----

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def pooling_strategy(self) -> str:
        return self._pooling_strategy

    @property
    def normalize(self) -> bool:
        return self._normalize

    @property
    def metadata(self) -> dict:
        return self._raw_metadata

    @property
    def embedding_metadata(self) -> Optional[dict]:
        return self._embedding_metadata

    # ---- Embedding callback (called from stream-back thread) ----

    def _embedding_callback(self, results) -> None:
        """Called from the C++ stream-back thread with Array<EmbeddingResult>."""
        for result in results:
            unpacked = self._f_embedding_result_unpack(result)
            request_id = str(unpacked[0])
            embeddings_nd = unpacked[1]  # CPU float32 NDArray [num_items, hidden_dim]
            prompt_tokens = int(unpacked[2])

            # Convert NDArray to numpy
            embeddings_np = embeddings_nd.numpy()  # [num_items, hidden_dim]
            embeddings_list = embeddings_np.tolist()

            with self._pending_lock:
                entry = self._pending.pop(request_id, None)

            if entry is None:
                # Request already cleaned up or terminated; silently ignore.
                continue

            entry.result = (embeddings_list, prompt_tokens)

            # Deliver to consumer
            if entry.event is not None:
                # Sync path: wake up blocking waiter
                entry.event.set()
            if entry.future is not None and entry.loop is not None:
                # Async path: deliver via call_soon_threadsafe
                entry.loop.call_soon_threadsafe(
                    _set_future_result, entry.future, (embeddings_list, prompt_tokens)
                )

    def _background_failure_message(self) -> Optional[str]:
        """Return a descriptive error if a background thread has exited unexpectedly."""
        if self._terminated:
            return "Embedding engine has been terminated"
        if self._bg_loop_thread is not None and not self._bg_loop_thread.is_alive():
            return "Embedding engine background loop thread exited before request completion"
        if self._bg_stream_thread is not None and not self._bg_stream_thread.is_alive():
            return "Embedding engine callback thread exited before request completion"
        return None

    # ---- Sync embed ----

    def embed(self, inputs: List[str]) -> Tuple[List[List[float]], int]:
        """Compute embeddings synchronously (blocks until result is ready)."""
        if self._terminated:
            raise RuntimeError("Engine has been terminated")

        request_id = str(uuid.uuid4())

        # Canonicalize inputs per lane.
        token_lists = self._canonicalize_inputs(inputs)

        # Create pending entry with sync event
        entry = _PendingEntry()
        entry.event = threading.Event()

        with self._pending_lock:
            self._pending[request_id] = entry

        # Build and submit — clean up pending on failure so no dangling entry remains.
        try:
            self._submit_embedding_request(request_id, token_lists)
        except Exception:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise

        # Block until callback sets the event, but surface background-thread failures
        # instead of waiting forever when the C++ engine crashes.
        while not entry.event.wait(timeout=_THREAD_WAIT_POLL_SEC):
            failure_message = self._background_failure_message()
            if failure_message is not None:
                with self._pending_lock:
                    self._pending.pop(request_id, None)
                raise RuntimeError(failure_message)

        if entry.result is None:
            raise RuntimeError("Embedding request completed without result (engine terminated?)")

        return entry.result

    # ---- Async embed ----

    async def async_embed(self, inputs: List[str]) -> Tuple[List[List[float]], int]:
        """Compute embeddings asynchronously (non-blocking, uses asyncio Future)."""
        if self._terminated:
            raise RuntimeError("Engine has been terminated")

        request_id = str(uuid.uuid4())

        # Canonicalize inputs per lane.
        token_lists = self._canonicalize_inputs(inputs)

        # Create pending entry with async future
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        entry = _PendingEntry()
        entry.future = future
        entry.loop = loop

        with self._pending_lock:
            self._pending[request_id] = entry

        # Build and submit — clean up pending on failure so no dangling entry/future remains.
        try:
            self._submit_embedding_request(request_id, token_lists)
        except Exception:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise

        while True:
            try:
                return await asyncio.wait_for(asyncio.shield(future), timeout=_THREAD_WAIT_POLL_SEC)
            except asyncio.TimeoutError:
                failure_message = self._background_failure_message()
                if failure_message is None:
                    continue
                with self._pending_lock:
                    self._pending.pop(request_id, None)
                if not future.done():
                    future.cancel()
                raise RuntimeError(failure_message)

    # ---- Internal helpers ----

    def _canonicalize_inputs(self, inputs: List[str]) -> List[List[int]]:
        """Dispatch canonicalization to the right lane helper.

        Encoder: CLS/SEP pre/post append + truncate to ``prefill_chunk_size``.
        Decoder: conditional EOS append + truncate to model context window.
        """
        if self._model_type == "encoder":
            return _canonicalize_encoder_inputs(
                inputs,
                self._tokenizer,
                self._cls_token_id,
                self._sep_token_id,
                self._prefill_chunk_size,
            )
        return _canonicalize_decoder_inputs(
            inputs,
            self._tokenizer,
            self._decoder_tokenizer_appends_eos,
            self._decoder_eos_token_id,
            self._max_seq_len,
        )

    def _submit_embedding_request(
        self, request_id: str, token_lists: List[List[int]]
    ) -> None:
        """Build FFI EmbeddingRequest and submit to C++ engine."""
        pooling_int = _POOLING_STRATEGY_MAP.get(self._pooling_strategy, 0)

        # Build packed args for mlc.serve.EmbeddingRequest:
        # request_id, num_items, then for each item: item_index, num_tokens, token_ids...,
        # then pooling_strategy (int), normalize (bool)
        args: List[Any] = [request_id, len(token_lists)]
        for item_idx, tokens in enumerate(token_lists):
            args.append(item_idx)
            args.append(len(tokens))
            args.extend(tokens)
        args.append(pooling_int)
        args.append(self._normalize)

        emb_request = self._f_embedding_request(*args)
        self._ffi["add_embedding_request"](emb_request)

    def terminate(self) -> None:
        """Stop the threaded engine and clean up."""
        if self._terminated:
            return
        self._terminated = True

        # Signal C++ to exit loops
        exit_fn = self._ffi.get("exit_background_loop")
        if exit_fn is not None:
            try:
                exit_fn()
            except Exception:  # pylint: disable=broad-except
                pass

        # Join background threads (may be None if init failed early)
        for t in [self._bg_loop_thread, self._bg_stream_thread]:
            if t is not None and t.is_alive():
                try:
                    t.join(timeout=5.0)
                except Exception:  # pylint: disable=broad-except
                    pass

        # Clean up pending requests
        with self._pending_lock:
            for req_id, entry in self._pending.items():
                if entry.event is not None:
                    entry.event.set()  # Wake up sync waiters
                if entry.future is not None and not entry.future.done():
                    try:
                        entry.loop.call_soon_threadsafe(
                            entry.future.set_exception,
                            RuntimeError("Embedding engine terminated"),
                        )
                    except Exception:  # pylint: disable=broad-except
                        pass
            self._pending.clear()

    def __del__(self):
        self.terminate()


class _PendingEntry:
    """Tracks a single in-flight embedding request."""

    __slots__ = ("result", "event", "future", "loop")

    def __init__(self):
        self.result: Optional[Tuple[List[List[float]], int]] = None
        self.event: Optional[threading.Event] = None  # For sync path
        self.future: Optional[asyncio.Future] = None  # For async path
        self.loop: Optional[asyncio.AbstractEventLoop] = None  # For async path


def _set_future_result(future: asyncio.Future, result: Any) -> None:
    """Thread-safe helper to set a future's result (only if not done)."""
    if not future.done():
        future.set_result(result)


# ====================================================================
# AsyncMLCEmbeddingEngine — backend selector + async wrapper
# ====================================================================

# Backend selection environment variable
_EMBEDDING_BACKEND_ENV = "MLC_SERVE_EMBEDDING_BACKEND"


class AsyncMLCEmbeddingEngine:
    """Asynchronous embedding inference engine with backend selection.

    This is the main entry point for embedding serving, aligned with
    AsyncMLCEngine for chat. It selects the best backend automatically
    and exposes sync/async embed methods.

    Backend selection (via env var MLC_SERVE_EMBEDDING_BACKEND):
    - "auto" (default): both encoder and decoder models try
      ThreadedEmbeddingRuntime first, fall back to TVMNativeEmbeddingRuntime
      if threaded init fails (e.g. missing prefill function in the compiled
      lib).
    - "cpp": force ThreadedEmbeddingRuntime; fail if not available.
    - "tvm_native": force TVMNativeEmbeddingRuntime.

    Parameters
    ----------
    model : str
        Path to the model weight directory.

    model_lib : str
        Path to the compiled model library (.so/.dylib file).

    device : Union[str, Device]
        Device string, e.g. "auto", "cuda:0", "metal".

    pooling_strategy : Optional[str]
        Pooling strategy: "cls" (first token), "mean" (masked average),
        or "last" (last token). If None, auto-detected based on model type.
    """

    def __init__(
        self,
        model: str,
        model_lib: str,
        device: Union[str, Device] = "auto",
        *,
        pooling_strategy: Optional[str] = None,
        _runtime: Optional[EmbeddingRuntime] = None,
    ) -> None:
        # Create or accept runtime
        if _runtime is not None:
            self._runtime = _runtime
        else:
            self._runtime = self._select_backend(model, model_lib, device, pooling_strategy)

        # Mirror core attributes from the abstract interface
        self.tokenizer = self._runtime.tokenizer
        self.model_type = self._runtime.model_type
        self.pooling_strategy = self._runtime.pooling_strategy
        self.normalize = self._runtime.normalize
        self._metadata = self._runtime.metadata

        # Mirror implementation-specific attributes for backward compatibility
        self.device = getattr(self._runtime, "device", None)
        self.embedding_metadata = getattr(self._runtime, "embedding_metadata", None)
        self._cls_token_id = getattr(self._runtime, "_cls_token_id", None)
        self._sep_token_id = getattr(self._runtime, "_sep_token_id", None)

        # Background thread pool (only used for TVM native runtime fallback)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="embedding"
        )
        self._terminated = False

    @staticmethod
    def _select_backend(
        model: str,
        model_lib: str,
        device: Union[str, Device],
        pooling_strategy: Optional[str],
    ) -> EmbeddingRuntime:
        """Select the embedding runtime backend."""
        backend = os.environ.get(_EMBEDDING_BACKEND_ENV, "auto").lower()

        # Read model metadata to determine model type
        config_path = os.path.join(model, "mlc-chat-config.json")
        model_type = None
        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
            emb_meta = cfg.get("embedding_metadata")
            if emb_meta:
                model_type = emb_meta.get("model_type")

        supports_threaded = model_type in ("encoder", "decoder")

        if backend == "tvm_native":
            logger.info("Embedding backend: tvm_native (forced)")
            return TVMNativeEmbeddingRuntime(
                model, model_lib, device, pooling_strategy=pooling_strategy
            )

        if backend == "cpp":
            if not supports_threaded:
                raise ValueError(
                    f"ThreadedEmbeddingRuntime requires model_type in "
                    f"{{'encoder', 'decoder'}}; got model_type={model_type!r}. "
                    f"Set {_EMBEDDING_BACKEND_ENV}=tvm_native for unsupported model types."
                )
            logger.info(
                "Embedding backend: ThreadedEmbeddingRuntime (forced, model_type=%s)",
                model_type,
            )
            return ThreadedEmbeddingRuntime(
                model, model_lib, device, pooling_strategy=pooling_strategy
            )

        # backend == "auto"
        if supports_threaded:
            try:
                runtime = ThreadedEmbeddingRuntime(
                    model, model_lib, device, pooling_strategy=pooling_strategy
                )
                logger.info(
                    "Embedding backend: ThreadedEmbeddingRuntime (auto, model_type=%s)",
                    model_type,
                )
                return runtime
            except Exception as e:  # pylint: disable=broad-except
                logger.warning(
                    "ThreadedEmbeddingRuntime failed to initialize, "
                    "falling back to TVMNativeEmbeddingRuntime. Error: %s",
                    e,
                )

        logger.info(
            "Embedding backend: tvm_native (auto, model_type=%s)",
            model_type or "unknown",
        )
        return TVMNativeEmbeddingRuntime(
            model, model_lib, device, pooling_strategy=pooling_strategy
        )

    def embed(self, inputs: List[str]) -> Tuple[List[List[float]], int]:
        """Compute embeddings synchronously."""
        return self._runtime.embed(inputs)

    async def async_embed(self, inputs: List[str]) -> Tuple[List[List[float]], int]:
        """Compute embeddings asynchronously.

        If the underlying runtime has a native async_embed (ThreadEncoderRuntime),
        use it directly. Otherwise, fall back to threadpool execution.
        """
        # Check for native async fast-path (duck typing)
        native_async = getattr(self._runtime, "async_embed", None)
        if native_async is not None and asyncio.iscoroutinefunction(native_async):
            return await native_async(inputs)

        # Fallback: run sync embed in threadpool
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.embed, inputs)

    def terminate(self) -> None:
        """Terminate the engine and clean up."""
        if getattr(self, "_terminated", True):
            return
        self._terminated = True

        # Terminate the underlying runtime if it has a terminate method
        if hasattr(self._runtime, "terminate"):
            self._runtime.terminate()

        self._executor.shutdown(wait=False)

    def __del__(self):
        self.terminate()


# Backward-compat aliases for code that still imports the old names.
AsyncEmbeddingEngine = AsyncMLCEmbeddingEngine
# Phase 3 name kept for downstream tests / notebooks. ThreadedEmbeddingRuntime
# handles both encoder and decoder lanes now.
ThreadEncoderRuntime = ThreadedEmbeddingRuntime
