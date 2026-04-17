"""Asynchronous embedding inference engine for encoder and decoder models."""

import abc
import asyncio
import concurrent.futures
import json
import os
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import tvm
from tvm import relax
from tvm.runtime import Device, ShapeTuple

from mlc_llm.serve import engine_utils
from mlc_llm.support.auto_device import detect_device
from mlc_llm.tokenizers import Tokenizer

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
        tok_config_path = os.path.join(model, "tokenizer_config.json")
        if os.path.exists(tok_config_path):
            with open(tok_config_path, encoding="utf-8") as f:
                tok_config = json.load(f)
            # Try added_tokens_decoder first (newer HF format)
            added = tok_config.get("added_tokens_decoder", {})
            for tid, info in added.items():
                if info.get("content") == tok_config.get("cls_token"):
                    self._cls_token_id = int(tid)
                if info.get("content") == tok_config.get("sep_token"):
                    self._sep_token_id = int(tid)
            # Fallback: encode the special token strings via tokenizer
            if self._cls_token_id is None and tok_config.get("cls_token"):
                ids = list(self._tokenizer.encode(tok_config["cls_token"]))
                if len(ids) == 1:
                    self._cls_token_id = ids[0]
            if self._sep_token_id is None and tok_config.get("sep_token"):
                ids = list(self._tokenizer.encode(tok_config["sep_token"]))
                if len(ids) == 1:
                    self._sep_token_id = ids[0]

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
        """Compute embeddings for a list of input strings (synchronous).

        Parameters
        ----------
        inputs : List[str]
            The input strings to embed.

        Returns
        -------
        embeddings : List[List[float]]
            The L2-normalized embedding vectors.
        total_tokens : int
            Total number of tokens processed.
        """
        if self._model_type == "encoder":
            return self._embed_encoder(inputs)
        return self._embed_decoder(inputs)

    def _embed_encoder(  # pylint: disable=too-many-locals
        self, inputs: List[str]
    ) -> Tuple[List[List[float]], int]:
        """Encoder model embedding (BERT-style).

        Processes each input individually to avoid batch padding artifacts.

        Encoder uses bidirectional attention, so chunked prefill is NOT possible
        (each token must attend to all other tokens in the full sequence).
        Inputs exceeding prefill_chunk_size are truncated.

        (Additional Strategy)
        TODO: For better long-text support, implement sliding window + mean pooling:
          1. Split text into overlapping windows of prefill_chunk_size (stride=chunk/2)
          2. Encode each window independently
          3. Mean-pool all window embeddings -> final embedding -> L2 normalize
          This preserves information from the full text at the cost of N x compute.
        """
        embeddings: List[List[float]] = []
        total_tokens = 0
        prefill_chunk = self._raw_metadata.get("prefill_chunk_size", 512)

        for text in inputs:
            tokens = list(self._tokenizer.encode(text))
            # Add [CLS] and [SEP] if needed
            if self._cls_token_id is not None and (
                len(tokens) == 0 or tokens[0] != self._cls_token_id
            ):
                tokens = [self._cls_token_id] + tokens
            if self._sep_token_id is not None and (
                len(tokens) == 0 or tokens[-1] != self._sep_token_id
            ):
                tokens = tokens + [self._sep_token_id]

            # Truncate to compiled buffer limit (keep [CLS] at start, [SEP] at end)
            if len(tokens) > prefill_chunk:
                tokens = tokens[:prefill_chunk]
                if self._sep_token_id is not None:
                    tokens[-1] = self._sep_token_id

            seq_len = len(tokens)
            total_tokens += seq_len

            token_ids = np.array([tokens], dtype=np.int32)  # [1, seq_len]
            attention_mask: np.ndarray = np.ones((1, seq_len), dtype=np.int32)  # [1, seq_len]

            tokens_tvm = tvm.runtime.tensor(token_ids, device=self._device)
            mask_tvm = tvm.runtime.tensor(attention_mask, device=self._device)

            output = self._prefill_func(tokens_tvm, mask_tvm, self._params)
            # .numpy() copies to CPU, escaping TVM workspace buffer reuse across calls.
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
        """Decoder model embedding with batch prefill optimization.

        When total tokens fit within prefill_chunk_size, all inputs are processed
        in a single batch forward pass using shared KV cache. Otherwise, falls back
        to sequential chunked prefill per input.
        """
        # Read KV cache config from metadata
        prefill_chunk = self._raw_metadata.get("prefill_chunk_size", 2048)
        max_seq_len = self._raw_metadata.get("context_window_size", 32768)
        if max_seq_len == -1:
            max_seq_len = self._raw_metadata.get("sliding_window_size", -1)
        assert max_seq_len > 0, f"max_seq_len must be positive, got {max_seq_len}"
        support_sliding = int(self._raw_metadata.get("sliding_window_size", -1) != -1)

        # Tokenize all inputs. Prefer tokenizer post-processor output. If absent (older models),
        # fall back to appending eos_token_id when missing.
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

        # Fast path: all tokens fit in one prefill chunk -> batch forward
        if total_tokens <= prefill_chunk and all(len(t) > 0 for t in token_lists):
            return self._batch_embed_decoder(
                token_lists, total_tokens, max_seq_len, prefill_chunk, support_sliding
            )

        # Greedy sub-batching: pack texts into sub-batches that fit within
        # prefill_chunk, preserving input order. Oversize texts (single text
        # exceeding prefill_chunk) fall back to sequential chunked prefill.
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
        """Partition token lists into sub-batches that fit within prefill_chunk.

        Each sub-batch is a tuple of (mode, token_lists, total_token_count).
        Empty token lists are skipped to avoid invalid batch processing.
        """
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

        # Create KV cache for the entire batch
        kv_cache = self._create_kv_cache_func(
            ShapeTuple([batch_size]),
            ShapeTuple([max_seq_len]),
            ShapeTuple([prefill_chunk]),
            ShapeTuple([16]),
            ShapeTuple([support_sliding]),
        )

        # Register all sequences
        seq_ids = list(range(batch_size))
        seq_lens = [len(t) for t in token_lists]
        for sid in seq_ids:
            self._kv_state_add_sequence(kv_cache, sid)

        # Begin forward with all sequences at once
        self._kv_state_begin_forward(kv_cache, ShapeTuple(seq_ids), ShapeTuple(seq_lens))

        # Concatenate all tokens -> embed -> batch prefill
        all_tokens = []
        for tokens in token_lists:
            all_tokens.extend(tokens)
        token_ids = tvm.runtime.tensor(np.array(all_tokens, dtype=np.int32), device=self._device)
        all_embed = self._embed_func(token_ids, self._params)
        all_embed = self._nd_reshape(all_embed, ShapeTuple([1, total_tokens, all_embed.shape[-1]]))

        hidden_states, _ = self._batch_prefill_to_hidden_func(all_embed, kv_cache, self._params)
        # .numpy() copies to CPU, escaping TVM workspace buffer reuse across calls.
        # (torch.from_dlpack is zero-copy and hits aliasing bugs on 2nd+ invocation.)
        hidden_np = hidden_states.numpy()
        self._kv_state_end_forward(kv_cache)
        for sid in seq_ids:
            self._kv_state_remove_sequence(kv_cache, sid)

        # Extract last token hidden state per sequence
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

            # Create KV cache for this single sequence
            kv_cache = self._create_kv_cache_func(
                ShapeTuple([1]),
                ShapeTuple([max_seq_len]),
                ShapeTuple([prefill_chunk]),
                ShapeTuple([16]),
                ShapeTuple([support_sliding]),
            )
            self._kv_state_add_sequence(kv_cache, 0)

            # Process tokens in chunks
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
                # .numpy() copies to CPU, escaping TVM buffer aliasing.
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
# AsyncEmbeddingEngine — thin async wrapper around EmbeddingRuntime
# ====================================================================


class AsyncEmbeddingEngine:
    """Asynchronous embedding inference engine.

    Thin wrapper around an EmbeddingRuntime that adds:
    - ThreadPoolExecutor for non-blocking async inference
    - async_embed convenience method
    - terminate lifecycle management

    Backward-compatible attributes are mirrored from the underlying runtime
    so that existing code accessing engine.tokenizer, engine.model_type, etc.
    continues to work without changes.

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
        or "last" (last token). If None, auto-detected based on model type:
        encoder -> "cls", decoder -> "last".
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
            self._runtime = TVMNativeEmbeddingRuntime(
                model, model_lib, device, pooling_strategy=pooling_strategy
            )

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

        # Background thread pool (1 worker = serialized GPU inference)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="embedding"
        )
        self._terminated = False

    def embed(self, inputs: List[str]) -> Tuple[List[List[float]], int]:
        """Compute embeddings for a list of input strings (synchronous).

        Parameters
        ----------
        inputs : List[str]
            The input strings to embed.

        Returns
        -------
        embeddings : List[List[float]]
            The L2-normalized embedding vectors.
        total_tokens : int
            Total number of tokens processed.
        """
        return self._runtime.embed(inputs)

    async def async_embed(self, inputs: List[str]) -> Tuple[List[List[float]], int]:
        """Compute embeddings asynchronously in a background thread.

        This method does not block the asyncio event loop.

        Parameters
        ----------
        inputs : List[str]
            The input strings to embed.

        Returns
        -------
        embeddings : List[List[float]]
            The L2-normalized embedding vectors.
        total_tokens : int
            Total number of tokens processed.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.embed, inputs)

    def terminate(self) -> None:
        """Terminate the engine and clean up the thread pool."""
        if getattr(self, "_terminated", True):
            return
        self._terminated = True
        self._executor.shutdown(wait=False)

    def __del__(self):
        self.terminate()
