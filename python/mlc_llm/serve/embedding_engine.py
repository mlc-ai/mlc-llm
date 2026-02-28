"""Asynchronous embedding inference engine for encoder and decoder models."""

import asyncio
import concurrent.futures
import json
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import tvm
from tvm import relax
from tvm.runtime import Device, ShapeTuple

from mlc_llm.serve import engine_utils
from mlc_llm.support.auto_device import detect_device
from mlc_llm.tokenizers import Tokenizer


class AsyncEmbeddingEngine:  # pylint: disable=too-many-instance-attributes
    """Asynchronous embedding inference engine.

    Supports both encoder models (BERT-style) and decoder-only embedding models
    (e.g. Qwen3-Embeddings). Uses a ThreadPoolExecutor for background inference
    so that the asyncio event loop is not blocked.

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

    def __init__(  # pylint: disable=too-many-branches
        self,
        model: str,
        model_lib: str,
        device: Union[str, Device] = "auto",
        *,
        pooling_strategy: Optional[str] = None,
    ) -> None:
        # Reuse existing utility: device detection
        self.device = detect_device(device) if isinstance(device, str) else device
        # Reuse existing utility: tokenizer
        self.tokenizer = Tokenizer(model)

        # Load TVM module, metadata, and params via engine_utils helpers
        ex = tvm.runtime.load_module(model_lib)
        vm = relax.VirtualMachine(ex, device=self.device)
        self._mod = vm.module
        self._metadata = json.loads(self._mod["_metadata"]())
        self._params = engine_utils.load_embedding_params(model, self.device, self._metadata)

        # Detect model type and set pooling strategy
        self.model_type = engine_utils.detect_embedding_model_type(self._mod)
        if pooling_strategy is not None:
            self.pooling_strategy = pooling_strategy
        else:
            self.pooling_strategy = "cls" if self.model_type == "encoder" else "last"

        # Initialize model-type-specific functions
        if self.model_type == "encoder":
            self._init_encoder(model)
        else:
            self._init_decoder(model)

        # Background thread pool (1 worker = serialized GPU inference)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="embedding"
        )
        self._terminated = False

    def _init_encoder(self, model: str) -> None:
        """Initialize encoder (BERT-style) model functions and special tokens."""
        self._prefill_func = self._mod["prefill"]
        self._cls_token_id: Optional[int] = None
        self._sep_token_id: Optional[int] = None
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
                ids = list(self.tokenizer.encode(tok_config["cls_token"]))
                if len(ids) == 1:
                    self._cls_token_id = ids[0]
            if self._sep_token_id is None and tok_config.get("sep_token"):
                ids = list(self.tokenizer.encode(tok_config["sep_token"]))
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
                test_tokens = list(self.tokenizer.encode("test"))
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
        if self.model_type == "encoder":
            return self._embed_encoder(inputs)
        return self._embed_decoder(inputs)

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
          3. Mean-pool all window embeddings → final embedding → L2 normalize
          This preserves information from the full text at the cost of N× compute.
        """
        embeddings: List[List[float]] = []
        total_tokens = 0
        prefill_chunk = self._metadata.get("prefill_chunk_size", 512)

        for text in inputs:
            tokens = list(self.tokenizer.encode(text))
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

            tokens_tvm = tvm.runtime.tensor(token_ids, device=self.device)
            mask_tvm = tvm.runtime.tensor(attention_mask, device=self.device)

            output = self._prefill_func(tokens_tvm, mask_tvm, self._params)
            # .numpy() copies to CPU, escaping TVM workspace buffer reuse across calls.
            output_np = output.numpy()  # [1, seq_len, hidden_size]

            # Pooling
            if self.pooling_strategy == "cls":
                pooled = output_np[0, 0, :]
            elif self.pooling_strategy == "mean":
                pooled = output_np[0].mean(axis=0)
            else:  # "last"
                pooled = output_np[0, -1, :]

            # L2 normalize
            pooled = pooled.astype(np.float32)
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
        sliding_window = self._metadata.get("sliding_window_size", -1)
        context_window = self._metadata.get("context_window_size", 32768)
        prefill_chunk = self._metadata.get("prefill_chunk_size", 2048)
        max_seq_len = sliding_window if context_window == -1 else context_window
        assert max_seq_len > 0, (
            f"max_seq_len must be positive, got {max_seq_len} "
            f"(context_window_size={context_window}, sliding_window_size={sliding_window})"
        )
        support_sliding = int(sliding_window != -1)

        # Tokenize all inputs. Prefer tokenizer post-processor output. If absent (older models),
        # fall back to appending eos_token_id when missing.
        token_lists: List[List[int]] = []
        for text in inputs:
            tokens = list(self.tokenizer.encode(text))
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

        # Fast path: all tokens fit in one prefill chunk → batch forward
        if total_tokens <= prefill_chunk and all(len(t) > 0 for t in token_lists):
            return self._batch_embed_decoder(
                token_lists, total_tokens, max_seq_len, prefill_chunk, support_sliding
            )

        # Fallback: sequential chunked prefill per input
        return self._sequential_embed_decoder(
            token_lists, total_tokens, max_seq_len, prefill_chunk, support_sliding
        )

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

        # Concatenate all tokens → embed → batch prefill
        all_tokens = []
        for tokens in token_lists:
            all_tokens.extend(tokens)
        token_ids = tvm.runtime.tensor(np.array(all_tokens, dtype=np.int32), device=self.device)
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
                    np.array(chunk_tokens, dtype=np.int32), device=self.device
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
            norm = np.linalg.norm(pooled)
            if norm > 1e-12:
                pooled = pooled / norm
            embeddings.append(pooled.tolist())

        return embeddings, total_tokens

    def terminate(self) -> None:
        """Terminate the engine and clean up the thread pool."""
        if getattr(self, "_terminated", True):
            return
        self._terminated = True
        self._executor.shutdown(wait=False)

    def __del__(self):
        self.terminate()
