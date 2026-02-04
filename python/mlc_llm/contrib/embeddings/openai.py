# pylint: disable=missing-docstring
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from langchain.embeddings import OpenAIEmbeddings  # pylint: disable=import-error
from langchain_community.embeddings.openai import (  # pylint: disable=import-error
    async_embed_with_retry,
    embed_with_retry,
)

from mlc_llm.support import logging

logger = logging.getLogger(__name__)


class MLCEmbeddings(OpenAIEmbeddings):
    def _chunk_tokens(self, texts: Sequence[str]) -> Tuple[List[List], List[int]]:
        """Tokenize and chunk texts to fit in the model's context window."""
        if not self.embedding_ctx_length:
            raise ValueError(
                "embedding_ctx_length must be defined to use _get_len_safe_embeddings."
            )

        try:
            import tiktoken  # pylint: disable=import-outside-toplevel
        except ImportError as err:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed in order to for OpenAIEmbeddings. "
                "Please install it with `pip install tiktoken`."
            ) from err

        tokens = []
        indices = []
        model_name = self.tiktoken_model_name or self.model
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning("Warning: model not found. Using cl100k_base encoding.")
            model = "cl100k_base"
            encoding = tiktoken.get_encoding(model)
        for i, text in enumerate(texts):
            if self.model.endswith("001"):
                # See: https://github.com/openai/openai-python/issues/418#issuecomment-1525939500
                # replace newlines, which can negatively affect performance.
                text = text.replace("\n", " ")
            token = encoding.encode(
                text,
                allowed_special=self.allowed_special,
                disallowed_special=self.disallowed_special,
            )
            for j in range(0, len(token), self.embedding_ctx_length):
                tokens.append(token[j : j + self.embedding_ctx_length])
                indices.append(i)
        return tokens, indices

    def _batch_embed(
        self, inputs: Sequence, *, chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        batched_embeddings: List[List[float]] = []
        _chunk_size = chunk_size or self.chunk_size
        _iter: Iterable = range(0, len(inputs), _chunk_size)
        if self.show_progress_bar:
            try:
                from tqdm import tqdm  # pylint: disable=import-outside-toplevel

                _iter = tqdm(_iter)
            except ImportError:
                pass

        for i in _iter:
            response = embed_with_retry(
                self,
                input=inputs[i : i + _chunk_size],
                **self._invocation_params,
            )
            batched_embeddings.extend(r["embedding"] for r in response["data"])
        return batched_embeddings

    async def _abatch_embed(
        self, inputs: Sequence, *, chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        batched_embeddings: List[List[float]] = []
        _chunk_size = chunk_size or self.chunk_size
        _iter: Iterable = range(0, len(inputs), _chunk_size)
        if self.show_progress_bar:
            try:
                from tqdm import tqdm  # pylint: disable=import-outside-toplevel

                _iter = tqdm(_iter)
            except ImportError:
                pass

        for i in _iter:
            response = await async_embed_with_retry(
                self,
                input=inputs[i : i + _chunk_size],
                **self._invocation_params,
            )
            batched_embeddings.extend(r["embedding"] for r in response["data"])
        return batched_embeddings

    # please refer to
    # https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
    def _get_len_safe_embeddings(  # pylint: disable=too-many-locals,unused-argument
        self,
        texts: List[str],
        *,
        engine: str,
        chunk_size: Optional[int] = None,
    ) -> List[List[float]]:
        tokens, indices = self._chunk_tokens(texts)
        batched_embeddings = self._batch_embed(tokens, chunk_size=chunk_size)
        results: List[List[List[float]]] = [[] for _ in range(len(texts))]
        num_tokens_in_batch: List[List[int]] = [[] for _ in range(len(texts))]
        for idx, tokens_i, batched_emb in zip(indices, tokens, batched_embeddings):
            results[idx].append(batched_emb)
            num_tokens_in_batch[idx].append(len(tokens_i))

        embeddings = []
        empty_average = embed_with_retry(
            self,
            input="",
            **self._invocation_params,
        )["data"][
            0
        ]["embedding"]
        for _result, num_tokens in zip(results, num_tokens_in_batch):
            if len(_result) == 0:
                average = empty_average
            else:
                average = np.average(_result, axis=0, weights=num_tokens)
            normalized = (average / np.linalg.norm(average)).tolist()
            embeddings.append(normalized)

        return embeddings

    # please refer to
    # https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
    async def _aget_len_safe_embeddings(  # pylint: disable=too-many-locals,unused-argument
        self,
        texts: List[str],
        *,
        engine: str,
        chunk_size: Optional[int] = None,
    ) -> List[List[float]]:
        tokens, indices = self._chunk_tokens(texts)
        batched_embeddings = await self._abatch_embed(tokens, chunk_size=chunk_size)

        results: List[List[List[float]]] = [[] for _ in range(len(texts))]
        num_tokens_in_batch: List[List[int]] = [[] for _ in range(len(texts))]
        for idx, tokens_i, batched_emb in zip(indices, tokens, batched_embeddings):
            results[idx].append(batched_emb)
            num_tokens_in_batch[idx].append(len(tokens_i))

        embeddings = []
        empty_average = (
            await async_embed_with_retry(
                self,
                input="",
                **self._invocation_params,
            )
        )[
            "data"
        ][0]["embedding"]
        for _result, num_tokens in zip(results, num_tokens_in_batch):
            if len(_result) == 0:
                average = empty_average
            else:
                average = np.average(_result, axis=0, weights=num_tokens)
            normalized = (average / np.linalg.norm(average)).tolist()
            embeddings.append(normalized)

        return embeddings

    def embed_documents(
        self, texts: List[str], chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        """Call out to OpenAI's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size of embeddings. If None, will use the chunk size
                specified by the class.

        Returns:
            List of embeddings, one for each text.
        """
        # NOTE: to keep things simple, as long as the embedding_ctx_length is defined,
        # we assume the list may contain texts longer than the maximum context and
        # use length-safe embedding function.
        if self.embedding_ctx_length:
            return self._get_len_safe_embeddings(
                texts, engine=self.deployment, chunk_size=chunk_size
            )

        embeddings = self._batch_embed(texts, chunk_size=chunk_size)
        return [(np.array(e) / np.linalg.norm(e)).tolist() for e in embeddings]

    async def aembed_documents(
        self, texts: List[str], chunk_size: Optional[int] = 0
    ) -> List[List[float]]:
        """Call out to OpenAI's embedding endpoint async for embedding search docs.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size of embeddings. If None, will use the chunk size
                specified by the class.

        Returns:
            List of embeddings, one for each text.
        """
        # NOTE: to keep things simple, as long as the embedding_ctx_length is defined,
        #       we assume the list may contain texts longer than the maximum context and
        #       use length-safe embedding function.
        if self.embedding_ctx_length:
            return await self._aget_len_safe_embeddings(texts, engine=self.deployment)

        embeddings = await self._abatch_embed(texts, chunk_size=chunk_size)
        return [(np.array(e) / np.linalg.norm(e)).tolist() for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        return self.embed_documents([text])[0]

    async def aembed_query(self, text: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint async for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        embeddings = await self.aembed_documents([text])
        return embeddings[0]
