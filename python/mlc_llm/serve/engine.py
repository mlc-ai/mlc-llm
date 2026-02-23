"""The MLC LLM Serving Engine."""

# pylint: disable=too-many-lines

import asyncio
import concurrent.futures
import json
import os
import queue
import sys
import weakref
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    overload,
)

import numpy as np
import tvm
from tvm import relax
from tvm.runtime import Device, ShapeTuple

from mlc_llm.protocol import debug_protocol, openai_api_protocol
from mlc_llm.protocol.generation_config import GenerationConfig
from mlc_llm.serve import data, engine_utils
from mlc_llm.serve.config import EngineConfig
from mlc_llm.support import logging
from mlc_llm.support.auto_device import detect_device
from mlc_llm.tokenizers import TextStreamer, Tokenizer

from . import engine_base

logger = logging.getLogger(__name__)


# Note: we define both AsyncChat and Chat for Python type analysis.
class AsyncChat:  # pylint: disable=too-few-public-methods
    """The proxy class to direct to async chat completions."""

    def __init__(self, engine: weakref.ReferenceType) -> None:
        assert isinstance(engine(), AsyncMLCEngine)
        self.completions = AsyncChatCompletion(engine)


class Chat:  # pylint: disable=too-few-public-methods
    """The proxy class to direct to chat completions."""

    def __init__(self, engine: weakref.ReferenceType) -> None:
        assert isinstance(engine(), MLCEngine)
        self.completions = ChatCompletion(engine)


class AsyncChatCompletion:  # pylint: disable=too-few-public-methods
    """The proxy class to direct to async chat completions."""

    if sys.version_info >= (3, 9):
        engine: weakref.ReferenceType["AsyncMLCEngine"]
    else:
        engine: weakref.ReferenceType

    def __init__(self, engine: weakref.ReferenceType) -> None:
        self.engine = engine

    @overload
    async def create(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        messages: List[Dict[str, Any]],
        stream: Literal[True],
        model: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream_options: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[Literal["none", "auto"], Dict]] = None,
        user: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[openai_api_protocol.ChatCompletionStreamResponse, Any]:
        """Asynchronous streaming chat completion interface with OpenAI API compatibility.
        The method is a coroutine that streams ChatCompletionStreamResponse
        that conforms to OpenAI API one at a time via yield.

        See https://platform.openai.com/docs/api-reference/chat/create for specification.

        Parameters
        ----------
        request_id : Optional[str]
            The optional request id.
            A random one will be generated if it is not given.

        extra_body: Optional[Dict[str, Any]] = None,
            Extra body options to pass to the request.
            Can be used to pass debug config as extra_body["debug_config"]

        Yields
        ------
        stream_response : ChatCompletionStreamResponse
            The stream response conforming to OpenAI API.
            See mlc_llm/protocol/openai_api_protocol.py or
            https://platform.openai.com/docs/api-reference/chat/streaming for specification.

        Raises
        ------
        e : BadRequestError
            BadRequestError is raised when the request is invalid.
        """

    @overload
    async def create(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Literal[False] = False,
        stream_options: Literal[None] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[Literal["none", "auto"], Dict]] = None,
        user: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> openai_api_protocol.ChatCompletionResponse:
        """Asynchronous non-streaming chat completion interface with OpenAI API compatibility.
        The method is a coroutine that streams ChatCompletionStreamResponse
        that conforms to OpenAI API one at a time via yield.

        See https://platform.openai.com/docs/api-reference/chat/create for specification.

        Parameters
        ----------
        request_id : Optional[str]
            The optional request id.
            A random one will be generated if it is not given.

        extra_body: Optional[Dict[str, Any]] = None,
            Extra body options to pass to the request.
            Can be used to pass debug config as extra_body["debug_config"]

        Returns
        -------
        response : ChatCompletionResponse
            The chat completion response conforming to OpenAI API.
            See mlc_llm/protocol/openai_api_protocol.py or
            https://platform.openai.com/docs/api-reference/chat/object for specification.

        Raises
        ------
        e : BadRequestError
            BadRequestError is raised when the request is invalid.
        """

    async def create(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        stream_options: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[Literal["none", "auto"], Dict]] = None,
        user: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Union[
        AsyncGenerator[openai_api_protocol.ChatCompletionStreamResponse, Any],
        openai_api_protocol.ChatCompletionResponse,
    ]:
        """Asynchronous chat completion interface with OpenAI API compatibility.

        See https://platform.openai.com/docs/api-reference/chat/create for specification.

        Parameters
        ----------
        request_id : Optional[str]
            The optional request id.
            A random one will be generated if it is not given.

        extra_body: Optional[Dict[str, Any]] = None,
            Extra body options to pass to the request.
            Can be used to pass debug config as extra_body["debug_config"]

        Raises
        ------
        e : BadRequestError
            BadRequestError is raised when the request is invalid.
        """
        return await self.engine()._chat_completion(  # pylint: disable=protected-access
            messages=messages,
            model=model,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            logit_bias=logit_bias,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
            stop=stop,
            stream=stream,
            stream_options=(
                openai_api_protocol.StreamOptions.model_validate(stream_options)
                if stream_options is not None
                else None
            ),
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            user=user,
            response_format=response_format,
            request_id=request_id,
            debug_config=(extra_body.get("debug_config", None) if extra_body is not None else None),
        )


class ChatCompletion:  # pylint: disable=too-few-public-methods
    """The proxy class to direct to chat completions."""

    if sys.version_info >= (3, 9):
        engine: weakref.ReferenceType["MLCEngine"]
    else:
        engine: weakref.ReferenceType

    def __init__(self, engine: weakref.ReferenceType) -> None:
        self.engine = engine

    @overload
    def create(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        messages: List[Dict[str, Any]],
        stream: Literal[True],
        model: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream_options: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[Literal["none", "auto"], Dict]] = None,
        user: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Iterator[openai_api_protocol.ChatCompletionStreamResponse]:
        """Synchronous streaming chat completion interface with OpenAI API compatibility.
        The method streams back ChatCompletionStreamResponse that conforms to
        OpenAI API one at a time via yield.

        See https://platform.openai.com/docs/api-reference/chat/create for specification.

        Parameters
        ----------
        request_id : Optional[str]
            The optional request id.
            A random one will be generated if it is not given.

        extra_body: Optional[Dict[str, Any]] = None,
            Extra body options to pass to the request.
            Can be used to pass debug config as extra_body["debug_config"]

        Yields
        ------
        stream_response : ChatCompletionStreamResponse
            The stream response conforming to OpenAI API.
            See mlc_llm/protocol/openai_api_protocol.py or
            https://platform.openai.com/docs/api-reference/chat/streaming for specification.

        Raises
        ------
        e : BadRequestError
            BadRequestError is raised when the request is invalid.
        """

    @overload
    def create(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Literal[False] = False,
        stream_options: Literal[None] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[Literal["none", "auto"], Dict]] = None,
        user: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> openai_api_protocol.ChatCompletionResponse:
        """Synchronous non-streaming chat completion interface with OpenAI API compatibility.

        See https://platform.openai.com/docs/api-reference/chat/create for specification.

        Parameters
        ----------
        request_id : Optional[str]
            The optional request id.
            A random one will be generated if it is not given.

        extra_body: Optional[Dict[str, Any]] = None,
            Extra body options to pass to the request.
            Can be used to pass debug config as extra_body["debug_config"]

        Returns
        ------
        response : ChatCompletionResponse
            The chat completion response conforming to OpenAI API.
            See mlc_llm/protocol/openai_api_protocol.py or
            https://platform.openai.com/docs/api-reference/chat/object for specification.

        Raises
        ------
        e : BadRequestError
            BadRequestError is raised when the request is invalid.
        """

    def create(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        stream_options: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[Literal["none", "auto"], Dict]] = None,
        user: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Union[
        Iterator[openai_api_protocol.ChatCompletionStreamResponse],
        openai_api_protocol.ChatCompletionResponse,
    ]:
        """Synchronous chat completion interface with OpenAI API compatibility.

        See https://platform.openai.com/docs/api-reference/chat/create for specification.

        Parameters
        ----------
        request_id : Optional[str]
            The optional request id.
            A random one will be generated if it is not given.

        extra_body: Optional[Dict[str, Any]] = None,
            Extra body options to pass to the request.
            Can be used to pass debug config as extra_body["debug_config"]

        Raises
        ------
        e : BadRequestError
            BadRequestError is raised when the request is invalid.
        """
        return self.engine()._chat_completion(  # pylint: disable=protected-access
            messages=messages,
            model=model,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            logit_bias=logit_bias,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
            stop=stop,
            stream=stream,
            stream_options=(
                openai_api_protocol.StreamOptions.model_validate(stream_options)
                if stream_options is not None
                else None
            ),
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            user=user,
            response_format=response_format,
            request_id=request_id,
            debug_config=(extra_body.get("debug_config", None) if extra_body is not None else None),
        )


class AsyncCompletion:  # pylint: disable=too-few-public-methods
    """The proxy class to direct to async completions."""

    if sys.version_info >= (3, 9):
        engine: weakref.ReferenceType["AsyncMLCEngine"]
    else:
        engine: weakref.ReferenceType

    def __init__(self, engine: weakref.ReferenceType) -> None:
        self.engine = engine

    @overload
    async def create(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        prompt: Union[str, List[int]],
        stream: Literal[True],
        model: Optional[str] = None,
        best_of: int = 1,
        echo: bool = False,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        logprobs: Optional[int] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream_options: Optional[Dict[str, Any]] = None,
        suffix: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        user: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[openai_api_protocol.CompletionResponse, Any]:
        """Asynchronous streaming completion interface with OpenAI API compatibility.
        The method is a coroutine that streams CompletionResponse
        that conforms to OpenAI API one at a time via yield.

        See https://platform.openai.com/docs/api-reference/completions/create for specification.

        Parameters
        ----------
        request_id : Optional[str]
            The optional request id.
            A random one will be generated if it is not given.

        extra_body: Optional[Dict[str, Any]] = None,
            Extra body options to pass to the request.
            Can be used to pass debug config as extra_body["debug_config"]

        Yields
        ------
        stream_response : CompletionResponse
            The stream response conforming to OpenAI API.
            See mlc_llm/protocol/openai_api_protocol.py or
            https://platform.openai.com/docs/api-reference/completions/object for specification.

        Raises
        ------
        e : BadRequestError
            BadRequestError is raised when the request is invalid.
        """

    @overload
    async def create(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        prompt: Union[str, List[int]],
        model: Optional[str] = None,
        best_of: int = 1,
        echo: bool = False,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        logprobs: Optional[int] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Literal[False] = False,
        stream_options: Literal[None] = None,
        suffix: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        user: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> openai_api_protocol.CompletionResponse:
        """Asynchronous non-streaming completion interface with OpenAI API compatibility.

        See https://platform.openai.com/docs/api-reference/completions/create for specification.

        Parameters
        ----------
        request_id : Optional[str]
            The optional request id.
            A random one will be generated if it is not given.

        extra_body: Optional[Dict[str, Any]] = None,
            Extra body options to pass to the request.
            Can be used to pass debug config as extra_body["debug_config"]

        Returns
        ------
        response : CompletionResponse
            The completion response conforming to OpenAI API.
            See mlc_llm/protocol/openai_api_protocol.py or
            https://platform.openai.com/docs/api-reference/completions/object for specification.

        Raises
        ------
        e : BadRequestError
            BadRequestError is raised when the request is invalid.
        """

    async def create(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        prompt: Union[str, List[int]],
        model: Optional[str] = None,
        best_of: int = 1,
        echo: bool = False,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        logprobs: Optional[int] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        stream_options: Optional[Dict[str, Any]] = None,
        suffix: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        user: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Union[
        AsyncGenerator[openai_api_protocol.CompletionResponse, Any],
        openai_api_protocol.CompletionResponse,
    ]:
        """Asynchronous completion interface with OpenAI API compatibility.

        See https://platform.openai.com/docs/api-reference/completions/create for specification.

        Parameters
        ----------
        request_id : Optional[str]
            The optional request id.
            A random one will be generated if it is not given.

        extra_body: Optional[Dict[str, Any]] = None,
            Extra body options to pass to the request.
            Can be used to pass debug config as extra_body["debug_config"]

        Raises
        ------
        e : BadRequestError
            BadRequestError is raised when the request is invalid.
        """
        return await self.engine()._completion(  # pylint: disable=protected-access
            model=model,
            prompt=prompt,
            best_of=best_of,
            echo=echo,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logprobs=logprobs,
            logit_bias=logit_bias,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
            stop=stop,
            stream=stream,
            stream_options=(
                openai_api_protocol.StreamOptions.model_validate(stream_options)
                if stream_options is not None
                else None
            ),
            suffix=suffix,
            temperature=temperature,
            top_p=top_p,
            user=user,
            response_format=response_format,
            request_id=request_id,
            debug_config=(extra_body.get("debug_config", None) if extra_body is not None else None),
        )


class Completion:  # pylint: disable=too-few-public-methods
    """The proxy class to direct to completions."""

    if sys.version_info >= (3, 9):
        engine: weakref.ReferenceType["MLCEngine"]
    else:
        engine: weakref.ReferenceType

    def __init__(self, engine: weakref.ReferenceType) -> None:
        self.engine = engine

    @overload
    def create(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        prompt: Union[str, List[int]],
        stream: Literal[True],
        model: Optional[str] = None,
        best_of: int = 1,
        echo: bool = False,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        logprobs: Optional[int] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream_options: Optional[Dict[str, Any]] = None,
        suffix: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        user: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Iterator[openai_api_protocol.CompletionResponse]:
        """Synchronous streaming completion interface with OpenAI API compatibility.
        The method streams back CompletionResponse that conforms to
        OpenAI API one at a time via yield.

        See https://platform.openai.com/docs/api-reference/completions/create for specification.

        Parameters
        ----------
        request_id : Optional[str]
            The optional request id.
            A random one will be generated if it is not given.

        extra_body: Optional[Dict[str, Any]] = None,
            Extra body options to pass to the request.
            Can be used to pass debug config as extra_body["debug_config"]

        Yields
        ------
        stream_response : CompletionResponse
            The stream response conforming to OpenAI API.
            See mlc_llm/protocol/openai_api_protocol.py or
            https://platform.openai.com/docs/api-reference/completions/object for specification.

        Raises
        ------
        e : BadRequestError
            BadRequestError is raised when the request is invalid.
        """

    @overload
    def create(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        prompt: Union[str, List[int]],
        model: Optional[str] = None,
        best_of: int = 1,
        echo: bool = False,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        logprobs: Optional[int] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Literal[False] = False,
        stream_options: Literal[None] = None,
        suffix: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        user: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> openai_api_protocol.CompletionResponse:
        """Synchronous non-streaming completion interface with OpenAI API compatibility.

        See https://platform.openai.com/docs/api-reference/completions/create for specification.

        Parameters
        ----------
        request_id : Optional[str]
            The optional request id.
            A random one will be generated if it is not given.

        extra_body: Optional[Dict[str, Any]] = None,
            Extra body options to pass to the request.
            Can be used to pass debug config as extra_body["debug_config"]

        Returns
        -------
        response : CompletionResponse
            The completion response conforming to OpenAI API.
            See mlc_llm/protocol/openai_api_protocol.py or
            https://platform.openai.com/docs/api-reference/completions/object for specification.

        Raises
        ------
        e : BadRequestError
            BadRequestError is raised when the request is invalid.
        """

    def create(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        prompt: Union[str, List[int]],
        model: Optional[str] = None,
        best_of: int = 1,
        echo: bool = False,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        logprobs: Optional[int] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        stream_options: Optional[Dict[str, Any]] = None,
        suffix: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        user: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Union[
        Iterator[openai_api_protocol.CompletionResponse],
        openai_api_protocol.CompletionResponse,
    ]:
        """Synchronous completion interface with OpenAI API compatibility.

        See https://platform.openai.com/docs/api-reference/completions/create for specification.

        Parameters
        ----------
        request_id : Optional[str]
            The optional request id.
            A random one will be generated if it is not given.

        extra_body: Optional[Dict[str, Any]] = None,
            Extra body options to pass to the request.
            Can be used to pass debug config as extra_body["debug_config"]

        Raises
        ------
        e : BadRequestError
            BadRequestError is raised when the request is invalid.
        """
        return self.engine()._completion(  # pylint: disable=protected-access
            model=model,
            prompt=prompt,
            best_of=best_of,
            echo=echo,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logprobs=logprobs,
            logit_bias=logit_bias,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
            stop=stop,
            stream=stream,
            stream_options=(
                openai_api_protocol.StreamOptions.model_validate(stream_options)
                if stream_options is not None
                else None
            ),
            suffix=suffix,
            temperature=temperature,
            top_p=top_p,
            user=user,
            response_format=response_format,
            request_id=request_id,
            debug_config=(extra_body.get("debug_config", None) if extra_body is not None else None),
        )


class AsyncMLCEngine(engine_base.MLCEngineBase):
    """The AsyncMLCEngine in MLC LLM that provides the asynchronous
    interfaces with regard to OpenAI API.

    Parameters
    ----------
    model : str
        A path to ``mlc-chat-config.json``, or an MLC model directory that contains
        `mlc-chat-config.json`.
        It can also be a link to a HF repository pointing to an MLC compiled model.

    device: Union[str, Device]
        The device used to deploy the model such as "cuda" or "cuda:0".
        Will default to "auto" and detect from local available GPUs if not specified.

    model_lib : Optional[str]
        The full path to the model library file to use (e.g. a ``.so`` file).
        If unspecified, we will use the provided ``model`` to search over possible paths.
        It the model lib is not found, it will be compiled in a JIT manner.

    mode : Literal["local", "interactive", "server"]
        The engine mode in MLC LLM.
        We provide three preset modes: "local", "interactive" and "server".
        The default mode is "local".
        The choice of mode decides the values of "max_num_sequence", "max_total_sequence_length"
        and "prefill_chunk_size" when they are not explicitly specified.
        1. Mode "local" refers to the local server deployment which has low
        request concurrency. So the max batch size will be set to 4, and max
        total sequence length and prefill chunk size are set to the context
        window size (or sliding window size) of the model.
        2. Mode "interactive" refers to the interactive use of server, which
        has at most 1 concurrent request. So the max batch size will be set to 1,
        and max total sequence length and prefill chunk size are set to the context
        window size (or sliding window size) of the model.
        3. Mode "server" refers to the large server use case which may handle
        many concurrent request and want to use GPU memory as much as possible.
        In this mode, we will automatically infer the largest possible max batch
        size and max total sequence length.

        You can manually specify arguments "max_num_sequence", "max_total_sequence_length" and
        "prefill_chunk_size" to override the automatic inferred values.

    engine_config : Optional[EngineConfig]
        Additional configurable arguments of MLC engine.
        See class "EngineConfig" for more detail.

    enable_tracing : bool
        A boolean indicating if to enable event logging for requests.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        model: str,
        device: Union[str, Device] = "auto",
        *,
        model_lib: Optional[str] = None,
        mode: Literal["local", "interactive", "server"] = "local",
        engine_config: Optional[EngineConfig] = None,
        enable_tracing: bool = False,
    ) -> None:
        super().__init__(
            "async",
            model=model,
            device=device,
            model_lib=model_lib,
            mode=mode,
            engine_config=engine_config,
            enable_tracing=enable_tracing,
        )
        self.chat = AsyncChat(weakref.ref(self))
        self.completions = AsyncCompletion(weakref.ref(self))

    async def abort(self, request_id: str) -> None:
        """Generation abortion interface.

        Parameters
        ---------
        request_id : str
            The id of the request to abort.
        """
        self._abort(request_id)

    async def metrics(self) -> engine_base.EngineMetrics:
        """Get engine metrics

        Returns
        -------
        metrics: EngineMetrics
            The engine metrics
        """
        # pylint: disable=protected-access
        return await engine_base._async_query_engine_metrics(self)

    async def _chat_completion(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        stream_options: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[Literal["none", "auto"], Dict]] = None,
        user: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        debug_config: Optional[Dict[str, Any]] = None,
    ) -> Union[
        AsyncGenerator[openai_api_protocol.ChatCompletionStreamResponse, Any],
        openai_api_protocol.ChatCompletionResponse,
    ]:
        """Asynchronous chat completion internal interface with OpenAI API compatibility.

        See https://platform.openai.com/docs/api-reference/chat/create for specification.

        Parameters
        ----------
        request_id : Optional[str]
            The optional request id.
            A random one will be generated if it is not given.
            Extra body options to pass to the request.
            Can be used to pass debug config as extra_body["debug_config"]

        debug_config: Optional[Dict[str, Any]] = None,
            Debug config body options to pass to the request.

        Raises
        ------
        e : BadRequestError
            BadRequestError is raised when the request is invalid.
        """
        if request_id is None:
            request_id = f"chatcmpl-{engine_utils.random_uuid()}"

        chatcmpl_generator = self._handle_chat_completion(
            openai_api_protocol.ChatCompletionRequest(
                messages=[
                    openai_api_protocol.ChatCompletionMessage.model_validate(message)
                    for message in messages
                ],
                model=model,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                logit_bias=logit_bias,
                max_tokens=max_tokens,
                n=n,
                seed=seed,
                stop=stop,
                stream=stream,
                stream_options=(
                    openai_api_protocol.StreamOptions.model_validate(stream_options)
                    if stream_options is not None
                    else None
                ),
                temperature=temperature,
                top_p=top_p,
                tools=(
                    [openai_api_protocol.ChatTool.model_validate(tool) for tool in tools]
                    if tools is not None
                    else None
                ),
                tool_choice=tool_choice,
                user=user,
                response_format=(
                    openai_api_protocol.RequestResponseFormat.model_validate(response_format)
                    if response_format is not None
                    else None
                ),
                debug_config=(
                    debug_protocol.DebugConfig.model_validate(debug_config)
                    if debug_config is not None
                    else None
                ),
            ),
            request_id=request_id,
            request_final_usage_include_extra=True,
        )
        if stream:
            # Stream response.
            return chatcmpl_generator
        # Normal response.
        output_texts = ["" for _ in range(n)]
        finish_reasons: List[Optional[str]] = [None for _ in range(n)]
        logprob_results: Optional[List[List[openai_api_protocol.LogProbsContent]]] = (
            [[] for _ in range(n)] if logprobs else None
        )
        request_final_usage = None
        try:
            async for response in chatcmpl_generator:
                # when usage is not None this is the last chunk
                if response.usage is not None:
                    request_final_usage = response.usage
                    continue
                for choice in response.choices:
                    assert isinstance(choice.delta.content, str)
                    output_texts[choice.index] += choice.delta.content
                    if choice.finish_reason is not None and finish_reasons[choice.index] is None:
                        finish_reasons[choice.index] = choice.finish_reason
                    if choice.logprobs is not None:
                        assert logprob_results is not None
                        logprob_results[  # pylint: disable=unsupported-assignment-operation
                            choice.index
                        ] += choice.logprobs.content
        except asyncio.CancelledError:  # pylint: disable=try-except-raise
            # for cancelled error, we can simply pass it through
            raise
        except Exception as err:  # pylint: disable=broad-exception-caught
            logger.error("Error in chat completion with request ID %s: %s", request_id, err)
            raise

        assert all(finish_reason is not None for finish_reason in finish_reasons)
        use_function_calling, tool_calls_list = engine_base.process_function_call_output(
            output_texts, finish_reasons
        )
        return engine_base.wrap_chat_completion_response(
            request_id=request_id,
            model=model,
            output_texts=output_texts,
            finish_reasons=finish_reasons,
            tool_calls_list=tool_calls_list,
            logprob_results=logprob_results,
            use_function_calling=use_function_calling,
            usage=request_final_usage,
        )

    async def _completion(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        prompt: Union[str, List[int]],
        model: Optional[str] = None,
        best_of: int = 1,
        echo: bool = False,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        logprobs: Optional[int] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        stream_options: Optional[Dict[str, Any]] = None,
        suffix: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        user: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        debug_config: Optional[Dict[str, Any]] = None,
    ) -> Union[
        AsyncGenerator[openai_api_protocol.CompletionResponse, Any],
        openai_api_protocol.CompletionResponse,
    ]:
        """Asynchronous completion internal interface with OpenAI API compatibility.

        See https://platform.openai.com/docs/api-reference/completions/create for specification.

        Parameters
        ----------
        request_id : Optional[str]
            The optional request id.
            A random one will be generated if it is not given.

        debug_config: Optional[Dict[str, Any]] = None,
            Extra debug options to pass to the request.

        Raises
        ------
        e : BadRequestError
            BadRequestError is raised when the request is invalid.
        """
        if request_id is None:
            request_id = f"cmpl-{engine_utils.random_uuid()}"
        cmpl_generator = self._handle_completion(
            openai_api_protocol.CompletionRequest(
                model=model,
                prompt=prompt,
                best_of=best_of,
                echo=echo,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logprobs=logprobs,
                logit_bias=logit_bias,
                max_tokens=max_tokens,
                n=n,
                seed=seed,
                stop=stop,
                stream=stream,
                stream_options=(
                    openai_api_protocol.StreamOptions.model_validate(stream_options)
                    if stream_options is not None
                    else None
                ),
                suffix=suffix,
                temperature=temperature,
                top_p=top_p,
                user=user,
                response_format=(
                    openai_api_protocol.RequestResponseFormat.model_validate(response_format)
                    if response_format is not None
                    else None
                ),
                debug_config=(
                    debug_protocol.DebugConfig.model_validate(debug_config)
                    if debug_config is not None
                    else None
                ),
            ),
            request_id=request_id,
            request_final_usage_include_extra=True,
        )
        if stream:
            # Stream response.
            return cmpl_generator
        # Normal response.
        request_final_usage = None
        output_texts = [""] * n
        finish_reasons: List[Optional[str]] = [None] * n
        logprob_results: List[Optional[openai_api_protocol.CompletionLogProbs]] = [None] * n

        async for response in cmpl_generator:
            # this is the final chunk
            if response.usage is not None:
                request_final_usage = response.usage
                continue
            for choice in response.choices:
                output_texts[choice.index] += choice.text
                if choice.finish_reason is not None and finish_reasons[choice.index] is None:
                    finish_reasons[choice.index] = choice.finish_reason
                if choice.logprobs is not None:
                    logprob_results[choice.index] = choice.logprobs

        assert all(finish_reason is not None for finish_reason in finish_reasons)

        return engine_base.wrap_completion_response(
            request_id=request_id,
            model=model,
            output_texts=output_texts,
            finish_reasons=finish_reasons,
            logprob_results=logprob_results,
            usage=request_final_usage,
        )

    async def _handle_chat_completion(
        self,
        request: openai_api_protocol.ChatCompletionRequest,
        request_id: str,
        request_final_usage_include_extra: bool,
    ) -> AsyncGenerator[openai_api_protocol.ChatCompletionStreamResponse, Any]:
        """The implementation fo asynchronous ChatCompletionRequest handling.

        Yields
        ------
        stream_response : ChatCompletionStreamResponse
            The stream response conforming to OpenAI API.
            See mlc_llm/protocol/openai_api_protocol.py or
            https://platform.openai.com/docs/api-reference/chat/streaming for specification.

        Raises
        ------
        e : BadRequestError
            BadRequestError is raised when the request is invalid.
        """
        (
            prompts,
            generation_cfg,
            use_function_calling,
            prompt_length,
        ) = engine_base.process_chat_completion_request(
            request,
            request_id,
            self.state,
            self.model_config_dicts[0],
            self.tokenizer.encode,
            self.max_input_sequence_length,
            self.conv_template.model_copy(deep=True),
        )
        # prompt length is not used
        _ = prompt_length
        finish_reasons: List[Optional[str]] = [None for _ in range(generation_cfg.n)]
        self.state.record_event(request_id, event="invoke generate")
        try:
            async for delta_outputs in self._generate(
                prompts,  # type: ignore[arg-type]
                generation_cfg,
                request_id,  # type: ignore
            ):
                response = engine_base.process_chat_completion_stream_output(
                    delta_outputs,
                    request,
                    request_id,
                    self.state,
                    use_function_calling,
                    finish_reasons,
                )

                if response is not None:
                    if response.usage is not None:
                        if not request_final_usage_include_extra:
                            response.usage.extra = None
                    yield response
            self.state.record_event(request_id, event="finish")
        except asyncio.CancelledError:  # pylint: disable=try-except-raise
            # for cancelled error, we can simply pass it through
            raise
        except Exception as err:  # pylint: disable=broad-exception-caught
            logger.error("Error in _handle_chat_completion for request %s: %s", request_id, err)
            raise

    async def _handle_completion(
        self,
        request: openai_api_protocol.CompletionRequest,
        request_id: str,
        request_final_usage_include_extra: bool,
    ) -> AsyncGenerator[openai_api_protocol.CompletionResponse, Any]:
        """The implementation fo asynchronous CompletionRequest handling.

        Yields
        ------
        stream_response : CompletionResponse
            The stream response conforming to OpenAI API.
            See mlc_llm/protocol/openai_api_protocol.py or
            https://platform.openai.com/docs/api-reference/completions/object for specification.

        Raises
        ------
        e : BadRequestError
            BadRequestError is raised when the request is invalid.
        """
        (
            prompt,
            generation_cfg,
            prompt_length,
            echo_response,
        ) = engine_base.process_completion_request(
            request,
            request_id,
            self.state,
            self.tokenizer,
            self.max_input_sequence_length,
            self.conv_template.model_copy(deep=True),
        )
        _ = prompt_length
        if echo_response is not None:
            yield echo_response

        finish_reasons: List[Optional[str]] = [None] * generation_cfg.n
        self.state.record_event(request_id, event="invoke generate")
        try:
            async for delta_outputs in self._generate(
                prompt,
                generation_cfg,
                request_id,  # type: ignore
            ):
                response = engine_base.process_completion_stream_output(
                    delta_outputs,
                    request,
                    request_id,
                    self.state,
                    finish_reasons,
                )

                if response is not None:
                    if response.usage is not None:
                        if not request_final_usage_include_extra:
                            response.usage.extra = None
                    yield response

            suffix_response = engine_base.create_completion_suffix_response(
                request, request_id, finish_reasons
            )
            if suffix_response is not None:
                yield suffix_response
            self.state.record_event(request_id, event="finish")
        except asyncio.CancelledError:  # pylint: disable=try-except-raise
            # for cancelled error, we can simply pass it through
            raise
        except Exception as err:  # pylint: disable=broad-exception-caught
            logger.error("Error in _handle_completion for request %s: %s", request_id, err)
            raise

    async def _generate(
        self,
        prompt: Union[str, List[int], List[Union[str, List[int], data.Data]]],
        generation_config: GenerationConfig,
        request_id: str,
    ) -> AsyncGenerator[List[engine_base.CallbackStreamOutput], Any]:
        """Internal asynchronous text generation interface of AsyncMLCEngine.
        The method is a coroutine that streams a list of CallbackStreamOutput
        at a time via yield. The returned list length is the number of
        parallel generations specified by `generation_config.n`.

        Parameters
        ----------
        prompt : Union[str, List[int], List[Union[str, List[int], data.Data]]]
            The input prompt in forms of text strings, lists of token ids or data.

        generation_config : GenerationConfig
            The generation config of the request.

        request_id : str
            The unique identifier (in string) or this generation request.

        Yields
        ------
        request_output : List[engine_base.CallbackStreamOutput]
            The delta generated outputs in a list.
            The number of list elements equals to `generation_config.n`,
            and each element corresponds to the delta output of a parallel
            generation.
        """
        if self._terminated:
            raise ValueError("The AsyncThreadedEngine has terminated.")
        self.state.async_lazy_init_event_loop()

        # Create the request with the given id, input data, generation
        # config and the created callback.
        input_data = engine_utils.convert_prompts_to_data(prompt)
        request = self._ffi["create_request"](
            request_id, input_data, generation_config.model_dump_json(by_alias=True)
        )

        # Create the unique async request stream of the request.
        stream = engine_base.AsyncRequestStream()
        if request_id in self.state.async_streamers:
            # Report error in the stream if the request id already exists.
            stream.push(
                RuntimeError(
                    f'The request id "{request_id} already exists. '
                    'Please make sure the request id is unique."'
                )
            )
        else:
            # Record the stream in the tracker
            self.state.async_streamers[request_id] = (
                stream,
                [TextStreamer(self.tokenizer) for _ in range(generation_config.n)],
            )
            self._ffi["add_request"](request)

        def abort_request():
            """clean up"""
            self._abort(request_id)
            logger.info("request %s cancelled", request_id)

        with engine_utils.ErrorCleanupScope(abort_request):
            # Iterate the stream asynchronously and yield the output.
            try:
                async for request_output in stream:
                    yield request_output
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                # for cancelled error, we can simply pass it through
                raise
            except Exception as exception:  # pylint: disable=broad-exception-caught
                logger.error("Exception in _generate for request %s: %s", request_id, exception)
                raise

    def _abort(self, request_id: str):
        """Internal implementation of request abortion."""
        self.state.async_streamers.pop(request_id, None)
        self._ffi["abort_request"](request_id)


class MLCEngine(engine_base.MLCEngineBase):
    """The MLCEngine in MLC LLM that provides the synchronous
    interfaces with regard to OpenAI API.

    Parameters
    ----------
    model : str
        A path to ``mlc-chat-config.json``, or an MLC model directory that contains
        `mlc-chat-config.json`.
        It can also be a link to a HF repository pointing to an MLC compiled model.

    device: Union[str, Device]
        The device used to deploy the model such as "cuda" or "cuda:0".
        Will default to "auto" and detect from local available GPUs if not specified.

    model_lib : Optional[str]
        The full path to the model library file to use (e.g. a ``.so`` file).
        If unspecified, we will use the provided ``model`` to search over possible paths.
        It the model lib is not found, it will be compiled in a JIT manner.

    mode : Literal["local", "interactive", "server"]
        The engine mode in MLC LLM.
        We provide three preset modes: "local", "interactive" and "server".
        The default mode is "local".
        The choice of mode decides the values of "max_num_sequence", "max_total_sequence_length"
        and "prefill_chunk_size" when they are not explicitly specified.
        1. Mode "local" refers to the local server deployment which has low
        request concurrency. So the max batch size will be set to 4, and max
        total sequence length and prefill chunk size are set to the context
        window size (or sliding window size) of the model.
        2. Mode "interactive" refers to the interactive use of server, which
        has at most 1 concurrent request. So the max batch size will be set to 1,
        and max total sequence length and prefill chunk size are set to the context
        window size (or sliding window size) of the model.
        3. Mode "server" refers to the large server use case which may handle
        many concurrent request and want to use GPU memory as much as possible.
        In this mode, we will automatically infer the largest possible max batch
        size and max total sequence length.

        You can manually specify arguments "max_num_sequence", "max_total_sequence_length" and
        "prefill_chunk_size" to override the automatic inferred values.

    engine_config : Optional[EngineConfig]
        Additional configurable arguments of MLC engine.
        See class "EngineConfig" for more detail.

    enable_tracing : bool
        A boolean indicating if to enable event logging for requests.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        model: str,
        device: Union[str, Device] = "auto",
        *,
        model_lib: Optional[str] = None,
        mode: Literal["local", "interactive", "server"] = "local",
        engine_config: Optional[EngineConfig] = None,
        enable_tracing: bool = False,
    ) -> None:
        super().__init__(
            "sync",
            model=model,
            device=device,
            model_lib=model_lib,
            mode=mode,
            engine_config=engine_config,
            enable_tracing=enable_tracing,
        )
        self.chat = Chat(weakref.ref(self))
        self.completions = Completion(weakref.ref(self))

    def abort(self, request_id: str) -> None:
        """Generation abortion interface.

        Parameters
        ---------
        request_id : str
            The id of the request to abort.
        """
        self._ffi["abort_request"](request_id)

    def metrics(self) -> engine_base.EngineMetrics:
        """Get engine metrics

        Returns
        -------
        metrics: EngineMetrics
            The engine metrics
        """
        # pylint: disable=protected-access
        return engine_base._query_engine_metrics(self)

    def _chat_completion(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        stream_options: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[Literal["none", "auto"], Dict]] = None,
        user: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        debug_config: Optional[Dict[str, Any]] = None,
    ) -> Union[
        Iterator[openai_api_protocol.ChatCompletionStreamResponse],
        openai_api_protocol.ChatCompletionResponse,
    ]:
        """Synchronous chat completion internal interface with OpenAI API compatibility.

        See https://platform.openai.com/docs/api-reference/chat/create for specification.

        Parameters
        ----------
        request_id : Optional[str]
            The optional request id.
            A random one will be generated if it is not given.

        debug_config: Optional[Dict[str, Any]] = None,
            Extra debug options to pass to the request.

        Raises
        ------
        e : BadRequestError
            BadRequestError is raised when the request is invalid.
        """
        if request_id is None:
            request_id = f"chatcmpl-{engine_utils.random_uuid()}"

        chatcmpl_generator = self._handle_chat_completion(
            openai_api_protocol.ChatCompletionRequest(
                messages=[
                    openai_api_protocol.ChatCompletionMessage.model_validate(message)
                    for message in messages
                ],
                model=model,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                logit_bias=logit_bias,
                max_tokens=max_tokens,
                n=n,
                seed=seed,
                stop=stop,
                stream=stream,
                stream_options=(
                    openai_api_protocol.StreamOptions.model_validate(stream_options)
                    if stream_options is not None
                    else None
                ),
                temperature=temperature,
                top_p=top_p,
                tools=(
                    [openai_api_protocol.ChatTool.model_validate(tool) for tool in tools]
                    if tools is not None
                    else None
                ),
                tool_choice=tool_choice,
                user=user,
                response_format=(
                    openai_api_protocol.RequestResponseFormat.model_validate(response_format)
                    if response_format is not None
                    else None
                ),
                debug_config=(
                    debug_protocol.DebugConfig.model_validate(debug_config)
                    if debug_config is not None
                    else None
                ),
            ),
            request_id=request_id,
        )
        if stream:
            # Stream response.
            return chatcmpl_generator
        # Normal response.
        request_final_usage = None
        output_texts = ["" for _ in range(n)]
        finish_reasons: List[Optional[str]] = [None for _ in range(n)]
        logprob_results: Optional[List[List[openai_api_protocol.LogProbsContent]]] = (
            [[] for _ in range(n)] if logprobs else None
        )
        for response in chatcmpl_generator:
            # if usage is not None, this is the last chunk
            if response.usage is not None:
                request_final_usage = response.usage
                continue
            for choice in response.choices:
                assert isinstance(choice.delta.content, str)
                output_texts[choice.index] += choice.delta.content
                if choice.finish_reason is not None and finish_reasons[choice.index] is None:
                    finish_reasons[choice.index] = choice.finish_reason
                if choice.logprobs is not None:
                    assert logprob_results is not None
                    logprob_results[  # pylint: disable=unsupported-assignment-operation
                        choice.index
                    ] += choice.logprobs.content

        assert all(finish_reason is not None for finish_reason in finish_reasons)
        use_function_calling, tool_calls_list = engine_base.process_function_call_output(
            output_texts, finish_reasons
        )
        return engine_base.wrap_chat_completion_response(
            request_id=request_id,
            model=model,
            output_texts=output_texts,
            finish_reasons=finish_reasons,
            tool_calls_list=tool_calls_list,
            logprob_results=logprob_results,
            use_function_calling=use_function_calling,
            usage=request_final_usage,
        )

    def _completion(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        prompt: Union[str, List[int]],
        model: Optional[str] = None,
        best_of: int = 1,
        echo: bool = False,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        logprobs: Optional[int] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        stream_options: Optional[Dict[str, Any]] = None,
        suffix: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        user: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        debug_config: Optional[Dict[str, Any]] = None,
    ) -> Union[
        Iterator[openai_api_protocol.CompletionResponse],
        openai_api_protocol.CompletionResponse,
    ]:
        """Synchronous completion internal interface with OpenAI API compatibility.

        See https://platform.openai.com/docs/api-reference/completions/create for specification.

        Parameters
        ----------
        request_id : Optional[str]
            The optional request id.
            A random one will be generated if it is not given.

        debug_config: Optional[Dict[str, Any]] = None,
            Extra debug options to pass to the request.

        Raises
        ------
        e : BadRequestError
            BadRequestError is raised when the request is invalid.
        """
        if request_id is None:
            request_id = f"cmpl-{engine_utils.random_uuid()}"

        cmpl_generator = self._handle_completion(
            openai_api_protocol.CompletionRequest(
                model=model,
                prompt=prompt,
                best_of=best_of,
                echo=echo,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logprobs=logprobs,
                logit_bias=logit_bias,
                max_tokens=max_tokens,
                n=n,
                seed=seed,
                stop=stop,
                stream=stream,
                stream_options=(
                    openai_api_protocol.StreamOptions.model_validate(stream_options)
                    if stream_options is not None
                    else None
                ),
                suffix=suffix,
                temperature=temperature,
                top_p=top_p,
                user=user,
                response_format=(
                    openai_api_protocol.RequestResponseFormat.model_validate(response_format)
                    if response_format is not None
                    else None
                ),
                debug_config=(
                    debug_protocol.DebugConfig.model_validate(debug_config)
                    if debug_config is not None
                    else None
                ),
            ),
            request_id=request_id,
        )
        if stream:
            # Stream response.
            return cmpl_generator
        # Normal response.
        request_final_usage = None
        output_texts = [""] * n
        finish_reasons: List[Optional[str]] = [None] * n
        logprob_results: List[Optional[openai_api_protocol.CompletionLogProbs]] = [None] * n

        for response in cmpl_generator:
            # this is the final chunk
            if response.usage is not None:
                request_final_usage = response.usage
                continue
            for choice in response.choices:
                output_texts[choice.index] += choice.text
                if choice.finish_reason is not None and finish_reasons[choice.index] is None:
                    finish_reasons[choice.index] = choice.finish_reason
                if choice.logprobs is not None:
                    logprob_results[choice.index] = choice.logprobs

        assert all(finish_reason is not None for finish_reason in finish_reasons)
        return engine_base.wrap_completion_response(
            request_id=request_id,
            model=model,
            output_texts=output_texts,
            finish_reasons=finish_reasons,
            logprob_results=logprob_results,
            usage=request_final_usage,
        )

    def _handle_chat_completion(
        self, request: openai_api_protocol.ChatCompletionRequest, request_id: str
    ) -> Iterator[openai_api_protocol.ChatCompletionStreamResponse]:
        """The implementation fo synchronous ChatCompletionRequest handling.

        Yields
        ------
        stream_response : CompletionResponse
            The stream response conforming to OpenAI API.
            See mlc_llm/protocol/openai_api_protocol.py or
            https://platform.openai.com/docs/api-reference/chat/streaming for specification.

        Raises
        ------
        e : BadRequestError
            BadRequestError is raised when the request is invalid.
        """
        (
            prompts,
            generation_cfg,
            use_function_calling,
            prompt_length,
        ) = engine_base.process_chat_completion_request(
            request,
            request_id,
            self.state,
            self.model_config_dicts[0],
            self.tokenizer.encode,
            self.max_input_sequence_length,
            self.conv_template.model_copy(deep=True),
        )
        _ = prompt_length

        finish_reasons: List[Optional[str]] = [None for _ in range(generation_cfg.n)]
        self.state.record_event(request_id, event="invoke generate")
        for delta_outputs in self._generate(prompts, generation_cfg, request_id):  # type: ignore
            response = engine_base.process_chat_completion_stream_output(
                delta_outputs,
                request,
                request_id,
                self.state,
                use_function_calling,
                finish_reasons,
            )
            if response is not None:
                yield response
        self.state.record_event(request_id, event="finish")

    def _handle_completion(
        self, request: openai_api_protocol.CompletionRequest, request_id: str
    ) -> Iterator[openai_api_protocol.CompletionResponse]:
        """The implementation for synchronous CompletionRequest handling.

        Yields
        ------
        stream_response : CompletionResponse
            The stream response conforming to OpenAI API.
            See mlc_llm/protocol/openai_api_protocol.py or
            https://platform.openai.com/docs/api-reference/completions/object for specification.

        Raises
        ------
        e : BadRequestError
            BadRequestError is raised when the request is invalid.
        """
        (
            prompt,
            generation_cfg,
            prompt_length,
            echo_response,
        ) = engine_base.process_completion_request(
            request,
            request_id,
            self.state,
            self.tokenizer,
            self.max_input_sequence_length,
            self.conv_template.model_copy(deep=True),
        )
        _ = prompt_length
        if echo_response is not None:
            yield echo_response

        finish_reasons: List[Optional[str]] = [None for _ in range(generation_cfg.n)]
        self.state.record_event(request_id, event="invoke generate")
        for delta_outputs in self._generate(prompt, generation_cfg, request_id):  # type: ignore
            response = engine_base.process_completion_stream_output(
                delta_outputs,
                request,
                request_id,
                self.state,
                finish_reasons,
            )
            if response is not None:
                yield response

        suffix_response = engine_base.create_completion_suffix_response(
            request, request_id, finish_reasons
        )
        if suffix_response is not None:
            yield suffix_response
        self.state.record_event(request_id, event="finish")

    def _generate(  # pylint: disable=too-many-locals
        self,
        prompt: Union[str, List[int], List[Union[str, List[int], data.Data]]],
        generation_config: GenerationConfig,
        request_id: str,
    ) -> Iterator[List[engine_base.CallbackStreamOutput]]:
        """Internal synchronous text generation interface of MLCEngine.
        The method is a coroutine that streams a list of CallbackStreamOutput
        at a time via yield. The returned list length is the number of
        parallel generations specified by `generation_config.n`
        except for the final chunk(which is always an List of size 1 and comes with usage)

        Parameters
        ----------
        prompt : Union[str, List[int], List[Union[str, List[int], data.Data]]]
            The input prompt in forms of text strings, lists of token ids or data.

        generation_config : GenerationConfig
            The generation config of the request.

        request_id : str
            The unique identifier (in string) or this generation request.

        Yields
        ------
        request_output : List[engine_base.CallbackStreamOutput]
            The delta generated outputs in a list.
            Except for the final chunk, the number of list elements equals to `generation_config.n`,
            and each element corresponds to the delta output of a parallel generation.
        """
        if self._terminated:
            raise ValueError("The engine has terminated.")

        # Create the request with the given id, input data, generation
        # config and the created callback.
        input_data = engine_utils.convert_prompts_to_data(prompt)
        request = self._ffi["create_request"](
            request_id, input_data, generation_config.model_dump_json(by_alias=True)
        )

        # Record the stream in the tracker
        self.state.sync_output_queue = queue.Queue()
        self.state.sync_text_streamers = [
            TextStreamer(self.tokenizer) for _ in range(generation_config.n)
        ]
        self._ffi["add_request"](request)

        def abort_request():
            """clean up request if exception happens"""
            self.abort(request_id)

        # Iterate the stream asynchronously and yield the token.
        with engine_utils.ErrorCleanupScope(abort_request):
            while True:
                delta_outputs = self.state.sync_output_queue.get()
                request_outputs, request_final_usage_json_str = self._request_stream_callback_impl(
                    delta_outputs
                )
                for request_output in request_outputs:  # pylint: disable=use-yield-from
                    yield request_output

                if request_final_usage_json_str is not None:
                    # final chunk, we can break
                    output = engine_base.CallbackStreamOutput(
                        delta_text="",
                        delta_logprob_json_strs=None,
                        finish_reason=None,
                        request_final_usage_json_str=request_final_usage_json_str,
                    )
                    yield [output]
                    break

    def _request_stream_callback_impl(
        self, delta_outputs: List[data.RequestStreamOutput]
    ) -> Tuple[List[List[engine_base.CallbackStreamOutput]], Optional[str]]:
        """The underlying implementation of request stream callback of MLCEngine."""
        batch_outputs: List[List[engine_base.CallbackStreamOutput]] = []
        for delta_output in delta_outputs:
            request_id, stream_outputs = delta_output.unpack()
            self.state.record_event(request_id, event="start callback")

            # final chunk is now always indicated by a chunk
            # where usage json is present
            # the backend engine always streams back this chunk
            # regardless of include_usage option
            is_final_chunk = stream_outputs[0].request_final_usage_json_str is not None
            if is_final_chunk:
                return (batch_outputs, stream_outputs[0].request_final_usage_json_str)

            outputs: List[engine_base.CallbackStreamOutput] = []
            for stream_output, text_streamer in zip(stream_outputs, self.state.sync_text_streamers):
                self.state.record_event(request_id, event="start detokenization")
                delta_text = stream_output.extra_prefix_string + (
                    text_streamer.put(stream_output.delta_token_ids)
                    if len(stream_output.delta_token_ids) > 0
                    else ""
                )
                if stream_output.finish_reason is not None:
                    delta_text += text_streamer.finish()
                self.state.record_event(request_id, event="finish detokenization")

                outputs.append(
                    engine_base.CallbackStreamOutput(
                        delta_text=delta_text,
                        delta_logprob_json_strs=stream_output.delta_logprob_json_strs,
                        finish_reason=stream_output.finish_reason,
                        request_final_usage_json_str=None,
                    )
                )
            batch_outputs.append(outputs)
            self.state.record_event(request_id, event="finish callback")
        return (batch_outputs, None)


# ====== Embedding Engine ======


class AsyncEmbeddingEngine:
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
        self._metadata = engine_utils.extract_embedding_metadata(ex)
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
        # Read EOS token from config — needed for last-token pooling to match HF behavior
        self._decoder_eos_token_id: Optional[int] = None
        config_path = os.path.join(model, "mlc-chat-config.json")
        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                chat_config = json.load(f)
            self._decoder_eos_token_id = chat_config.get("eos_token_id")

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

    def _embed_encoder(self, inputs: List[str]) -> Tuple[List[List[float]], int]:
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
            attention_mask = np.ones((1, seq_len), dtype=np.int32)  # [1, seq_len]

            tokens_tvm = tvm.runtime.tensor(token_ids, device=self.device)
            mask_tvm = tvm.runtime.tensor(attention_mask, device=self.device)

            output = self._prefill_func(tokens_tvm, mask_tvm, self._params)
            output_np = output.numpy()  # [1, seq_len, hidden_size]

            # Pooling
            if self.pooling_strategy == "cls":
                pooled = output_np[0, 0, :]
            elif self.pooling_strategy == "mean":
                pooled = output_np[0].mean(axis=0)
            else:  # "last"
                pooled = output_np[0, -1, :]

            # L2 normalize
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
        support_sliding = int(sliding_window != -1)

        # Tokenize all inputs, appending EOS to match HF tokenizer behavior
        # (HF adds eos_token_id by default; last-token pooling uses its hidden state)
        token_lists: List[List[int]] = []
        for text in inputs:
            tokens = list(self.tokenizer.encode(text))
            if self._decoder_eos_token_id is not None:
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

    def _batch_embed_decoder(
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
        self._kv_state_end_forward(kv_cache)

        # Extract last token hidden state per sequence
        hidden_np = hidden_states.numpy()
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

    def _sequential_embed_decoder(
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
                self._kv_state_end_forward(kv_cache)

            hidden_np = hidden.numpy()
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
