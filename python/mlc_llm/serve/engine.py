"""The MLC LLM Serving Engine."""

# pylint: disable=too-many-lines

import asyncio
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
    Union,
    overload,
)

from tvm.runtime import Device

from mlc_llm.protocol import openai_api_protocol
from mlc_llm.serve import data, engine_utils
from mlc_llm.serve.config import EngineConfig, GenerationConfig
from mlc_llm.serve.request import Request
from mlc_llm.streamer import TextStreamer
from mlc_llm.support import logging

from . import engine_base

logging.enable_logging()
logger = logging.getLogger(__name__)


class Chat:  # pylint: disable=too-few-public-methods
    """The proxy class to direct to chat completions."""

    def __init__(self, engine: weakref.ReferenceType) -> None:
        assert isinstance(engine(), (AsyncEngine, Engine))
        self.completions = (
            AsyncChatCompletion(engine)  # type: ignore
            if isinstance(engine(), AsyncEngine)
            else ChatCompletion(engine)  # type: ignore
        )


class AsyncChatCompletion:  # pylint: disable=too-few-public-methods
    """The proxy class to direct to async chat completions."""

    if sys.version_info >= (3, 9):
        engine: weakref.ReferenceType["AsyncEngine"]
    else:
        engine: weakref.ReferenceType

    def __init__(self, engine: weakref.ReferenceType) -> None:
        self.engine = engine

    @overload
    async def create(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        stream: Literal[True],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 0,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[Literal["none", "auto"], Dict]] = None,
        user: Optional[str] = None,
        ignore_eos: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
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
        model: str,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 0,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Literal[False] = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[Literal["none", "auto"], Dict]] = None,
        user: Optional[str] = None,
        ignore_eos: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
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

    async def create(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 0,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[Literal["none", "auto"], Dict]] = None,
        user: Optional[str] = None,
        ignore_eos: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
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
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            user=user,
            ignore_eos=ignore_eos,
            response_format=response_format,
            request_id=request_id,
        )


class ChatCompletion:  # pylint: disable=too-few-public-methods
    """The proxy class to direct to chat completions."""

    if sys.version_info >= (3, 9):
        engine: weakref.ReferenceType["Engine"]
    else:
        engine: weakref.ReferenceType

    def __init__(self, engine: weakref.ReferenceType) -> None:
        self.engine = engine

    @overload
    def create(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        stream: Literal[True],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 0,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[Literal["none", "auto"], Dict]] = None,
        user: Optional[str] = None,
        ignore_eos: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
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
        model: str,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 0,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Literal[False] = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[Literal["none", "auto"], Dict]] = None,
        user: Optional[str] = None,
        ignore_eos: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> openai_api_protocol.ChatCompletionResponse:
        """Synchronous non-streaming chat completion interface with OpenAI API compatibility.

        See https://platform.openai.com/docs/api-reference/chat/create for specification.

        Parameters
        ----------
        request_id : Optional[str]
            The optional request id.
            A random one will be generated if it is not given.

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
        model: str,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 0,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[Literal["none", "auto"], Dict]] = None,
        user: Optional[str] = None,
        ignore_eos: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
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
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            user=user,
            ignore_eos=ignore_eos,
            response_format=response_format,
            request_id=request_id,
        )


class AsyncCompletion:  # pylint: disable=too-few-public-methods
    """The proxy class to direct to async completions."""

    if sys.version_info >= (3, 9):
        engine: weakref.ReferenceType["AsyncEngine"]
    else:
        engine: weakref.ReferenceType

    def __init__(self, engine: weakref.ReferenceType) -> None:
        self.engine = engine

    @overload
    async def create(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        model: str,
        prompt: Union[str, List[int]],
        stream: Literal[True],
        best_of: int = 1,
        echo: bool = False,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 0,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: int = 16,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        suffix: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        user: Optional[str] = None,
        ignore_eos: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
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
        model: str,
        prompt: Union[str, List[int]],
        best_of: int = 1,
        echo: bool = False,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 0,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: int = 16,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Literal[False] = False,
        suffix: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        user: Optional[str] = None,
        ignore_eos: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> openai_api_protocol.CompletionResponse:
        """Asynchronous non-streaming completion interface with OpenAI API compatibility.

        See https://platform.openai.com/docs/api-reference/completions/create for specification.

        Parameters
        ----------
        request_id : Optional[str]
            The optional request id.
            A random one will be generated if it is not given.

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
        model: str,
        prompt: Union[str, List[int]],
        best_of: int = 1,
        echo: bool = False,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 0,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: int = 16,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        suffix: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        user: Optional[str] = None,
        ignore_eos: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
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
            top_logprobs=top_logprobs,
            logit_bias=logit_bias,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
            stop=stop,
            stream=stream,
            suffix=suffix,
            temperature=temperature,
            top_p=top_p,
            user=user,
            ignore_eos=ignore_eos,
            response_format=response_format,
            request_id=request_id,
        )


class Completion:  # pylint: disable=too-few-public-methods
    """The proxy class to direct to completions."""

    if sys.version_info >= (3, 9):
        engine: weakref.ReferenceType["Engine"]
    else:
        engine: weakref.ReferenceType

    def __init__(self, engine: weakref.ReferenceType) -> None:
        self.engine = engine

    @overload
    def create(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        model: str,
        prompt: Union[str, List[int]],
        stream: Literal[True],
        best_of: int = 1,
        echo: bool = False,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 0,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: int = 16,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        suffix: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        user: Optional[str] = None,
        ignore_eos: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> openai_api_protocol.CompletionResponse:
        """Synchronous streaming completion interface with OpenAI API compatibility.
        The method streams back CompletionResponse that conforms to
        OpenAI API one at a time via yield.

        See https://platform.openai.com/docs/api-reference/completions/create for specification.

        Parameters
        ----------
        request_id : Optional[str]
            The optional request id.
            A random one will be generated if it is not given.

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
        model: str,
        prompt: Union[str, List[int]],
        best_of: int = 1,
        echo: bool = False,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 0,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: int = 16,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Literal[False] = False,
        suffix: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        user: Optional[str] = None,
        ignore_eos: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> Iterator[openai_api_protocol.CompletionResponse]:
        """Synchronous non-streaming completion interface with OpenAI API compatibility.

        See https://platform.openai.com/docs/api-reference/completions/create for specification.

        Parameters
        ----------
        request_id : Optional[str]
            The optional request id.
            A random one will be generated if it is not given.

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

    def create(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        model: str,
        prompt: Union[str, List[int]],
        best_of: int = 1,
        echo: bool = False,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 0,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: int = 16,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        suffix: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        user: Optional[str] = None,
        ignore_eos: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> Iterator[openai_api_protocol.CompletionResponse]:
        """Synchronous completion interface with OpenAI API compatibility.

        See https://platform.openai.com/docs/api-reference/completions/create for specification.

        Parameters
        ----------
        request_id : Optional[str]
            The optional request id.
            A random one will be generated if it is not given.

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
            top_logprobs=top_logprobs,
            logit_bias=logit_bias,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
            stop=stop,
            stream=stream,
            suffix=suffix,
            temperature=temperature,
            top_p=top_p,
            user=user,
            ignore_eos=ignore_eos,
            response_format=response_format,
            request_id=request_id,
        )


class AsyncEngine(engine_base.EngineBase):
    """The AsyncEngine in MLC LLM that provides the asynchronous
    interfaces with regard to OpenAI API.

    Parameters
    ----------
    models : str
        A path to ``mlc-chat-config.json``, or an MLC model directory that contains
        `mlc-chat-config.json`.
        It can also be a link to a HF repository pointing to an MLC compiled model.

    device: Union[str, Device]
        The device used to deploy the model such as "cuda" or "cuda:0".
        Will default to "auto" and detect from local available GPUs if not specified.

    model_lib_path : Optional[str]
        The full path to the model library file to use (e.g. a ``.so`` file).
        If unspecified, we will use the provided ``model`` to search over possible paths.
        It the model lib path is not found, it will be compiled in a JIT manner.

    mode : Literal["local", "interactive", "server"]
        The engine mode in MLC LLM.
        We provide three preset modes: "local", "interactive" and "server".
        The default mode is "local".
        The choice of mode decides the values of "max_batch_size", "max_total_sequence_length"
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

        You can manually specify arguments "max_batch_size", "max_total_sequence_length" and
        "prefill_chunk_size" to override the automatic inferred values.

    additional_models : Optional[List[str]]
        The model paths and (optional) model library paths of additional models
        (other than the main model).
        When engine is enabled with speculative decoding, additional models are needed.
        Each string in the list is either in form "model_path" or "model_path:model_lib_path".
        When the model lib path of a model is not given, JIT model compilation will
        be activated to compile the model automatically.

    max_batch_size : Optional[int]
        The maximum allowed batch size set for the KV cache to concurrently support.

    max_total_sequence_length : Optional[int]
        The KV cache total token capacity, i.e., the maximum total number of tokens that
        the KV cache support. This decides the GPU memory size that the KV cache consumes.
        If not specified, system will automatically estimate the maximum capacity based
        on the vRAM size on GPU.

    prefill_chunk_size : Optional[int]
        The maximum number of tokens the model passes for prefill each time.
        It should not exceed the prefill chunk size in model config.
        If not specified, this defaults to the prefill chunk size in model config.

    gpu_memory_utilization : Optional[float]
        A number in (0, 1) denoting the fraction of GPU memory used by the server in total.
        It is used to infer to maximum possible KV cache capacity.
        When it is unspecified, it defaults to 0.90.
        Under mode "local" or "interactive", the actual memory usage may be
        significantly smaller than this number. Under mode "server", the actual
        memory usage may be slightly larger than this number.

    engine_config : Optional[EngineConfig]
        The Engine execution configuration.
        Currently speculative decoding mode is specified via engine config.
        For example, you can use "--engine-config='spec_draft_length=4;speculative_mode=EAGLE'"
        to specify the eagle-style speculative decoding.
        Check out class `EngineConfig` in mlc_llm/serve/config.py for detailed specification.

    enable_tracing : bool
        A boolean indicating if to enable event logging for requests.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model: str,
        device: Union[str, Device] = "auto",
        *,
        model_lib_path: Optional[str] = None,
        mode: Literal["local", "interactive", "server"] = "local",
        additional_models: Optional[List[str]] = None,
        max_batch_size: Optional[int] = None,
        max_total_sequence_length: Optional[int] = None,
        prefill_chunk_size: Optional[int] = None,
        gpu_memory_utilization: Optional[float] = None,
        engine_config: Optional[EngineConfig] = None,
        enable_tracing: bool = False,
    ) -> None:
        super().__init__(
            "async",
            model=model,
            device=device,
            model_lib_path=model_lib_path,
            mode=mode,
            additional_models=additional_models,
            max_batch_size=max_batch_size,
            max_total_sequence_length=max_total_sequence_length,
            prefill_chunk_size=prefill_chunk_size,
            gpu_memory_utilization=gpu_memory_utilization,
            engine_config=engine_config,
            enable_tracing=enable_tracing,
        )
        self.chat = Chat(weakref.ref(self))
        self.completions = AsyncCompletion(weakref.ref(self))

    async def abort(self, request_id: str) -> None:
        """Generation abortion interface.

        Parameter
        ---------
        request_id : str
            The id of the request to abort.
        """
        self._abort(request_id)

    async def _chat_completion(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 0,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[Literal["none", "auto"], Dict]] = None,
        user: Optional[str] = None,
        ignore_eos: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
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
                temperature=temperature,
                top_p=top_p,
                tools=(
                    [openai_api_protocol.ChatTool.model_validate(tool) for tool in tools]
                    if tools is not None
                    else None
                ),
                tool_choice=tool_choice,
                user=user,
                ignore_eos=ignore_eos,
                response_format=(
                    openai_api_protocol.RequestResponseFormat.model_validate(response_format)
                    if response_format is not None
                    else None
                ),
            ),
            request_id=request_id,
        )
        if stream:
            # Stream response.
            return chatcmpl_generator
        # Normal response.
        num_prompt_tokens = 0
        num_completion_tokens = 0
        output_texts = ["" for _ in range(n)]
        finish_reasons: List[Optional[str]] = [None for _ in range(n)]
        logprob_results: Optional[List[List[openai_api_protocol.LogProbsContent]]] = (
            [[] for _ in range(n)] if logprobs else None
        )
        async for response in chatcmpl_generator:
            num_prompt_tokens = response.usage.prompt_tokens
            num_completion_tokens = response.usage.completion_tokens
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
            num_prompt_tokens=num_prompt_tokens,
            num_completion_tokens=num_completion_tokens,
        )

    async def _completion(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        model: str,
        prompt: Union[str, List[int]],
        best_of: int = 1,
        echo: bool = False,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 0,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: int = 16,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        suffix: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        user: Optional[str] = None,
        ignore_eos: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
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
                top_logprobs=top_logprobs,
                logit_bias=logit_bias,
                max_tokens=max_tokens,
                n=n,
                seed=seed,
                stop=stop,
                stream=stream,
                suffix=suffix,
                temperature=temperature,
                top_p=top_p,
                user=user,
                ignore_eos=ignore_eos,
                response_format=(
                    openai_api_protocol.RequestResponseFormat.model_validate(response_format)
                    if response_format is not None
                    else None
                ),
            ),
            request_id,
        )
        if stream:
            # Stream response.
            return cmpl_generator
        # Normal response.
        num_prompt_tokens = 0
        num_completion_tokens = 0
        output_texts = ["" for _ in range(n)]
        finish_reasons: List[Optional[str]] = [None for _ in range(n)]
        logprob_results: Optional[List[List[openai_api_protocol.LogProbsContent]]] = (
            [[] for _ in range(n)] if logprobs else None
        )

        async for response in cmpl_generator:
            num_prompt_tokens = response.usage.prompt_tokens
            num_completion_tokens = response.usage.completion_tokens
            for choice in response.choices:
                output_texts[choice.index] += choice.text
                if choice.finish_reason is not None and finish_reasons[choice.index] is None:
                    finish_reasons[choice.index] = choice.finish_reason
                if choice.logprobs is not None:
                    assert logprob_results is not None
                    logprob_results[  # pylint: disable=unsupported-assignment-operation
                        choice.index
                    ] += choice.logprobs.content

        assert all(finish_reason is not None for finish_reason in finish_reasons)
        return engine_base.wrap_completion_response(
            request_id=request_id,
            model=model,
            output_texts=output_texts,
            finish_reasons=finish_reasons,
            logprob_results=logprob_results,
            num_prompt_tokens=num_prompt_tokens,
            num_completion_tokens=num_completion_tokens,
        )

    async def _handle_chat_completion(
        self, request: openai_api_protocol.ChatCompletionRequest, request_id: str
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

        finish_reasons: List[Optional[str]] = [None for _ in range(generation_cfg.n)]
        num_completion_tokens = 0
        self.state.record_event(request_id, event="invoke generate")
        async for delta_outputs in self._generate(
            prompts, generation_cfg, request_id  # type: ignore
        ):
            response, num_completion_tokens = engine_base.process_chat_completion_stream_output(
                delta_outputs,
                request_id,
                self.state,
                request.model,
                generation_cfg,
                use_function_calling,
                prompt_length,
                finish_reasons,
                num_completion_tokens,
            )
            if response is not None:
                yield response
        self.state.record_event(request_id, event="finish")

    async def _handle_completion(
        self, request: openai_api_protocol.CompletionRequest, request_id: str
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
        )
        if echo_response is not None:
            yield echo_response

        num_completion_tokens = 0
        finish_reasons: List[Optional[str]] = [None for _ in range(generation_cfg.n)]
        self.state.record_event(request_id, event="invoke generate")
        async for delta_outputs in self._generate(
            prompt, generation_cfg, request_id  # type: ignore
        ):
            response, num_completion_tokens = engine_base.process_completion_stream_output(
                delta_outputs,
                request_id,
                self.state,
                request.model,
                generation_cfg,
                prompt_length,
                finish_reasons,
                num_completion_tokens,
            )
            if response is not None:
                yield response

        suffix_response = engine_base.create_completion_suffix_response(
            request, request_id, prompt_length, finish_reasons, num_completion_tokens
        )
        if suffix_response is not None:
            yield suffix_response
        self.state.record_event(request_id, event="finish")

    async def _generate(
        self,
        prompt: Union[str, List[int], List[Union[str, List[int], data.Data]]],
        generation_config: GenerationConfig,
        request_id: str,
    ) -> AsyncGenerator[List[engine_base.CallbackStreamOutput], Any]:
        """Internal asynchronous text generation interface of AsyncEngine.
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
        request = Request(request_id, input_data, generation_config)

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
            self.state.async_num_unfinished_generations[request_id] = generation_config.n
            self._ffi["add_request"](request)

        # Iterate the stream asynchronously and yield the output.
        try:
            async for request_output in stream:
                yield request_output
        except (
            Exception,
            asyncio.CancelledError,
        ) as exception:  # pylint: disable=broad-exception-caught
            await self.abort(request_id)
            raise exception

    def _abort(self, request_id: str):
        """Internal implementation of request abortion."""
        self.state.async_streamers.pop(request_id, None)
        self.state.async_num_unfinished_generations.pop(request_id, None)
        self._ffi["abort_request"](request_id)


class Engine(engine_base.EngineBase):
    """The Engine in MLC LLM that provides the synchronous
    interfaces with regard to OpenAI API.

    Parameters
    ----------
    models : str
        A path to ``mlc-chat-config.json``, or an MLC model directory that contains
        `mlc-chat-config.json`.
        It can also be a link to a HF repository pointing to an MLC compiled model.

    device: Union[str, Device]
        The device used to deploy the model such as "cuda" or "cuda:0".
        Will default to "auto" and detect from local available GPUs if not specified.

    model_lib_path : Optional[str]
        The full path to the model library file to use (e.g. a ``.so`` file).
        If unspecified, we will use the provided ``model`` to search over possible paths.
        It the model lib path is not found, it will be compiled in a JIT manner.

    mode : Literal["local", "interactive", "server"]
        The engine mode in MLC LLM.
        We provide three preset modes: "local", "interactive" and "server".
        The default mode is "local".
        The choice of mode decides the values of "max_batch_size", "max_total_sequence_length"
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

        You can manually specify arguments "max_batch_size", "max_total_sequence_length" and
        "prefill_chunk_size" to override the automatic inferred values.

    additional_models : Optional[List[str]]
        The model paths and (optional) model library paths of additional models
        (other than the main model).
        When engine is enabled with speculative decoding, additional models are needed.
        Each string in the list is either in form "model_path" or "model_path:model_lib_path".
        When the model lib path of a model is not given, JIT model compilation will
        be activated to compile the model automatically.

    max_batch_size : Optional[int]
        The maximum allowed batch size set for the KV cache to concurrently support.

    max_total_sequence_length : Optional[int]
        The KV cache total token capacity, i.e., the maximum total number of tokens that
        the KV cache support. This decides the GPU memory size that the KV cache consumes.
        If not specified, system will automatically estimate the maximum capacity based
        on the vRAM size on GPU.

    prefill_chunk_size : Optional[int]
        The maximum number of tokens the model passes for prefill each time.
        It should not exceed the prefill chunk size in model config.
        If not specified, this defaults to the prefill chunk size in model config.

    gpu_memory_utilization : Optional[float]
        A number in (0, 1) denoting the fraction of GPU memory used by the server in total.
        It is used to infer to maximum possible KV cache capacity.
        When it is unspecified, it defaults to 0.90.
        Under mode "local" or "interactive", the actual memory usage may be
        significantly smaller than this number. Under mode "server", the actual
        memory usage may be slightly larger than this number.

    engine_config : Optional[EngineConfig]
        The Engine execution configuration.
        Currently speculative decoding mode is specified via engine config.
        For example, you can use "--engine-config='spec_draft_length=4;speculative_mode=EAGLE'"
        to specify the eagle-style speculative decoding.
        Check out class `EngineConfig` in mlc_llm/serve/config.py for detailed specification.

    enable_tracing : bool
        A boolean indicating if to enable event logging for requests.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model: str,
        device: Union[str, Device] = "auto",
        *,
        model_lib_path: Optional[str] = None,
        mode: Literal["local", "interactive", "server"] = "local",
        additional_models: Optional[List[str]] = None,
        max_batch_size: Optional[int] = None,
        max_total_sequence_length: Optional[int] = None,
        prefill_chunk_size: Optional[int] = None,
        gpu_memory_utilization: Optional[float] = None,
        engine_config: Optional[EngineConfig] = None,
        enable_tracing: bool = False,
    ) -> None:
        super().__init__(
            "sync",
            model=model,
            device=device,
            model_lib_path=model_lib_path,
            mode=mode,
            additional_models=additional_models,
            max_batch_size=max_batch_size,
            max_total_sequence_length=max_total_sequence_length,
            prefill_chunk_size=prefill_chunk_size,
            gpu_memory_utilization=gpu_memory_utilization,
            engine_config=engine_config,
            enable_tracing=enable_tracing,
        )
        self.chat = Chat(weakref.ref(self))
        self.completions = Completion(weakref.ref(self))

    def abort(self, request_id: str) -> None:
        """Generation abortion interface.

        Parameter
        ---------
        request_id : str
            The id of the request to abort.
        """
        self._ffi["abort_request"](request_id)

    def _chat_completion(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 0,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[Literal["none", "auto"], Dict]] = None,
        user: Optional[str] = None,
        ignore_eos: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
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
                temperature=temperature,
                top_p=top_p,
                tools=(
                    [openai_api_protocol.ChatTool.model_validate(tool) for tool in tools]
                    if tools is not None
                    else None
                ),
                tool_choice=tool_choice,
                user=user,
                ignore_eos=ignore_eos,
                response_format=(
                    openai_api_protocol.RequestResponseFormat.model_validate(response_format)
                    if response_format is not None
                    else None
                ),
            ),
            request_id=request_id,
        )
        if stream:
            # Stream response.
            return chatcmpl_generator
        # Normal response.
        num_prompt_tokens = 0
        num_completion_tokens = 0
        output_texts = ["" for _ in range(n)]
        finish_reasons: List[Optional[str]] = [None for _ in range(n)]
        logprob_results: Optional[List[List[openai_api_protocol.LogProbsContent]]] = (
            [[] for _ in range(n)] if logprobs else None
        )
        for response in chatcmpl_generator:
            num_prompt_tokens = response.usage.prompt_tokens
            num_completion_tokens = response.usage.completion_tokens
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
            num_prompt_tokens=num_prompt_tokens,
            num_completion_tokens=num_completion_tokens,
        )

    def _completion(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        model: str,
        prompt: Union[str, List[int]],
        best_of: int = 1,
        echo: bool = False,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 0,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: int = 16,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        suffix: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        user: Optional[str] = None,
        ignore_eos: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> Iterator[openai_api_protocol.CompletionResponse]:
        """Synchronous completion internal interface with OpenAI API compatibility.

        See https://platform.openai.com/docs/api-reference/completions/create for specification.

        Parameters
        ----------
        request_id : Optional[str]
            The optional request id.
            A random one will be generated if it is not given.

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
                top_logprobs=top_logprobs,
                logit_bias=logit_bias,
                max_tokens=max_tokens,
                n=n,
                seed=seed,
                stop=stop,
                stream=stream,
                suffix=suffix,
                temperature=temperature,
                top_p=top_p,
                user=user,
                ignore_eos=ignore_eos,
                response_format=(
                    openai_api_protocol.RequestResponseFormat.model_validate(response_format)
                    if response_format is not None
                    else None
                ),
            ),
            request_id,
        )
        if stream:
            # Stream response.
            return cmpl_generator
        # Normal response.
        num_prompt_tokens = 0
        num_completion_tokens = 0
        output_texts = ["" for _ in range(n)]
        finish_reasons: List[Optional[str]] = [None for _ in range(n)]
        logprob_results: Optional[List[List[openai_api_protocol.LogProbsContent]]] = (
            [[] for _ in range(n)] if logprobs else None
        )

        for response in cmpl_generator:
            num_prompt_tokens = response.usage.prompt_tokens
            num_completion_tokens = response.usage.completion_tokens
            for choice in response.choices:
                output_texts[choice.index] += choice.text
                if choice.finish_reason is not None and finish_reasons[choice.index] is None:
                    finish_reasons[choice.index] = choice.finish_reason
                if choice.logprobs is not None:
                    assert logprob_results is not None
                    logprob_results[  # pylint: disable=unsupported-assignment-operation
                        choice.index
                    ] += choice.logprobs.content

        assert all(finish_reason is not None for finish_reason in finish_reasons)
        return engine_base.wrap_completion_response(
            request_id=request_id,
            model=model,
            output_texts=output_texts,
            finish_reasons=finish_reasons,
            logprob_results=logprob_results,
            num_prompt_tokens=num_prompt_tokens,
            num_completion_tokens=num_completion_tokens,
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

        finish_reasons: List[Optional[str]] = [None for _ in range(generation_cfg.n)]
        num_completion_tokens = 0
        self.state.record_event(request_id, event="invoke generate")
        for delta_outputs in self._generate(prompts, generation_cfg, request_id):  # type: ignore
            response, num_completion_tokens = engine_base.process_chat_completion_stream_output(
                delta_outputs,
                request_id,
                self.state,
                request.model,
                generation_cfg,
                use_function_calling,
                prompt_length,
                finish_reasons,
                num_completion_tokens,
            )
            if response is not None:
                yield response
        self.state.record_event(request_id, event="finish")

    def _handle_completion(
        self, request: openai_api_protocol.CompletionRequest, request_id: str
    ) -> Iterator[openai_api_protocol.CompletionResponse]:
        """The implementation fo synchronous CompletionRequest handling.

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
        )
        if echo_response is not None:
            yield echo_response

        num_completion_tokens = 0
        finish_reasons: List[Optional[str]] = [None for _ in range(generation_cfg.n)]
        self.state.record_event(request_id, event="invoke generate")
        for delta_outputs in self._generate(prompt, generation_cfg, request_id):  # type: ignore
            response, num_completion_tokens = engine_base.process_completion_stream_output(
                delta_outputs,
                request_id,
                self.state,
                request.model,
                generation_cfg,
                prompt_length,
                finish_reasons,
                num_completion_tokens,
            )
            if response is not None:
                yield response

        suffix_response = engine_base.create_completion_suffix_response(
            request, request_id, prompt_length, finish_reasons, num_completion_tokens
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
        """Internal synchronous text generation interface of AsyncEngine.
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
            raise ValueError("The engine has terminated.")

        # Create the request with the given id, input data, generation
        # config and the created callback.
        input_data = engine_utils.convert_prompts_to_data(prompt)
        request = Request(request_id, input_data, generation_config)

        # Record the stream in the tracker
        self.state.sync_output_queue = queue.Queue()
        self.state.sync_text_streamers = [
            TextStreamer(self.tokenizer) for _ in range(generation_config.n)
        ]
        self.state.sync_num_unfinished_generations = generation_config.n
        self._ffi["add_request"](request)

        # Iterate the stream asynchronously and yield the token.
        try:
            while self.state.sync_num_unfinished_generations > 0:
                delta_outputs = self.state.sync_output_queue.get()
                request_outputs = self._request_stream_callback_impl(delta_outputs)
                for request_output in request_outputs:
                    yield request_output
        except Exception as exception:  # pylint: disable=broad-exception-caught
            self.abort(request_id)
            raise exception

    def _request_stream_callback_impl(
        self, delta_outputs: List[data.RequestStreamOutput]
    ) -> List[List[engine_base.CallbackStreamOutput]]:
        """The underlying implementation of request stream callback of Engine."""
        batch_outputs: List[List[engine_base.CallbackStreamOutput]] = []
        for delta_output in delta_outputs:
            request_id, stream_outputs = delta_output.unpack()
            self.state.record_event(request_id, event="start callback")
            outputs: List[engine_base.CallbackStreamOutput] = []
            for stream_output, text_streamer in zip(stream_outputs, self.state.sync_text_streamers):
                self.state.record_event(request_id, event="start detokenization")
                delta_text = (
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
                        num_delta_tokens=len(stream_output.delta_token_ids),
                        delta_logprob_json_strs=stream_output.delta_logprob_json_strs,
                        finish_reason=stream_output.finish_reason,
                    )
                )
                if stream_output.finish_reason is not None:
                    self.state.sync_num_unfinished_generations -= 1
            batch_outputs.append(outputs)
            self.state.record_event(request_id, event="finish callback")
        return batch_outputs
