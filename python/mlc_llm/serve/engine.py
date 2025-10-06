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
    Tuple,
    Union,
    overload,
)

from tvm.runtime import Device

from mlc_llm.protocol import debug_protocol, openai_api_protocol
from mlc_llm.protocol.generation_config import GenerationConfig
from mlc_llm.serve import data, engine_utils
from mlc_llm.serve.config import EngineConfig
from mlc_llm.support import logging
from mlc_llm.tokenizers import TextStreamer

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
                prompts, generation_cfg, request_id  # type: ignore
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
                prompt, generation_cfg, request_id  # type: ignore
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
