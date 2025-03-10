# pylint: disable=chained-comparison,missing-docstring,too-few-public-methods,too-many-instance-attributes
# pylint: disable=too-many-arguments,too-many-locals,unused-argument,unused-variable
import json
import queue
import threading
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Union

import tvm

from mlc_llm.protocol import debug_protocol, openai_api_protocol
from mlc_llm.serve import engine_utils
from mlc_llm.serve.engine_base import (
    EngineConfig,
    EngineMetrics,
    _check_engine_config,
    _parse_models,
    _process_model_args,
    _query_engine_metrics,
    detect_device,
)
from mlc_llm.tokenizers import Tokenizer


class EngineState:
    sync_queue: queue.Queue

    def get_request_stream_callback(self) -> Callable[[str], None]:
        # ChatCompletionStreamResponse

        def _callback(chat_completion_stream_responses_json_str: str) -> None:
            self._sync_request_stream_callback(chat_completion_stream_responses_json_str)

        return _callback

    def _sync_request_stream_callback(self, chat_completion_stream_responses_json_str: str) -> None:
        # Put the delta outputs to the queue in the unblocking way.
        self.sync_queue.put_nowait(chat_completion_stream_responses_json_str)

    def handle_chat_completion(
        self, ffi: dict, request_json_str: str, include_usage: bool, request_id: str
    ) -> Iterator[openai_api_protocol.ChatCompletionStreamResponse]:
        """Helper class to handle chat completion

        Note
        ----
        ffi is explicitly passed in to avoid cylic dependency
        as ffi will capture EngineState
        """
        self.sync_queue = queue.Queue()

        success = bool(ffi["chat_completion"](request_json_str, request_id))

        try:
            last_chunk_arrived = False
            while not last_chunk_arrived:
                chat_completion_responses_json_str = self.sync_queue.get()
                chat_completion_responses_list = json.loads(chat_completion_responses_json_str)
                for chat_completion_response_json_dict in chat_completion_responses_list:
                    chat_completion_response = (
                        openai_api_protocol.ChatCompletionStreamResponse.model_validate(
                            chat_completion_response_json_dict
                        )
                    )
                    # the chunk with usage is always the last chunk
                    if chat_completion_response.usage is not None:
                        if include_usage:
                            yield chat_completion_response
                        last_chunk_arrived = True
                        break
                    yield chat_completion_response
        except Exception as exception:  # pylint: disable=broad-exception-caught
            ffi["abort"](request_id)
            raise exception


class BackgroundLoops:
    """Helper class to keep track of background loops"""

    def __init__(self, ffi: dict):
        self._ffi = ffi
        # important: avoid self reference in closure
        background_loop = self._ffi["run_background_loop"]
        background_stream_back_loop = self._ffi["run_background_stream_back_loop"]

        # Create the background engine-driving thread and start the loop.
        self._background_loop_thread: threading.Thread = threading.Thread(target=background_loop)
        self._background_stream_back_loop_thread: threading.Thread = threading.Thread(
            target=background_stream_back_loop
        )
        self._background_loop_thread.start()
        self._background_stream_back_loop_thread.start()
        self._terminated = False

    def __del__(self):
        self.terminate()

    def terminate(self):
        if self._terminated:
            return
        self._terminated = True
        self._ffi["exit_background_loop"]()
        self._background_loop_thread.join()
        self._background_stream_back_loop_thread.join()


class Completions:
    """Completions class to be compatible with OpenAI API"""

    _ffi: dict
    _state: EngineState
    _background_loops: BackgroundLoops

    def __init__(self, ffi: dict, state: EngineState, background_loops: BackgroundLoops):
        self._ffi = ffi
        self._state = state
        self._background_loops = background_loops

    def create(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
        logit_bias: Optional[Dict[int, float]] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = True,
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
        if request_id is None:
            request_id = f"chatcmpl-{engine_utils.random_uuid()}"
        debug_config = extra_body.get("debug_config", None) if extra_body is not None else None
        if not stream:
            raise ValueError("JSONFFIEngine only support stream=True")
        request = openai_api_protocol.ChatCompletionRequest(
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
        )
        chatcmpl_generator = self._state.handle_chat_completion(
            self._ffi,
            request.model_dump_json(by_alias=True),
            include_usage=(
                request.stream_options is not None and request.stream_options.include_usage
            ),
            request_id=request_id,
        )
        for response in chatcmpl_generator:  # pylint: disable=use-yield-from
            yield response


class Chat:
    """Chat class to be compatible with OpenAI API"""

    completions: Completions

    def __init__(self, ffi: dict, state: EngineState, background_loops: BackgroundLoops):
        self.completions = Completions(ffi, state, background_loops)


class JSONFFIEngine:
    chat: Chat

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        model: str,
        device: Union[str, tvm.runtime.Device] = "auto",
        *,
        model_lib: Optional[str] = None,
        mode: Literal["local", "interactive", "server"] = "local",
        engine_config: Optional[EngineConfig] = None,
    ) -> None:
        # - Check the fields fields of `engine_config`.
        if engine_config is None:
            engine_config = EngineConfig()
        _check_engine_config(model, model_lib, mode, engine_config)

        # - Initialize model loading info.
        models = _parse_models(model, model_lib, engine_config.additional_models)
        if isinstance(device, str):
            device = detect_device(device)
        assert isinstance(device, tvm.runtime.Device)
        model_args = _process_model_args(models, device, engine_config)[0]

        # - Load the raw model config into dict
        for i, model_info in enumerate(models):
            model_info.model_lib = model_args[i][1]

        # - Initialize engine state and engine.
        self._state = EngineState()
        module = tvm.get_global_func("mlc.json_ffi.CreateJSONFFIEngine", allow_missing=False)()
        self._ffi = {
            key: module[key]
            for key in [
                "init_background_engine",
                "reload",
                "unload",
                "reset",
                "chat_completion",
                "abort",
                "run_background_loop",
                "run_background_stream_back_loop",
                "exit_background_loop",
            ]
        }
        self.tokenizer = Tokenizer(model_args[0][0])
        self._background_loops = BackgroundLoops(self._ffi)

        engine_config.model = model_args[0][0]
        engine_config.model_lib = model_args[0][1]
        engine_config.additional_models = model_args[1:]  # type: ignore
        engine_config.mode = mode
        self.engine_config = engine_config

        self._ffi["init_background_engine"](
            device.device_type, device.device_id, self._state.get_request_stream_callback()
        )
        self._ffi["reload"](self.engine_config.asjson())

        self.chat = Chat(self._ffi, self._state, self._background_loops)

    def metrics(self) -> EngineMetrics:
        """Get the engine metrics."""
        return _query_engine_metrics(self)

    def _raw_chat_completion(
        self, request_json_str: str, include_usage: bool, request_id: str
    ) -> Iterator[openai_api_protocol.ChatCompletionStreamResponse]:
        """Raw chat completion API"""
        return self._state.handle_chat_completion(
            self._ffi, request_json_str, include_usage, request_id
        )

    def terminate(self):
        """Explicitly terminate the engine"""
        self._background_loops.terminate()

    def _test_reload(self):
        self._ffi["reload"](self.engine_config.asjson())

    def _test_reset(self):
        self._ffi["reset"]()

    def _test_unload(self):
        self._ffi["unload"]()
