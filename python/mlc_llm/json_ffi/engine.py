# pylint: disable=chained-comparison,missing-docstring,too-few-public-methods,too-many-instance-attributes
# pylint: disable=too-many-arguments,too-many-locals,unused-argument,unused-variable
import json
import queue
import threading
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Union

import tvm

from mlc_llm.protocol import openai_api_protocol
from mlc_llm.serve import engine_utils
from mlc_llm.serve.engine_base import (
    EngineConfig,
    _parse_models,
    _process_model_args,
    detect_device,
)
from mlc_llm.tokenizer import Tokenizer


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


class JSONFFIEngine:
    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        model: str,
        device: Union[str, tvm.runtime.Device] = "auto",
        *,
        model_lib: Optional[str] = None,
        mode: Literal["local", "interactive", "server"] = "local",
        additional_models: Optional[List[str]] = None,
        max_batch_size: Optional[int] = None,
        max_total_sequence_length: Optional[int] = None,
        max_history_size: Optional[int] = None,
        prefill_chunk_size: Optional[int] = None,
        speculative_mode: Literal["disable", "small_draft", "eagle"] = "disable",
        spec_draft_length: int = 4,
        gpu_memory_utilization: Optional[float] = None,
    ) -> None:
        # - Initialize model loading info.
        models = _parse_models(model, model_lib, additional_models)
        if isinstance(device, str):
            device = detect_device(device)
        assert isinstance(device, tvm.runtime.Device)
        model_args = _process_model_args(models, device)[0]

        # TODO(mlc-team) Remove the model config parsing, estimation below
        # in favor of a simple direct passing of parameters into backend.
        # JSONFFIEngine do not have to support automatic mode
        #
        # Instead, its config should default to interactive mode always
        # and allow overrides of parameters through json config via reload
        #
        # This is to simplify the logic of users of JSONFFI
        # since we won't have similar logics in android/iOS
        #
        # - Load the raw model config into dict
        for i, model_info in enumerate(models):
            model_info.model_lib = model_args[i][1]

        # - Initialize engine state and engine.
        self.state = EngineState()
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
                "get_last_error",
                "run_background_loop",
                "run_background_stream_back_loop",
                "exit_background_loop",
            ]
        }
        self.tokenizer = Tokenizer(model_args[0][0])

        def _background_loop():
            self._ffi["run_background_loop"]()

        def _background_stream_back_loop():
            self._ffi["run_background_stream_back_loop"]()

        # Create the background engine-driving thread and start the loop.
        self._background_loop_thread: threading.Thread = threading.Thread(target=_background_loop)
        self._background_stream_back_loop_thread: threading.Thread = threading.Thread(
            target=_background_stream_back_loop
        )
        self._background_loop_thread.start()
        self._background_stream_back_loop_thread.start()
        self._terminated = False

        self.engine_config = EngineConfig(
            model=model_args[0][0],
            model_lib=model_args[0][1],
            additional_models=[model_arg[0] for model_arg in model_args[1:]],
            additional_model_libs=[model_arg[1] for model_arg in model_args[1:]],
            mode=mode,
            gpu_memory_utilization=gpu_memory_utilization,
            kv_cache_page_size=16,
            max_num_sequence=max_batch_size,
            max_total_sequence_length=max_total_sequence_length,
            prefill_chunk_size=prefill_chunk_size,
            max_history_size=max_history_size,
            speculative_mode=speculative_mode,
            spec_draft_length=spec_draft_length,
            verbose=False,
        )

        self._ffi["init_background_engine"](
            device.device_type, device.device_id, self.state.get_request_stream_callback()
        )
        self._ffi["reload"](self.engine_config.asjson())

    def terminate(self):
        self._terminated = True
        self._ffi["exit_background_loop"]()
        self._background_loop_thread.join()
        self._background_stream_back_loop_thread.join()

    def chat_completion(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
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
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[Literal["none", "auto"], Dict]] = None,
        user: Optional[str] = None,
        ignore_eos: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> Iterator[openai_api_protocol.ChatCompletionStreamResponse]:
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
            ).model_dump_json(),
            n=n,
            request_id=request_id,
        )
        for response in chatcmpl_generator:
            yield response

    def _handle_chat_completion(
        self, request_json_str: str, n: int, request_id: str
    ) -> Iterator[openai_api_protocol.ChatCompletionStreamResponse]:
        self.state.sync_queue = queue.Queue()
        num_unfinished_requests = n

        success = bool(self._ffi["chat_completion"](request_json_str, request_id))

        try:
            while num_unfinished_requests > 0:
                chat_completion_responses_json_str = self.state.sync_queue.get()
                chat_completion_responses_list = json.loads(chat_completion_responses_json_str)
                for chat_completion_response_json_dict in chat_completion_responses_list:
                    chat_completion_response = (
                        openai_api_protocol.ChatCompletionStreamResponse.model_validate(
                            chat_completion_response_json_dict
                        )
                    )
                    for choice in chat_completion_response.choices:
                        if choice.finish_reason is not None:
                            num_unfinished_requests -= 1
                    yield chat_completion_response
        except Exception as exception:  # pylint: disable=broad-exception-caught
            self._ffi["abort"](request_id)
            raise exception

    def _test_reload(self):
        self._ffi["reload"](self.engine_config.asjson())

    def _test_reset(self):
        self._ffi["reset"]()

    def _test_unload(self):
        self._ffi["unload"]()
