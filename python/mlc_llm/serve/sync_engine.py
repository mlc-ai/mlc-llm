"""The MLC LLM synchronized engine.

NOTE: This engine defined in this file directly wraps the underlying
Engine implementation in C++, is not optimized by multi-threading and
does not offer standard OpenAI API interface.

We do not expose it and use it by default. As of now it mainly serves
the test and debug purpose because of its simplicity.
"""

import json
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import tvm

from mlc_llm.serve import data
from mlc_llm.serve.config import EngineConfig, GenerationConfig
from mlc_llm.serve.engine_base import (
    _parse_models,
    _print_engine_mode_logging_msg,
    _process_model_args,
    detect_device,
)
from mlc_llm.serve.event_trace_recorder import EventTraceRecorder
from mlc_llm.serve.request import Request
from mlc_llm.streamer import TextStreamer
from mlc_llm.support import logging
from mlc_llm.tokenizer import Tokenizer

logging.enable_logging()
logger = logging.getLogger(__name__)


def _create_tvm_module(
    creator: str, ffi_funcs: Sequence[str], creator_args: Optional[List[Any]] = None
) -> Dict[str, Callable]:
    """Internal method to create a module."""
    if creator_args is None:
        creator_args = []
    module = tvm.get_global_func(creator, allow_missing=False)(*creator_args)
    return {key: module[key] for key in ffi_funcs}


class SyncMLCEngine:
    """The Python interface of synchronize request serving engine for MLC LLM.

    The engine receives requests from the "add_request" method. For
    an given request, the engine will keep generating new tokens for
    the request until finish (under certain criterion). After finish,
    the engine will return the generation result through the callback
    function provided by the request.

    NOTE: This engine directly wraps the underlying Engine implementation
    in C++, is not optimized by multi-threading and does not offer standard
    OpenAI API interface. We do not expose it and use it by default.
    As of now it mainly serves the test and debug purpose because of its
    simplicity.

    Parameters
    ----------
    request_stream_callback : Optional[Callable[[str, data.TokenData, Optional[str]], None]]
        The provided callback function to handle the generation
        output. It has the signature of `(str, data.TokenData, bool) -> None`,
        where
        - the first string is the request id,
        - the TokenData contains the generated **delta** token ids since
        the last invocation of the callback on the specific request,
        - the optional string value denotes the finish reason if the
        generation of the request is finished, or None if it has not finished.

        The callback function is optional at construction, but it needs to
        be set before the engine executing requests. This can be done via
        the `set_request_stream_callback` method. Otherwise, the engine will raise
        exception.

    enable_tracing : bool
        A boolean indicating if to enable event logging for requests.

    verbose : bool
        A boolean indicating whether to print logging info in engine.
    """

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
        prefill_chunk_size: Optional[int] = None,
        max_history_size: Optional[int] = None,
        gpu_memory_utilization: Optional[float] = None,
        enable_tracing: bool = False,
        speculative_mode: Literal["disable", "small_draft", "eagle"] = "disable",
        spec_draft_length: int = 4,
        verbose: bool = True,
        request_stream_callback: Optional[Callable[[List[data.RequestStreamOutput]], None]] = None,
    ):
        # - Initialize model loading info.
        models = _parse_models(model, model_lib, additional_models)
        if isinstance(device, str):
            device = detect_device(device)
        assert isinstance(device, tvm.runtime.Device)
        (
            model_args,
            model_config_paths,
            self.conv_template,
        ) = _process_model_args(models, device)

        # - Load the raw model config into dict
        self.model_config_dicts = []
        for i, model_info in enumerate(models):
            model_info.model_lib = model_args[i][1]
            with open(model_config_paths[i], "r", encoding="utf-8") as file:
                self.model_config_dicts.append(json.load(file))

        # - Print logging info for regarding the mode selection.
        if verbose:
            _print_engine_mode_logging_msg(mode)

        self._ffi = _create_tvm_module(
            "mlc.serve.create_engine",
            ffi_funcs=[
                "init",
                "add_request",
                "abort_request",
                "step",
                "stats",
                "reset",
                "get_request_stream_callback",
                "set_request_stream_callback",
                "get_default_generation_config",
            ],
        )
        self.trace_recorder = EventTraceRecorder() if enable_tracing else None

        self._ffi["init"](
            EngineConfig(
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
                verbose=verbose,
            ).asjson(),
            device,
            request_stream_callback,
            self.trace_recorder,
        )
        self.default_generation_cfg_json_str: str = self._ffi["get_default_generation_config"]()
        self.tokenizer = Tokenizer(model_args[0][0])

    def generate(  # pylint: disable=too-many-locals
        self,
        prompts: Union[str, List[str], List[int], List[List[int]], List[List[data.Data]]],
        generation_config: Union[GenerationConfig, List[GenerationConfig]],
    ) -> Tuple[List[List[str]], List[Optional[List[List[str]]]]]:
        """Generate texts for a list of input prompts.
        Each prompt can be a string or a list of token ids.
        The generation for each prompt is independent.
        Return the generation results, one for each prompt.

        Parameters
        ----------
        prompts : Union[str, List[str], List[int], List[List[int]]]
            One or a list of input prompts for text generation.
            Each prompt can be a string or a list of token ids.

        generation_config : Union[GenerationConfig, List[GenerationConfig]]
            The generation config for each requests.
            If the it is a single GenerationConfig instance,
            this config will be shared by all the prompts.
            Otherwise, one generation config is required for every
            prompt.

        Returns
        -------
        output_text : List[List[str]]
            The text generation results, one list of strings for each input prompt.
            The length of each list is the parallel generation `n` in
            generation config.

        output_logprobs_str : List[Optional[List[List[str]]]]
            The logprob strings of each token for each input prompt, or None
            if an input prompt does not require logprobs.
        """
        if isinstance(prompts, str):
            # `prompts` is a single string.
            prompts = [prompts]
        else:
            assert isinstance(prompts, list), (
                "Input `prompts` is expected to be a string, a list of "
                "str, a list of token ids or multiple lists of token ids. "
            )
            if len(prompts) == 0:
                return [], []
            if isinstance(prompts[0], int):
                # `prompts` is a list of token ids
                prompts = [prompts]  # type: ignore

        num_requests = len(prompts)
        if not isinstance(generation_config, list):
            generation_config = [generation_config] * num_requests

        assert (
            len(generation_config) == num_requests
        ), "Number of generation config and number of prompts mismatch"

        num_finished_generations = 0
        output_texts: List[List[str]] = []
        output_logprobs_str: List[Optional[List[List[str]]]] = []
        text_streamers: List[List[TextStreamer]] = []
        for i in range(num_requests):
            output_texts.append([])
            output_logprobs_str.append([] if generation_config[i].logprobs else None)
            text_streamers.append([])
            for _ in range(generation_config[i].n):
                output_texts[i].append("")
                text_streamers[i].append(TextStreamer(self.tokenizer))
                if output_logprobs_str[i] is not None:
                    output_logprobs_str[i].append([])

        num_total_generations = sum(cfg.n for cfg in generation_config)

        # Save a copy of the original function callback since `generate`
        # overrides the callback function.
        # The original callback will be set back later on.
        original_callback = self._ffi["get_request_stream_callback"]()

        # Define the callback function for request generation results
        def request_stream_callback(delta_outputs: List[data.RequestStreamOutput]):
            nonlocal num_finished_generations
            for delta_output in delta_outputs:
                request_id, stream_outputs = delta_output.unpack()
                rid = int(request_id)

                assert len(stream_outputs) == generation_config[rid].n
                for i, (stream_output, text_streamer) in enumerate(
                    zip(stream_outputs, text_streamers[rid])
                ):
                    if output_logprobs_str[rid] is not None:
                        assert stream_output.delta_logprob_json_strs is not None
                        output_logprobs_str[rid][i] += stream_output.delta_logprob_json_strs

                    delta_text = (
                        text_streamer.put(stream_output.delta_token_ids)
                        if len(stream_output.delta_token_ids) > 0
                        else ""
                    )
                    if stream_output.finish_reason is not None:
                        delta_text += text_streamer.finish()

                    output_texts[rid][i] += delta_text
                    if stream_output.finish_reason is not None:
                        num_finished_generations += 1

        # Override the callback function in engine.
        self._ffi["set_request_stream_callback"](request_stream_callback)

        def convert_to_data(prompt: Union[str, List[int], List[data.Data]]) -> List[data.Data]:
            if isinstance(prompt, str):
                return [data.TextData(prompt)]
            if isinstance(prompt[0], int):
                return [data.TokenData(prompt)]  # type: ignore
            return prompt  # type: ignore

        # Add requests to engine.
        for req_id, (prompt, generation_cfg) in enumerate(zip(prompts, generation_config)):
            input_data = convert_to_data(prompt)  # type: ignore
            self.add_request(
                Request(
                    request_id=str(req_id),
                    inputs=input_data,
                    generation_config=generation_cfg,
                    default_generation_config_json_str=self.default_generation_cfg_json_str,
                )
            )

        while num_finished_generations != num_total_generations:
            self.step()

        # Restore the callback function in engine.
        self._ffi["set_request_stream_callback"](original_callback)
        return output_texts, output_logprobs_str

    def add_request(self, request: Request) -> None:
        """Add a new request to the engine.

        Parameters
        ----------
        request : Request
            The request to add.
        """
        self._ffi["add_request"](request)

    def abort_request(self, request_id: str) -> None:
        """Abort the generation of the request corresponding to the input request id.

        Parameters
        ----------
        request_id : str
            The unique id of the request to abort.
        """
        self._ffi["abort_request"](request_id)

    def step(self) -> None:
        """The main function that the engine takes a step of action.

        At each step, the engine may decide to
        - run prefill for one (or more) requests,
        - run one-step decode for the all existing requests
        ...

        In the end of certain actions (e.g., decode), the engine will
        check if any request has finished, and will return the
        generation results for those finished requests.
        """
        self._ffi["step"]()

    def reset(self) -> None:
        """Reset the engine, clean up all running data and statistics."""
        self._ffi["reset"]()

    def stats(self) -> Dict[str, float]:
        """The engine runtime statistics.
        We collect the following entries:
        - single token prefill latency (s/tok): avg latency of processing one token in prefill
        - single token decode latency (s/tok): avg latency of processing one token in decode
        - engine time for prefill (sec)
        - engine time for decode (sec)
        - total number of processed tokens in prefill.
        - total number of processed tokens in decode.
        """
        stats_json_str = self._ffi["stats"]()
        return json.loads(stats_json_str)
