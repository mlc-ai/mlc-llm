"""Python entrypoint of chat."""

import dataclasses
from typing import Any, Dict, List, Optional, Union

from prompt_toolkit import prompt as get_prompt  # pylint: disable=import-error
from prompt_toolkit.key_binding import KeyBindings  # pylint: disable=import-error

from mlc_llm.json_ffi import JSONFFIEngine
from mlc_llm.protocol import openai_api_protocol
from mlc_llm.serve.config import EngineConfig
from mlc_llm.serve.engine import MLCEngine
from mlc_llm.serve.engine_base import _query_engine_metrics
from mlc_llm.support import argparse
from mlc_llm.support.config import ConfigOverrideBase


def _print_help_str():
    help_str = """You can use the following special commands:
  /help               print the special commands
  /exit               quit the cli
  /stats              print out stats of last request (token/sec)
  /metrics            print out full engine metrics
  /reset              restart a fresh chat
  /set [overrides]    override settings in the generation config. For example,
                      `/set temperature=0.5;top_p=0.8;seed=23;max_tokens=100;stop=str1,str2`
                      Note: Separate stop words in the `stop` option with commas (,).
  Multi-line input: Use escape+enter to start a new line.
"""
    print(help_str)


def _set_up_key_bindings():
    kb = KeyBindings()

    @kb.add("escape", "enter")
    def _(event):
        event.current_buffer.insert_text("\n")

    @kb.add("enter")
    def _(event):
        event.current_buffer.validate_and_handle()

    return kb


@dataclasses.dataclass
class ChatCompletionOverride(ConfigOverrideBase):  # pylint: disable=too-many-instance-attributes
    """Flags for overriding chat completions."""

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None

    @staticmethod
    def from_str(source: str) -> "ChatCompletionOverride":
        """Parse model config override values from a string."""
        parser = argparse.ArgumentParser(description="chat completion override values")
        parser.add_argument("--temperature", type=float, default=None)
        parser.add_argument("--top_p", type=float, default=None)
        parser.add_argument("--frequency_penalty", type=float, default=None)
        parser.add_argument("--presence_penalty", type=float, default=None)
        parser.add_argument("--max_tokens", type=int, default=None)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--stop", type=str, default=None)
        results = parser.parse_args([f"--{i}" for i in source.split(";") if i])
        return ChatCompletionOverride(
            temperature=results.temperature,
            top_p=results.top_p,
            frequency_penalty=results.frequency_penalty,
            presence_penalty=results.presence_penalty,
            max_tokens=results.max_tokens,
            seed=results.seed,
            stop=results.stop.split(",") if results.stop is not None else None,
        )


@dataclasses.dataclass
class ModelConfigOverride(ConfigOverrideBase):  # pylint: disable=too-many-instance-attributes
    """Flags for overriding model config."""

    context_window_size: Optional[int] = None
    sliding_window_size: Optional[int] = None
    prefill_chunk_size: Optional[int] = None
    attention_sink_size: Optional[int] = None
    tensor_parallel_shards: Optional[int] = None
    pipeline_parallel_stages: Optional[int] = None
    opt: Optional[str] = None

    @staticmethod
    def from_str(source: str) -> "ModelConfigOverride":
        """Parse model config override values from a string."""
        parser = argparse.ArgumentParser(description="model config override values")
        parser.add_argument("--tensor_parallel_shards", type=int, default=None)
        parser.add_argument("--pipeline_parallel_stages", type=int, default=None)
        parser.add_argument("--opt", type=str, default=None)
        parser.add_argument("--context_window_size", type=int, default=None)
        parser.add_argument("--sliding_window_size", type=int, default=None)
        parser.add_argument("--prefill_chunk_size", type=int, default=None)
        parser.add_argument("--attention_sink_size", type=int, default=None)

        results = parser.parse_args([f"--{i}" for i in source.split(";") if i])
        return ModelConfigOverride(
            tensor_parallel_shards=results.tensor_parallel_shards,
            pipeline_parallel_stages=results.pipeline_parallel_stages,
            opt=results.opt,
            context_window_size=results.context_window_size,
            sliding_window_size=results.sliding_window_size,
            prefill_chunk_size=results.prefill_chunk_size,
            attention_sink_size=results.attention_sink_size,
        )


class ChatState:
    """Simple helper class to manage chat state.

    Chat state wraps around a  engine instance
    and exposes the minimum set of tools to perform
    interactive chat. It provides support for mlc_llm chat.
    It also can be used to do interactive debugging
    with different engine instance.

    Examples
    --------
    .. code:: python

        from openai import OpenAI
        from mlc_llm import MLCEngine
        from mlc_llm.serve import PopenServer
        from mlc_llm.interface.chat import ChatState

        def chat_with_engine(model):
            # hookup with MLCEngine
            ChatState(MLCEngine(model)).chat()

        def chat_with_server(model):
            # hookup with AsyncMLCEngine backed api server
            with PopenServer(model) as server:
                ChatState(
                    OpenAI(base_url=server.openai_v1_base_url, api_key="None")
                ).chat()
    """

    history: List[Dict[str, Any]]
    history_begin: int
    # kwargs passed to completions
    overrides: ChatCompletionOverride
    # Underlying engine
    engine: Union[JSONFFIEngine, MLCEngine]
    last_finished_request_usage: Optional[openai_api_protocol.CompletionUsage]

    def __init__(self, engine: Union[JSONFFIEngine, MLCEngine]):
        self.engine = engine
        self.history = []
        self.history_window_begin = 0
        self.overrides = ChatCompletionOverride()
        # model is mainly used for compact reasons
        self.model = "chat_model"
        self.last_finished_request_usage = None

    def slide_history(self):
        """Slide history to fit into context window"""
        history_window_size = len(self.history) - self.history_window_begin
        assert history_window_size % 2 == 0
        self.history_window_begin += ((history_window_size + 3) // 4) * 2

    def process_system_prompts(self):
        """Process system prompts"""
        # TODO(mlc-team): possibly leverage debug option
        # pass a simple prompt to warm up
        for _ in self.engine.chat.completions.create(
            messages=[{"role": "user", "content": ""}], max_tokens=1, model=self.model, stream=True
        ):
            pass

    def generate(self, prompt: str):
        """Run one generation with the prompt.

        Parameters
        ----------
        prompt: str
            The input prompt
        """
        self.history.append({"role": "user", "content": prompt})
        output_text = ""
        finish_reason_length = False
        messages = self.history[self.history_window_begin :]

        for response in self.engine.chat.completions.create(
            messages=messages,
            model=self.model,
            stream=True,
            stream_options={"include_usage": True},
            **dataclasses.asdict(self.overrides),
        ):
            if response.usage is not None:
                self.last_finished_request_usage = response.usage
                continue
            for choice in response.choices:
                assert choice.delta.role == "assistant"
                if isinstance(choice.delta.content, str):
                    output_text += choice.delta.content
                    print(choice.delta.content, end="", flush=True)
                if choice.finish_reason == "length":
                    finish_reason_length = True
        if finish_reason_length:
            print(" [output truncated due to context length limit...]")
        # print additional \n when generation ends
        print()
        # record the history
        self.history.append({"role": "assistant", "content": output_text})
        if finish_reason_length:
            self.slide_history()

    def stats(self):
        """Print statistics of the prefill and decode speed."""

        def get_stats_text():
            """Get text"""
            if self.last_finished_request_usage is None:
                return "N/A"
            last_finished_request = self.last_finished_request_usage.extra
            if last_finished_request is None:
                return "N/A"
            prefill_speed = last_finished_request.get("prefill_tokens_per_s", None)
            decode_speed = last_finished_request.get("decode_tokens_per_s", None)
            prefill_speed = f"{prefill_speed:.1f}" if prefill_speed is not None else "N/A"
            decode_speed = f"{decode_speed:.1f}" if decode_speed is not None else "N/A"
            return f"prefill: {prefill_speed} tok/s, decode: {decode_speed} tok/s"

        print(get_stats_text(), flush=True)

    def metrics(self):
        """Print metrics as prometheus text"""
        print(_query_engine_metrics(self.engine).prometheus_text(), flush=True)

    def reset(self):
        """Reset the chat history"""
        self.history = []
        self.history_window_begin = 0

    def chat(self):
        """Start an interactive chat session."""
        _print_help_str()

        self.process_system_prompts()  # pylint: disable=protected-access
        # Multi-line input support: set escape+enter as start a new line
        kb = _set_up_key_bindings()

        while True:
            prompt = get_prompt(
                ">>> ",  # pylint: disable=protected-access
                key_bindings=kb,
                multiline=True,
            )
            if prompt[:4] == "/set":
                overrides = ChatCompletionOverride.from_str(prompt.split()[1])
                for key, value in dataclasses.asdict(overrides).items():
                    if value is not None:
                        setattr(self.overrides, key, value)
            elif prompt[:6] == "/stats":
                self.stats()
            elif prompt[:8] == "/metrics":
                self.metrics()
            elif prompt[:6] == "/reset":
                self.reset()
            elif prompt[:5] == "/exit":
                break
            elif prompt[:5] == "/help":
                _print_help_str()
            else:
                self.generate(prompt)


def chat(
    model: str,
    device: str,
    model_lib: Optional[str],
    overrides: ModelConfigOverride,
):
    """Chat cli entry"""
    # By default we use JSONFFIEngine
    ChatState(
        JSONFFIEngine(
            model,
            device,
            model_lib=model_lib,
            mode="interactive",
            engine_config=EngineConfig(
                max_single_sequence_length=overrides.context_window_size,
                prefill_chunk_size=overrides.prefill_chunk_size,
                sliding_window_size=overrides.sliding_window_size,
                attention_sink_size=overrides.attention_sink_size,
                tensor_parallel_shards=overrides.tensor_parallel_shards,
                pipeline_parallel_stages=overrides.pipeline_parallel_stages,
                opt=overrides.opt,
            ),
        )
    ).chat()
