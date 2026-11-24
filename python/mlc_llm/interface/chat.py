"""Python entrypoint of chat."""

import dataclasses
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union  # noqa: UP035

from prompt_toolkit import prompt as get_prompt
from prompt_toolkit.key_binding import KeyBindings

from mlc_llm.json_ffi import JSONFFIEngine
from mlc_llm.protocol import openai_api_protocol
from mlc_llm.serve.config import EngineConfig
from mlc_llm.serve.engine import MLCEngine
from mlc_llm.serve.engine_base import _query_engine_metrics
from mlc_llm.support import argparse
from mlc_llm.support.config import ConfigOverrideBase


def _print_help_str():
    help_str = """You can use the following special commands:
  /help                    print the special commands
  /exit                    quit the cli
  /stats                   print out stats of last request (tok/s and exact token counts)
  /metrics                 print out full engine metrics
  /reset                   restart a fresh chat
  /random-tokens <N>       generate N random prompt tokens and run the model
  /set [overrides]         override settings in the generation config. For example,
                           `/set temperature=0.5;top_p=0.8;seed=23;max_tokens=100;stop=str1,str2`
                           Note: Separate stop words in the `stop` option with commas (,).
  Multi-line input: Use escape+enter to start a new line.
"""
    print(help_str)


def _get_vocab_size(engine: Any) -> int:
    """Return tokenizer vocab size from the model directory, or a common fallback."""
    engine_config = getattr(engine, "engine_config", None)
    model_path = getattr(engine_config, "model", None)
    if model_path is None:
        return 32000

    tokenizer_json_path = Path(model_path) / "tokenizer.json"
    if not tokenizer_json_path.exists():
        return 32000

    try:
        with tokenizer_json_path.open("r", encoding="utf-8") as tokenizer_file:
            tokenizer_data = json.load(tokenizer_file)
    except (OSError, json.JSONDecodeError):
        return 32000

    vocab = tokenizer_data.get("model", {}).get("vocab", {})
    return len(vocab) if vocab else 32000


def _parse_random_tokens_command(prompt: str) -> Tuple[int, Optional[int]]:  # noqa: UP006
    """Parse `/random-tokens <N> [max_decode_tokens]`."""
    parts = prompt.split()
    if len(parts) < 2 or len(parts) > 3:
        raise ValueError(
            "Usage: /random-tokens <N> [max_decode_tokens]\n"
            "  N                 - target prompt token count\n"
            "  max_decode_tokens - optional decode token limit"
        )

    try:
        token_size = int(parts[1])
        max_decode_tokens = int(parts[2]) if len(parts) == 3 else None
    except ValueError as err:
        raise ValueError("Both N and max_decode_tokens must be integers.") from err

    if token_size <= 0:
        raise ValueError("Token size must be a positive integer.")
    if max_decode_tokens is not None and max_decode_tokens <= 0:
        raise ValueError("Max decode tokens must be a positive integer.")
    return token_size, max_decode_tokens


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
class ChatCompletionOverride(ConfigOverrideBase):
    """Flags for overriding chat completions."""

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None  # noqa: UP006

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
class ModelConfigOverride(ConfigOverrideBase):
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

    history: List[Dict[str, Any]]  # noqa: UP006
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
            messages=[{"role": "user", "content": ""}],
            max_tokens=1,
            model=self.model,
            stream=True,
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

    def _get_template_overhead(self) -> int:
        """Measure and cache the number of tokens added by the chat template."""
        if hasattr(self, "_template_overhead_cache"):
            return self._template_overhead_cache

        if not hasattr(self.engine, "tokenizer"):
            self._template_overhead_cache = 0
            return 0

        calibration_text = "a"
        content_token_count = len(self.engine.tokenizer.encode(calibration_text))
        saved_usage = self.last_finished_request_usage

        overhead = 0
        for response in self.engine.chat.completions.create(
            messages=[{"role": "user", "content": calibration_text}],
            max_tokens=1,
            model=self.model,
            stream=True,
            stream_options={"include_usage": True},
        ):
            if response.usage is not None:
                overhead = max(0, response.usage.prompt_tokens - content_token_count)
                break

        self._template_overhead_cache = overhead
        self.last_finished_request_usage = saved_usage
        return overhead

    def generate_random_tokens(self, token_size: int, max_decode_tokens: Optional[int] = None):
        """Generate a random prompt targeting exactly ``token_size`` prompt tokens."""
        vocab_size = max(2, _get_vocab_size(self.engine))
        template_overhead = self._get_template_overhead()
        content_needed = max(1, token_size - template_overhead)

        if hasattr(self.engine, "tokenizer"):
            token_ids = [random.randint(1, vocab_size - 1) for _ in range(content_needed)]
            prompt = self.engine.tokenizer.decode(token_ids)

            for _ in range(3):
                actual_content = len(self.engine.tokenizer.encode(prompt))
                if actual_content == content_needed:
                    break
                current_token_ids = self.engine.tokenizer.encode(prompt)
                if actual_content < content_needed:
                    current_token_ids.extend(
                        random.randint(1, vocab_size - 1)
                        for _ in range(content_needed - actual_content)
                    )
                else:
                    current_token_ids = current_token_ids[:content_needed]
                prompt = self.engine.tokenizer.decode(current_token_ids)

            actual_content = len(self.engine.tokenizer.encode(prompt))
        else:
            prompt = " ".join(
                str(token_id)
                for token_id in [random.randint(1, vocab_size - 1) for _ in range(content_needed)]
            )
            actual_content = content_needed

        expected_total = actual_content + template_overhead
        print(
            f"[random-tokens] target={token_size}, "
            f"template_overhead={template_overhead}, "
            f"content_tokens={actual_content}, "
            f"expected_total~{expected_total}"
        )

        saved_max_tokens = self.overrides.max_tokens
        if max_decode_tokens is not None:
            self.overrides.max_tokens = max_decode_tokens
        try:
            self.generate(prompt)
        finally:
            self.overrides.max_tokens = saved_max_tokens

    def stats(self):
        """Print statistics of the prefill/decode speed and exact token counts."""

        def get_stats_text():
            """Get text"""
            if self.last_finished_request_usage is None:
                return "N/A"

            usage = self.last_finished_request_usage
            last_finished_request = usage.extra
            if last_finished_request is None:
                prefill_speed = "N/A"
                decode_speed = "N/A"
            else:
                prefill_speed = last_finished_request.get("prefill_tokens_per_s", None)
                decode_speed = last_finished_request.get("decode_tokens_per_s", None)
                prefill_speed = f"{prefill_speed:.1f}" if prefill_speed is not None else "N/A"
                decode_speed = f"{decode_speed:.1f}" if decode_speed is not None else "N/A"
            return (
                f"prefill: {prefill_speed} tok/s, decode: {decode_speed} tok/s\n"
                f"prompt tokens: {usage.prompt_tokens}, "
                f"completion tokens: {usage.completion_tokens}, "
                f"total tokens: {usage.total_tokens}"
            )

        print(get_stats_text(), flush=True)

    def metrics(self):
        """Print metrics as prometheus text"""
        print(_query_engine_metrics(self.engine).prometheus_text(), flush=True)

    def reset(self):
        """Reset the chat history"""
        self.history = []
        self.history_window_begin = 0

    def chat(
        self,
        random_tokens: Optional[int] = None,
        max_decode_tokens: Optional[int] = None,
    ):
        """Start an interactive chat session."""
        _print_help_str()

        self.process_system_prompts()
        if random_tokens is not None and random_tokens > 0:
            print(f"[random-tokens] Auto-running with {random_tokens} random tokens...")
            self.generate_random_tokens(random_tokens, max_decode_tokens=max_decode_tokens)
            self.stats()

        # Multi-line input support: set escape+enter as start a new line
        kb = _set_up_key_bindings()

        while True:
            try:
                prompt = get_prompt(
                    ">>> ",
                    key_bindings=kb,
                    multiline=True,
                )
            except (KeyboardInterrupt, EOFError):
                break
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
            elif prompt[:14] == "/random-tokens":
                try:
                    token_size, max_decode_tokens = _parse_random_tokens_command(prompt)
                except ValueError as err:
                    print(err)
                else:
                    self.generate_random_tokens(
                        token_size,
                        max_decode_tokens=max_decode_tokens,
                    )
            else:
                self.generate(prompt)


def chat(
    model: str,
    device: str,
    model_lib: Optional[str],
    overrides: ModelConfigOverride,
    random_tokens: Optional[int] = None,
    max_decode_tokens: Optional[int] = None,
):
    """Chat cli entry"""
    # By default we use JSONFFIEngine
    engine = JSONFFIEngine(
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
    try:
        ChatState(engine).chat(random_tokens=random_tokens, max_decode_tokens=max_decode_tokens)
    finally:
        engine.terminate()
