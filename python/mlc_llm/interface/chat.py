"""Python entrypoint of chat."""

import dataclasses
from typing import Dict, List, Optional, Union

from prompt_toolkit import prompt as get_prompt  # pylint: disable=import-error
from prompt_toolkit.key_binding import KeyBindings  # pylint: disable=import-error

from mlc_llm.json_ffi import JSONFFIEngine
from mlc_llm.support import argparse
from mlc_llm.support.config import ConfigOverrideBase


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


class ChatState:
    """Helper class to manage chat state"""

    history: List[Dict]
    history_begin: int
    # kwargs passed to completions
    overrides: ChatCompletionOverride
    # we use JSON ffi engine to ensure broader coverage
    engine: JSONFFIEngine

    def __init__(self, engine):
        self.engine = engine
        self.history = []
        self.history_window_begin = 0
        self.overrides = ChatCompletionOverride()

    def process_system_prompts(self):
        """Process system prompts"""
        # TODO(mlc-team): possibly leverage debug option
        # pass a simple prompt to warm up
        for _ in self.engine.chat.completions.create(
            messages=[{"role": "user", "content": ""}], max_tokens=1, stream=True
        ):
            pass

    def slide_history(self):
        """Slide history to fit into context window"""
        history_window_size = len(self.history) - self.history_window_begin
        assert history_window_size % 2 == 0
        self.history_window_begin += ((history_window_size + 3) // 4) * 2

    def generate(self, prompt: str):
        """Run one generatiohn with the prompt"""
        self.history.append({"role": "user", "content": prompt})
        output_text = ""
        finish_reason_length = False
        messages = self.history[self.history_window_begin :]
        for response in self.engine.chat.completions.create(
            messages=messages,
            stream=True,
            **dataclasses.asdict(self.overrides),
        ):
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

    def stats(self) -> str:
        """Return the statistics of the prefill and decode speed."""
        metrics = self.engine.metrics()
        last_finished_request = metrics["last_finished_request"]
        prefill_speed = last_finished_request.get("prefill_tokens_per_s", None)
        decode_speed = last_finished_request.get("decode_tokens_per_s", None)
        prefill_speed = f"{prefill_speed:.1f}" if prefill_speed is not None else "N/A"
        decode_speed = f"{decode_speed:.1f}" if decode_speed is not None else "N/A"
        return f"prefill: {prefill_speed} tok/s, decode: {decode_speed} tok/s"

    def metrics(self) -> str:
        """Return metrics as prometheus text"""
        return self.engine.metrics().prometheus_text()

    def reset_chat(self):
        """Reset the chat history"""
        self.history = []
        self.history_window_begin = 0


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


def chat(
    model: str,
    device: str,
    model_lib: Optional[str],
):
    """chat with a model."""

    # Set up ChatModule
    engine = JSONFFIEngine(model, device, model_lib=model_lib, mode="interactive")
    _print_help_str()

    chat_state = ChatState(engine)
    chat_state.process_system_prompts()  # pylint: disable=protected-access

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
                    setattr(chat_state.overrides, key, value)
        elif prompt[:6] == "/stats":
            print(chat_state.stats(), flush=True)
        elif prompt[:8] == "/metrics":
            print(chat_state.metrics(), flush=True)
        elif prompt[:6] == "/reset":
            chat_state.reset_chat()
        elif prompt[:5] == "/exit":
            break
        elif prompt[:5] == "/help":
            _print_help_str()
        else:
            chat_state.generate(prompt)
