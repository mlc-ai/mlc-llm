"""Python entrypoint of chat."""
import dataclasses
from typing import List, Optional, Union

from prompt_toolkit import prompt as get_prompt
from prompt_toolkit.key_binding import KeyBindings

from mlc_chat.callback import StreamToStdout
from mlc_chat.chat_module import ChatConfig, ChatModule, GenerationConfig
from mlc_chat.support import argparse
from mlc_chat.support.config import ConfigOverrideBase

from .compiler_flags import ModelConfigOverride


@dataclasses.dataclass
class GenerationConfigOverride(ConfigOverrideBase):  # pylint: disable=too-many-instance-attributes
    """Flags for overriding generation config."""

    temperature: Optional[float] = None
    repetition_penalty: Optional[float] = None
    top_p: Optional[float] = None
    mean_gen_len: Optional[int] = None
    max_gen_len: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    n: Optional[int] = None  # pylint: disable=invalid-name
    stop: Optional[Union[str, List[str]]] = None

    @staticmethod
    def from_str(source: str) -> "ModelConfigOverride":
        """Parse model config override values from a string."""
        parser = argparse.ArgumentParser(description="model config override values")
        parser.add_argument("--temperature", type=float, default=None)
        parser.add_argument("--repetition_penalty", type=float, default=None)
        parser.add_argument("--top_p", type=float, default=None)
        parser.add_argument("--mean_gen_len", type=int, default=None)
        parser.add_argument("--max_gen_len", type=int, default=None)
        parser.add_argument("--presence_penalty", type=float, default=None)
        parser.add_argument("--frequency_penalty", type=float, default=None)
        parser.add_argument("--n", type=int, default=None)
        parser.add_argument("--stop", type=str, default=None)
        results = parser.parse_args([f"--{i}" for i in source.split(";") if i])
        return GenerationConfigOverride(
            temperature=results.temperature,
            repetition_penalty=results.repetition_penalty,
            top_p=results.top_p,
            mean_gen_len=results.mean_gen_len,
            max_gen_len=results.max_gen_len,
            presence_penalty=results.presence_penalty,
            frequency_penalty=results.frequency_penalty,
            n=results.n,
            stop=results.stop.split(",") if results.stop is not None else None,
        )


def _print_help_str():
    help_str = """You can use the following special commands:
  /help               print the special commands
  /exit               quit the cli
  /stats              print out the latest stats (token/sec)
  /reset              restart a fresh chat
  /set [overrides]    override settings in the generation config. For example,
                      `/set temperature=0.5;max_gen_len=100;stop=end,stop`
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
    opt: str,
    overrides: ModelConfigOverride,
    model_lib_path: Optional[str],
):
    """chat with a model."""
    # Set up chat config and generate config
    config = ChatConfig(opt=opt)
    generate_config = GenerationConfig()
    # Apply overrides
    config = overrides.apply(config)
    # Set up ChatModule
    cm = ChatModule(model, device, chat_config=config, model_lib_path=model_lib_path)
    _print_help_str()
    cm._process_system_prompts()  # pylint: disable=protected-access

    # Multi-line input support: set escape+enter as start a new line
    kb = _set_up_key_bindings()

    while True:
        prompt = get_prompt(
            f"{cm._get_role_0()}: ",  # pylint: disable=protected-access
            key_bindings=kb,
            multiline=True,
        )
        if prompt[:6] == "/reset":
            cm.reset_chat()
        elif prompt[:5] == "/exit":
            break
        elif prompt[:6] == "/stats":
            print(cm.stats(), flush=True)
        elif prompt[:4] == "/set":
            gen_config_overrides = GenerationConfigOverride.from_str(prompt.split()[1])
            generate_config = gen_config_overrides.apply(generate_config)
        elif prompt[:5] == "/help":
            _print_help_str()
        else:
            print(f"{cm._get_role_1()}: ")  # pylint: disable=protected-access
            cm.generate(
                prompt,
                progress_callback=StreamToStdout(callback_interval=2),
                generation_config=generate_config,
            )
