"""Python entrypoint of chat."""

from typing import List, Optional

from prompt_toolkit import prompt as get_prompt  # pylint: disable=import-error
from prompt_toolkit.key_binding import KeyBindings  # pylint: disable=import-error

from mlc_llm.json_ffi import JSONFFIEngine


class ChatState:
    """Helper class to manage chat state"""

    history: List[dict]
    history_begin: int
    # we use JSON ffi engine to ensure broader coverage
    engine: JSONFFIEngine

    def __init__(self, engine):
        self.engine = engine
        self.history = []
        self.history_window_begin = 0

    def process_system_prompts(self):
        """Process system prompts"""
        # TODO(mlc-team): possibly leverage debug option
        # pass a simple prompt to warm up
        for _ in self.engine.chat.completions.create(
            messages=[{"role": "user", "content": "hello"}], max_tokens=1, stream=True
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
        for response in self.engine.chat.completions.create(messages=messages, stream=True):
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

    def reset_chat(self):
        """Reset the chat history"""
        self.history = []
        self.history_window_begin = 0


# TODO(mlc-team): add back support for stats
def _print_help_str():
    help_str = """You can use the following special commands:
  /help               print the special commands
  /exit               quit the cli
  /reset              restart a fresh chat
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
        if prompt[:6] == "/reset":
            chat_state.reset_chat()
        elif prompt[:5] == "/exit":
            break
        # elif prompt[:6] == "/stats":
        #     print(cm.stats(), flush=True)
        elif prompt[:5] == "/help":
            _print_help_str()
        else:
            chat_state.generate(prompt)
