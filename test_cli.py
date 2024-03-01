help_str = """You can use the following special commands:
  /help               print the special commands
  /exit               quit the cli
  /stats              print out the latest stats (token/sec)
  /reset              restart a fresh chat
  /reload [model]  reload model `model` from disk, or reload the current \
model if `model` is not specified
"""
print(help_str)

from prompt_toolkit import prompt
from prompt_toolkit.key_binding import KeyBindings

kb = KeyBindings()


@kb.add("escape", "enter")
def _(event):
    event.current_buffer.insert_text("\n")


@kb.add("enter")
def _(event):
    event.current_buffer.validate_and_handle()


result = prompt(">", key_bindings=kb, multiline=True)
print("12")
print(result)

# user_input = get_multiline_input()
# print(user_input)
