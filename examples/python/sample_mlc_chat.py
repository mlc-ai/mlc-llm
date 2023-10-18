from mlc_chat import ChatModule
from mlc_chat.callback import StreamToStdout

# From the mlc-llm directory, run
# $ python examples/python/sample_mlc_chat.py

# Create a ChatModule instance
cm = ChatModule(model="Llama-2-7b-chat-hf-q4f16_1")
# You can change to other models that you downloaded, for example,
# cm = ChatModule(model="Llama-2-13b-chat-hf-q4f16_1")  # Llama2 13b model

# Generate a response for a given prompt
output = cm.generate(
    prompt="What is the meaning of life?",
    progress_callback=StreamToStdout(callback_interval=2),
)

# Print prefill and decode performance statistics
print(f"Statistics: {cm.stats()}\n")

output = cm.generate(
    prompt="How many points did you list out?",
    progress_callback=StreamToStdout(callback_interval=2),
)

# Reset the chat module by
# cm.reset_chat()
