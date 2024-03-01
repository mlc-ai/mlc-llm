from mlc_chat import ChatConfig, ChatModule
from mlc_chat.callback import StreamToStdout

# From the mlc-llm directory, run
# $ python examples/python/sample_mlc_chat.py

# Create a ChatModule instance
# Multiple GPU
cm = ChatModule(
    # model="dist/llama-2-7b-chat-hf-q4f16_1/params",
    model="dist/llama-2-7b-chat-hf-q4f16_1-presharded-2gpu/params",
    chat_config=ChatConfig(tensor_parallel_shards=2)
    # model="dist/Mixtral-8x7B-Instruct-v0.1-q0f16/params",
    # chat_config=ChatConfig(tensor_parallel_shards=2),
)

# Generate a response for a given prompt
output = cm.generate(
    prompt="What is the meaning of life?",
    progress_callback=StreamToStdout(callback_interval=2),
)

# Print prefill and decode performance statistics
print(f"Statistics: {cm.stats()}\n")

# Reset the chat module by
cm.reset_chat()
