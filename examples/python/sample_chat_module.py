from mlc_chat.chat_module import ChatModule

# From the mlc-llm directory, run
# $ python -m examples.python.sample_chat_module
# or directly run
# $ python examples/python/sample_chat_module.py

# Create a ChatModule instance
cm = ChatModule(model="Llama-2-7b-chat-hf-q4f16_1")

# Generate a response for a given prompt
output = cm.generate(prompt="What is the meaning of life?")
print(f"Generated text:\n{output}\n")

# Print some runtime statistics for the generation
print(f"Statistics: {cm.stats()}")

# Reset the chat module
cm.reset_chat()
