from python.mlc_chat.chat_module  import ChatModule

# From the mlc-llm directory, run
# $ python -m examples.python.sample_chat_module 

# Create a ChatModule instance
cm = ChatModule(model='Llama-2-7b-chat-hf-q4f16_1', device_name="vulkan")

# Generate a response for a given prompt
# TODO: Update once generate API is complete
cm.generate()

# Print some runtime statistics for the generation
runtime_stats = cm.get_runtime_stats()
print(f"{runtime_stats=}")

# Reset the chat module
cm.reset_chat()