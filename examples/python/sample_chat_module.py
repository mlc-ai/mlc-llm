from mlc_chat.chat_module  import ChatModule

# From the mlc-llm directory, run
# $ python -m examples.python.sample_chat_module 

# Create a ChatModule instance
cm = ChatModule(model='Llama-2-7b-chat-hf-q4f16_1')

# Generate a response for a given prompt
cm.generate(prompt="What is the meaning of life?")

# Print some runtime statistics for the generation
runtime_stats = cm.runtime_stats_text()
print(f"{runtime_stats=}")

# Reset the chat module
cm.reset_chat()
