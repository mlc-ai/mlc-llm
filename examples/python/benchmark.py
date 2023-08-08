from mlc_chat import ChatModule

# From the mlc-llm directory, run
# $ python examples/python/benchmark.py

# Create a ChatModule instance
cm = ChatModule(model="Llama-2-7b-chat-hf-q4f16_1")

output = cm.benchmark_generate("What's the meaning of life?", generate_length=256)
print(f"Generated text:\n{output}\n")
print(f"Statistics: {cm.stats()}")
