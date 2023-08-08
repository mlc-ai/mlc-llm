from mlc_chat.chat_module import ChatModule

# From the mlc-llm directory, run
# $ python -m examples.python.benchmark
# or directly run
# $ python examples/python/benchmark.py

# Create a ChatModule instance
cm = ChatModule(model="Llama-2-7b-chat-hf-q4f16_1")
cm.evaluate(prompt_len=1, generate_len=512)
