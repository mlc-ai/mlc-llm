from mlc_llm import Engine

# Create engine
model = "HF://mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC"
engine = Engine(model)

# Run chat completion in OpenAI API.
for response in engine.chat.completions.create(
    messages=[{"role": "user", "content": "What is the meaning of life?"}],
    model=model,
    stream=True,
):
    for choice in response.choices:
        print(choice.delta.content, end="", flush=True)
print("\n")

engine.terminate()
