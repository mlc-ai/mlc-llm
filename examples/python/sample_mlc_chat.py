from mlc_chat import ChatModule
from mlc_chat.callback import StreamToStdout

# From the mlc-llm directory, run
# $ python examples/python/sample_mlc_chat.py

# Create a ChatModule instance
cm = ChatModule(
    model="dist/Llama-2-7b-chat-hf-q4f16_1-MLC",
    model_lib_path="dist/prebuilt_libs/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f16_1-cuda.so"
    # Vulkan on Linux: Llama-2-7b-chat-hf-q4f16_1-vulkan.so
    # Metal on macOS: Llama-2-7b-chat-hf-q4f16_1-metal.so
    # Other platforms: Llama-2-7b-chat-hf-q4f16_1-{backend}.{suffix}
)

# You can change to other models that you downloaded
# Model variants of the same architecture can reuse the same model library
# Here WizardMath reuses Mistral's model library
# cm = ChatModule(
#     model="dist/Mistral-7B-Instruct-v0.2-q4f16_1-MLC",  # or "dist/WizardMath-7B-V1.1-q4f16_1-MLC"
#     model_lib_path="dist/prebuilt_libs/Mistral-7B-Instruct-v0.2/Mistral-7B-Instruct-v0.2-q4f16_1-cuda.so"
# )

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
