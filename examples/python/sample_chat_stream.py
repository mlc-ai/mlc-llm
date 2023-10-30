from mlc_chat import ChatModule
from mlc_chat.callback import StreamToStdout, StreamIterator

# From the mlc-llm directory, run
# $ python examples/python/sample_chat_stream.py

# Create a ChatModule instance
cm = ChatModule(model="Llama-2-7b-chat-hf-q4f16_1")

# Stream to Stdout
output = cm.generate(
    prompt="What is the meaning of life?",
    progress_callback=StreamToStdout(callback_interval=2),
)

# Stream to an Iterator
from threading import Thread

stream = StreamIterator(callback_interval=2)
generation_thread = Thread(
    target=cm.generate,
    kwargs={"prompt": "What is the meaning of life?", "progress_callback": stream},
)
generation_thread.start()

output = ""
for delta_message in stream:
    output += delta_message

generation_thread.join()
