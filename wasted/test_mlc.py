import tvm

from mlc_chat import ChatModule
from mlc_chat.chat_module import ChatConfig


@tvm.register_func("effect.print")
def _print(_, array) -> None:
    print(f"effect.print: shape = {array.shape}, dtype = {array.dtype}, data =\n{array}")


import numpy as np

array = tvm.nd.array(np.random.random((4, 20)))

print(f"effect.print: shape = {array.shape}, dtype = {array.dtype}, data =\n{array}")

# cm = ChatModule(
#     # model="dist/llama-2-7b-chat-hf-q4f16_1/params",
#     model="dist/new-llama-q4f16_1/params",
#     chat_config=ChatConfig(
#         temperature=0.01,
#         top_p=0.9,
#         max_gen_len=4096,
#     ),
#     model_lib_path="dist/new-llama-q4f16_1/llama.so",
# )

# output = cm.generate("What is the meaning of life?")
# print(output)
