from mlc_chat import ChatModule
from mlc_chat.chat_module import ChatConfig

cm = ChatModule(
    model="./dist/phi-2-q0f16/params",
    model_lib_path="./dist/phi-2-q0f16/model.so",
    # chat_config=ChatConfig(
    #     temperature=0.01,
    #     top_p=0.9,
    #     max_gen_len=4096,
    # ),
)

output = cm.generate("What is the meaning of life?")
print(output)
print(output)
