from mlc_chat import ChatModule
from mlc_chat.chat_module import ChatConfig

cm = ChatModule(
    model="dist/RedPajama-INCITE-Chat-3B-v1-q0f16/params",
    chat_config=ChatConfig(
        temperature=0.01,
        top_p=0.9,
        max_gen_len=4096,
    ),
)

output = cm.generate("What is the meaning of life?")
print(output)
