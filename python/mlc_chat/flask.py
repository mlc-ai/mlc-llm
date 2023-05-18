from chat_module import LLMChatModule

if __name__ == "__main__":
    cm = LLMChatModule("/home/sudeepag/mlc-llm/dist/vicuna-params-q3f16_0", target="vulkan")
    cm.init_chat()
    prompt = "Write a poem"
    cm.encode(prompt)
    msg = None
    while not cm.stopped():
        cm.decode()
        msg = cm.get_message()
        # print(cm.get_message())
    print(msg)