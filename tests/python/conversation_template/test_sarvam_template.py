from mlc_llm.conversation_template import ConvTemplateRegistry


def test_sarvam_prompt():
    conversation = ConvTemplateRegistry.get_conv_template("sarvam")
    assert conversation is not None
    conversation.messages.append(("user", "What is the capital of India?"))
    conversation.messages.append(("assistant", None))

    prompt = conversation.as_prompt()[0]

    assert prompt == (
        "[@BOS@]\n"
        "<|start_of_turn|><|user|>\n"
        "What is the capital of India?<|end_of_turn|>\n"
        "<|start_of_turn|><|assistant|>"
    )
    assert conversation.stop_str == ["<|end_of_turn|>"]
