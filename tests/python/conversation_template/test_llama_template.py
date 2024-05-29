import pytest

from mlc_llm.conversation_template import ConvTemplateRegistry

pytestmark = [pytest.mark.runtime_unittest]


# From the official Llama-3 example:
# https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
def test_llama3_prompt():
    conversation = ConvTemplateRegistry.get_conv_template("llama-3")
    system_msg = "You are a helpful AI assistant for travel tips and recommendations"
    user_msg1 = "What is France's capital?"
    assistant_msg1 = "Bonjour! The capital of France is Paris!"
    user_msg2 = "What can I do there?"
    assistant_msg2 = "Paris, the City of Light, offers a romantic getaway with must-see attractions like the Eiffel Tower and Louvre Museum, romantic experiences like river cruises and charming neighborhoods, and delicious food and drink options, with helpful tips for making the most of your trip."
    prompt = "Give me a detailed list of the attractions I should visit, and time it takes in each one, to plan my trip accordingly."

    conversation.system_message = system_msg
    conversation.messages.append(("user", user_msg1))
    conversation.messages.append(("assistant", assistant_msg1))
    conversation.messages.append(("user", user_msg2))
    conversation.messages.append(("assistant", assistant_msg2))
    conversation.messages.append(("user", prompt))
    conversation.messages.append(("assistant", None))
    res = conversation.as_prompt()

    expected = (
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful AI assistant for travel tips and recommendations<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "What is France's capital?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        "Bonjour! The capital of France is Paris!<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "What can I do there?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        "Paris, the City of Light, offers a romantic getaway with must-see attractions like the Eiffel Tower and Louvre Museum, romantic experiences like river cruises and charming neighborhoods, and delicious food and drink options, with helpful tips for making the most of your trip.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "Give me a detailed list of the attractions I should visit, and time it takes in each one, to plan my trip accordingly.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    assert res[0] == expected


if __name__ == "__main__":
    test_llama3_prompt()
