# pylint: disable=missing-function-docstring
from mlc_serve.engine.streamer import TextStreamer
from transformers import AutoTokenizer
import os
import pytest


def _get_tokenizer_path() -> str:
    path = os.environ.get("MLC_LLAMA_TOKENIZER_PATH")
    if path is None:
        raise ValueError(
            'Environment variable "MLC_LLAMA_TOKENIZER_PATH" not found. '
            "Please set it to the a valid llama tokenizer path."
        )
    return path


@pytest.fixture
def llama_tokenizer_path() -> str:
    return _get_tokenizer_path()


def test_streamer(tokenizer_path, input_tokens, answer):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    text_streamer = TextStreamer(tokenizer)
    total_text = ""
    for token in input_tokens:
        delta = text_streamer.put([token])
        total_text += delta
    total_text += text_streamer.finish()
    assert total_text == answer, f"decoded text: {total_text}, expected: {answer}"


# fmt: off
tests = [
    [
        # input tokens
        [
            18585, 29892, 1244, 29915,
            29879, 263,   3273, 14880,
            1048,  953,  29877,  2397,
        ],
        # answer
        "Sure, here's a short paragraph about emoji"
    ],
    [
        # input tokens
        [
            18585, 29892, 1244, 29915, 29879, 263, 3273, 14880, 1048, 953, 29877, 2397,
            29892, 988, 1269, 1734, 338, 5643, 491, 385, 953, 29877, 2397, 29901, 13, 13,
            29950, 1032, 727, 29991, 29871, 243, 162, 148, 142, 306, 29915, 29885, 1244, 304,
            1371, 1234, 738, 5155, 366, 505, 1048, 953, 29877, 2397, 29871, 243, 162, 167, 151,
            29889, 7440, 366, 1073, 393, 953, 29877, 2397, 508, 367, 1304, 304, 27769, 23023,
            1080, 322, 21737, 297, 263, 2090, 322, 1708, 1319, 982, 29973, 29871, 243, 162, 155,
            135, 2688, 508, 884, 367, 1304, 304, 788, 263, 6023, 310, 2022, 2877, 304, 596, 7191,
            322, 11803, 29889, 29871, 243, 162, 149, 152, 1126, 29892, 1258, 366, 1073, 393, 727,
            526, 1584, 953, 29877, 2397, 8090, 322, 14188, 366, 508, 1708, 29973, 29871, 243, 162,
            145, 177, 243, 162, 148, 131, 1105, 29892, 748, 14432, 322, 679, 907, 1230, 411, 953,
            29877, 2397, 29991, 29871, 243, 162, 149, 168, 243, 162, 145, 171
        ],
        # answer
        (
            "Sure, here's a short paragraph about emoji, "
            "where each word is followed by an emoji:\n\n"
            "Hey there! 👋 I'm here to help answer any questions you have about emoji 🤔. "
            "Did you know that emoji can be used to convey emotions and feelings in a "
            "fun and playful way? 😄 "
            "They can also be used to add a touch of personality to your messages and posts. 💕 "
            "And, did you know that there are even emoji games and activities you can play? 🎮👀 "
            "So, go ahead and get creative with emoji! 💥🎨"
        )   
    ],
    [
        # input tokens
        [243, 162, 148, 131, 243, 162, 148, 131], 
        # answer
        "👀👀"
    ],
    [
        # input tokens
        [243, 162, 148, 131, 243, 162, 148, 131, 243, 162, 148, 131], 
        # answer
        "👀👀👀"
    ],
    [
        # input tokens
        [243, 162, 148, 131, 162, 148, 131, 505, 243, 162, 148, 131], 
        # answer
        "👀��� have👀"
    ],
    [
        # input tokens
        [243, 162, 148, 131, 162, 148, 131, 243, 162, 148, 131], 
        # answer
        "👀�������"
    ],
    [
        # input tokens
        [177, 243, 162, 148, 131], 
        # answer
        "�����"
    ]
]
# fmt: on

if __name__ == "__main__":
    tokenizer_path = _get_tokenizer_path()
    for input_tokens, answer in tests:
        test_streamer(tokenizer_path, input_tokens, answer)
