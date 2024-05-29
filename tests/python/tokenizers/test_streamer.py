"""Streamer tests in MLC LLM.

Please specify the local path to llama2 tokenizer via environment
variable before running this test.
The recommended way to run the tests is to use the following command:
  MLC_LLAMA_TOKENIZER_PATH="path/to/llama/tokenizer" \
  pytest -vv tests/python/support/test_text_streamer_stop_handler.py

Here "MLC_LLAMA_TOKENIZER_PATH" can be chosen from
- a llama2 weight directory (e.g., "path/to/Llama-2-7b-chat-hf"),
- a sentencepiece llama2 tokenizer path
  (e.g., "path/to/Llama-2-7b-chat-hf/tokenizer.model").

To directly run the Python file (a.k.a., not using pytest), you also need to
specify the tokenizer path via environment variable.
"""

# pylint: disable=missing-function-docstring
import time
from typing import List, Tuple

import pytest

from mlc_llm.testing import require_test_tokenizers
from mlc_llm.tokenizers import StopStrHandler, TextStreamer, Tokenizer

# test category "unittest"
pytestmark = [pytest.mark.unittest]


# fmt: off
para_input_tokens = [18585, 29892, 1244, 29915, 29879, 263, 3273, 14880, 1048, 953, 29877, 2397,
          29892, 988, 1269, 1734, 338, 5643, 491, 385, 953, 29877, 2397, 29901, 13, 13,
          29950, 1032, 727, 29991, 29871, 243, 162, 148, 142, 306, 29915, 29885, 1244, 304,
          1371, 1234, 738, 5155, 366, 505, 1048, 953, 29877, 2397, 29871, 243, 162, 167, 151,
          29889, 7440, 366, 1073, 393, 953, 29877, 2397, 508, 367, 1304, 304, 27769, 23023,
          1080, 322, 21737, 297, 263, 2090, 322, 1708, 1319, 982, 29973, 29871, 243, 162, 155,
          135, 2688, 508, 884, 367, 1304, 304, 788, 263, 6023, 310, 2022, 2877, 304, 596, 7191,
          322, 11803, 29889, 29871, 243, 162, 149, 152, 1126, 29892, 1258, 366, 1073, 393, 727,
          526, 1584, 953, 29877, 2397, 8090, 322, 14188, 366, 508, 1708, 29973, 29871, 243, 162,
          145, 177, 243, 162, 148, 131, 1105, 29892, 748, 14432, 322, 679, 907, 1230, 411, 953,
          29877, 2397, 29991, 29871, 243, 162, 149, 168, 243, 162, 145, 171]

DECODED_PARAGRAPH = (
    "Sure, here's a short paragraph about emoji, "
    "where each word is followed by an emoji:\n\n"
    "Hey there! 👋 I'm here to help answer any questions you have about emoji 🤔. "
    "Did you know that emoji can be used to convey emotions and feelings in a "
    "fun and playful way? 😄 "
    "They can also be used to add a touch of personality to your messages and posts. 💕 "
    "And, did you know that there are even emoji games and activities you can play? 🎮👀 "
    "So, go ahead and get creative with emoji! 💥🎨"
)
# fmt: on


@require_test_tokenizers("Llama-2-7b-chat-hf-q4f16_1-MLC")
def test_text_streamer(llama_tokenizer_path: str):  # pylint: disable=redefined-outer-name
    text_streamer = TextStreamer(Tokenizer(llama_tokenizer_path))
    total_text = ""
    for token in para_input_tokens:
        total_text += text_streamer.put([token])
    total_text += text_streamer.finish()

    assert total_text == DECODED_PARAGRAPH


def stop_handler_process_tokens(
    stop_handler: StopStrHandler, tokens: List[int], tokenizer: Tokenizer
) -> str:
    returned_tokens = []
    for token in tokens:
        returned_tokens += stop_handler.put(token)
        if stop_handler.stop_triggered:
            break

    if not stop_handler.stop_triggered:
        returned_tokens += stop_handler.finish()

    return tokenizer.decode(returned_tokens)


@require_test_tokenizers("Llama-2-7b-chat-hf-q4f16_1-MLC")
def test_stop_str_handler_stop(llama_tokenizer_path: str):  # pylint: disable=redefined-outer-name
    stop_strs = [" 🤔"]
    tokenizer = Tokenizer(llama_tokenizer_path)
    stop_handler = StopStrHandler(stop_strs, tokenizer)

    total_text = stop_handler_process_tokens(stop_handler, para_input_tokens, tokenizer)
    expected_text = (
        "Sure, here's a short paragraph about emoji, "
        "where each word is followed by an emoji:\n\n"
        "Hey there! 👋 I'm here to help answer any questions you have about emoji"
    )

    assert total_text == expected_text


@require_test_tokenizers("Llama-2-7b-chat-hf-q4f16_1-MLC")
def test_stop_str_handler_not_stop(
    llama_tokenizer_path: str,  # pylint: disable=redefined-outer-name
):
    stop_strs = ["^^"]
    tokenizer = Tokenizer(llama_tokenizer_path)
    stop_handler = StopStrHandler(stop_strs, tokenizer)

    total_text = stop_handler_process_tokens(stop_handler, para_input_tokens, tokenizer)
    assert total_text == DECODED_PARAGRAPH


@require_test_tokenizers("Llama-2-7b-chat-hf-q4f16_1-MLC")
def test_stop_str_handler_return_cached_tokens(
    llama_tokenizer_path: str,  # pylint: disable=redefined-outer-name
):
    tokens = para_input_tokens[:26]  # until "\n\n"
    stop_strs = ["\n\n\n"]
    tokenizer = Tokenizer(llama_tokenizer_path)
    stop_handler = StopStrHandler(stop_strs, tokenizer)

    total_text = stop_handler_process_tokens(stop_handler, tokens, tokenizer)
    expected_text = (
        "Sure, here's a short paragraph about emoji, "
        "where each word is followed by an emoji:\n\n"
    )

    assert total_text == expected_text


@require_test_tokenizers("Llama-2-7b-chat-hf-q4f16_1-MLC")
def test_stop_str_handler_throughput(
    llama_tokenizer_path: str,  # pylint: disable=redefined-outer-name
):
    stop_strs = ["[INST]"]
    tokenizer = Tokenizer(llama_tokenizer_path)
    stop_handler = StopStrHandler(stop_strs, tokenizer)

    tokens = para_input_tokens * 20
    returned_tokens = []

    tbegin = time.perf_counter()
    for token in tokens:
        returned_tokens += stop_handler.put(token)
        assert not stop_handler.stop_triggered
    tend = time.perf_counter()

    throughput = len(tokens) / (tend - tbegin)
    print(
        f"num tokens = {len(tokens)}, "
        f"time elapsed = {tend - tbegin:.5f} sec, "
        f"throughput = {throughput}"
    )
    assert throughput >= 100000


emoji_tokens_expected_result = [
    # HF: "�����", SentencePiece: "�👀"
    ([177, 243, 162, 148, 131], ("�����", "�👀")),
    # Both: "👀👀"
    ([243, 162, 148, 131, 243, 162, 148, 131], ("👀👀",)),
    # Both: "👀👀👀"
    ([243, 162, 148, 131, 243, 162, 148, 131, 243, 162, 148, 131], ("👀👀👀",)),
    # HF: "👀�������", SentencePiece: "👀���👀"
    ([243, 162, 148, 131, 162, 148, 131, 243, 162, 148, 131], ("👀�������", "👀���👀")),
    # Both: "👀��� have👀"
    ([243, 162, 148, 131, 162, 148, 131, 505, 243, 162, 148, 131], ("👀��� have👀",)),
]


@pytest.mark.parametrize("tokens_and_results", emoji_tokens_expected_result)
@require_test_tokenizers("Llama-2-7b-chat-hf-q4f16_1-MLC")
def test_text_streamer_emojis(
    llama_tokenizer_path: str, tokens_and_results: Tuple[List[int], Tuple[str]]
):  # pylint: disable=redefined-outer-name
    text_streamer = TextStreamer(Tokenizer(llama_tokenizer_path))
    total_text = ""
    tokens, expected_results = tokens_and_results
    for token in tokens:
        total_text += text_streamer.put([token])
    total_text += text_streamer.finish()
    assert total_text in expected_results


if __name__ == "__main__":
    test_text_streamer()
    test_stop_str_handler_stop()
    test_stop_str_handler_not_stop()
    test_stop_str_handler_return_cached_tokens()
    test_stop_str_handler_throughput()

    for tokens_and_res in emoji_tokens_expected_result:
        test_text_streamer_emojis(tokens_and_res)
