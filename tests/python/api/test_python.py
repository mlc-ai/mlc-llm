# pylint: disable=missing-docstring
import pytest

from mlc_chat import ChatModule, GenerationConfig
from mlc_chat.callback import StreamToStdout

MODELS = ["Llama-2-7b-chat-hf-q4f16_1"]


@pytest.mark.parametrize("model", MODELS)
def test_chat_module_creation_and_generate(model: str):
    chat_module = ChatModule(model=model)
    _ = chat_module.generate(
        prompt="How to make a cake?",
    )
    print(f"Statistics: {chat_module.stats()}\n")


@pytest.mark.parametrize("model", MODELS)
def test_chat_module_creation_and_generate_with_stream(model: str):
    chat_module = ChatModule(model=model)
    _ = chat_module.generate(
        prompt="How to make a cake?",
        progress_callback=StreamToStdout(callback_interval=2),
    )
    print(f"Statistics: {chat_module.stats()}\n")


@pytest.mark.parametrize(
    "generation_config",
    [
        GenerationConfig(temperature=0.7, presence_penalty=0.1, frequency_penalty=0.5, top_p=0.9),
        GenerationConfig(stop=["cake", "make"], n=3),
        GenerationConfig(max_gen_len=40, repetition_penalty=1.2),
    ],
)
@pytest.mark.parametrize("model", MODELS)
def test_chat_module_generation_config(generation_config: GenerationConfig, model: str):
    chat_module = ChatModule(model=model)
    output = chat_module.generate(
        prompt="How to make a cake?",
        generation_config=generation_config,
    )
    print(output)
    print(f"Statistics: {chat_module.stats()}\n")
