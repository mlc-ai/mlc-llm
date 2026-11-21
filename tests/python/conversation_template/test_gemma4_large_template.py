"""Pinned Hugging Face Gemma 4 chat template cases."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlc_llm.conversation_template import ConvTemplateRegistry
from mlc_llm.protocol import openai_api_protocol
from mlc_llm.serve.engine_base import process_chat_completion_request

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures/gemma4/large_text_chat.json"
CASES = json.loads(FIXTURE.read_text())["cases"]


@pytest.mark.parametrize(
    "case", CASES, ids=[f"{case['name']}-{case['thinking']}" for case in CASES]
)
def test_hf_plain_text_prompt(case):
    name = "gemma4_large_thinking" if case["thinking"] else "gemma4_large"
    conv = ConvTemplateRegistry.get_conv_template(name).model_copy(deep=True)
    request = openai_api_protocol.ChatCompletionRequest.model_validate(
        {"messages": case["messages"]}
    )
    rendered = []

    def tokenize(text):
        rendered.append(text)
        return [7, 11]

    prompts, config, _, _ = process_chat_completion_request(
        request,
        "test",
        SimpleNamespace(record_event=lambda *_args, **_kwargs: None),
        {},
        tokenize,
        1024,
        conv,
    )
    assert rendered == [case["prompt"]]
    assert prompts == [[2, 7, 11]]
    assert config.stop_token_ids == [1, 106, 50]
