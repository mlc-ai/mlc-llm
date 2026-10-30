"""Non-streaming MLCEngine/AsyncMLCEngine completions keep every token's logprobs."""

import asyncio
import json

import pytest

from mlc_llm.protocol import openai_api_protocol
from mlc_llm.serve import engine_base
from mlc_llm.serve.engine import AsyncMLCEngine, MLCEngine

pytestmark = [pytest.mark.unittest]

TOKENS = [" 2", ",", " and"]


def _logprob_json(token: str, logprob: float) -> str:
    return json.dumps(
        {
            "token": token,
            "logprob": logprob,
            "bytes": None,
            "top_logprobs": [{"token": token, "logprob": logprob, "bytes": None}],
        }
    )


def _chunks(request: openai_api_protocol.CompletionRequest, request_id: str):
    """What _handle_completion yields for one token per engine step, then usage."""
    finish_reasons = [None]
    state = engine_base.EngineState(enable_tracing=False)
    for i, token in enumerate(TOKENS):
        delta = engine_base.CallbackStreamOutput(
            delta_text=token,
            delta_logprob_json_strs=[_logprob_json(token, -0.1 * (i + 1))],
            finish_reason="length" if i == len(TOKENS) - 1 else None,
            request_final_usage_json_str=None,
        )
        yield engine_base.process_completion_stream_output(
            [delta], request, request_id, state, finish_reasons
        )
    yield openai_api_protocol.CompletionResponse(
        id=request_id,
        choices=[],
        usage=openai_api_protocol.CompletionUsage(
            prompt_tokens=4, completion_tokens=3, total_tokens=7
        ),
    )


def test_sync_completion_keeps_all_logprobs():
    engine = object.__new__(MLCEngine)
    engine._handle_completion = _chunks
    response = engine._completion(prompt="1 + 1 =", logprobs=1, max_tokens=3)
    assert response.choices[0].text == "".join(TOKENS)
    assert response.choices[0].logprobs.tokens == TOKENS


def test_async_completion_keeps_all_logprobs():
    async def _async_chunks(request, request_id, request_final_usage_include_extra):
        for chunk in _chunks(request, request_id):
            yield chunk

    engine = object.__new__(AsyncMLCEngine)
    engine._handle_completion = _async_chunks
    response = asyncio.run(engine._completion(prompt="1 + 1 =", logprobs=1, max_tokens=3))
    assert response.choices[0].text == "".join(TOKENS)
    assert response.choices[0].logprobs.tokens == TOKENS
