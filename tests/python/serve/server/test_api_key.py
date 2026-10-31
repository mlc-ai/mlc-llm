"""Tests that --api-key is enforced on the routes that run completions."""

import fastapi
import pytest
from fastapi.testclient import TestClient

from mlc_llm.protocol import openai_api_protocol
from mlc_llm.serve.entrypoints import microserving_entrypoints, openai_entrypoints
from mlc_llm.serve.server import ServerContext

pytestmark = [pytest.mark.unittest]

API_KEY = "test-key"
COMPLETION_BODY = {"model": "test-model", "prompt": "hi", "max_tokens": 2}
ROUTES = [
    ("/v1/completions", COMPLETION_BODY),
    ("/microserving/prep_recv", {**COMPLETION_BODY, "end": 1, "debug_config": {}}),
    (
        "/microserving/remote_send",
        {
            **COMPLETION_BODY,
            "begin": 0,
            "end": 1,
            "kv_addr_info": "",
            "recv_rank": 0,
            "debug_config": {},
        },
    ),
    ("/microserving/start_generate", {**COMPLETION_BODY, "begin": 0, "debug_config": {}}),
]


class FakeEngine:
    """An engine that answers every completion request with a fixed text."""

    def terminate(self):
        pass

    async def _handle_completion(self, request, request_id, request_final_usage_include_extra):
        yield openai_api_protocol.CompletionResponse(
            id=request_id,
            choices=[openai_api_protocol.CompletionResponseChoice(text="hello")],
            model=request.model,
        )
        yield openai_api_protocol.CompletionResponse(
            id=request_id,
            choices=[],
            model=request.model,
            usage=openai_api_protocol.CompletionUsage(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
                extra={"prefix_matched_length": 0, "kv_append_metadata": ""},
            ),
        )


def make_server(api_key):
    app = fastapi.FastAPI()
    app.include_router(openai_entrypoints.app)
    app.include_router(microserving_entrypoints.app)
    server_context = ServerContext()
    server_context.add_model("test-model", FakeEngine())
    server_context.api_key = api_key
    server_context.enable_debug = True
    return server_context, TestClient(app)


@pytest.mark.parametrize("path, body", ROUTES)
def test_missing_or_wrong_api_key_is_rejected(path, body):
    server_context, client = make_server(API_KEY)
    with server_context:
        assert client.post(path, json=body).status_code == 401
        wrong_key = {"Authorization": "Bearer wrong-key"}
        assert client.post(path, json=body, headers=wrong_key).status_code == 401


@pytest.mark.parametrize("path, body", ROUTES)
def test_right_api_key_is_accepted(path, body):
    server_context, client = make_server(API_KEY)
    with server_context:
        headers = {"Authorization": f"Bearer {API_KEY}"}
        assert client.post(path, json=body, headers=headers).status_code == 200


@pytest.mark.parametrize("path, body", ROUTES)
def test_no_api_key_configured_disables_authentication(path, body):
    server_context, client = make_server(None)
    with server_context:
        assert client.post(path, json=body).status_code == 200
