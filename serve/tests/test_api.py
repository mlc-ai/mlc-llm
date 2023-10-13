import pytest
from fastapi.testclient import TestClient
from mlc_serve.api import create_app
from mlc_serve.api.protocol import ChatCompletionRequest
from mlc_serve.engine import AsyncEngineConnector, InferenceEngine
from mlc_serve.engine.dummy import DummyInferenceEngine


@pytest.fixture
def engine() -> InferenceEngine:
    return DummyInferenceEngine()


@pytest.fixture
def client(engine):
    connector = AsyncEngineConnector(engine, engine_wait_timeout=0.1)
    app = create_app(connector)
    with TestClient(app) as client:
        yield client


def test_chat_completion(client):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "test", "messages": "test prompt", "max_tokens": 10},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == " test" * 10
