import json

import pytest
from fastapi.testclient import TestClient
from httpx_sse import connect_sse
from mlc_serve.api import create_app
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


@pytest.mark.timeout(3)
def test_chat_completion(client):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "test", "messages": "test prompt", "max_tokens": 10},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == " test" * 10


@pytest.mark.timeout(3)
def test_stream_chat_completion(client):
    data = []
    with connect_sse(
        client,
        "POST",
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": "test prompt",
            "max_tokens": 10,
            "stream": True,
        },
    ) as event_source:
        for sse in event_source.iter_sse():
            data.append(sse.data)

    events = [json.loads(d) for d in data[:-1]]

    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    assert events[0]["choices"][0]["delta"]["content"] == ""

    assert all(e["choices"][0]["delta"]["content"] == " test" for e in events[1:-1])

    assert events[-1]["choices"][0]["delta"] == {}
    assert events[-1]["choices"][0]["finish_reason"] == "length"

    assert data[-1] == "[DONE]"
    
