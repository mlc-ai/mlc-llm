# pylint: disable=missing-docstring
import json
import os
import signal
import subprocess
import time

import pytest
import requests

MODELS = ["Llama-2-7b-chat-hf-q4f16_1"]


@pytest.fixture
def run_rest_server(model):
    cmd = f"python -m mlc_chat.rest --model {model}"
    print(cmd)
    os.environ["PYTHONPATH"] = "./python"
    with subprocess.Popen(cmd.split()) as server_proc:
        # wait for server to start
        while True:
            try:
                _ = requests.get("http://localhost:8000/stats", timeout=5)
                break
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        yield
        server_proc.send_signal(signal.SIGINT)
        server_proc.wait()


@pytest.mark.usefixtures("run_rest_server")
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("model", MODELS)
def test_rest_chat_completions(model, stream):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Hello, I am Bob",
            },
            {
                "role": "assistant",
                "content": "Hello, I am a chatbot.",
            },
            {
                "role": "user",
                "content": "What is my name?",
            },
        ],
        "stream": stream,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "temperature": 1.0,
        "top_p": 0.95,
    }
    if stream:
        with requests.post(
            "http://127.0.0.1:8000/v1/chat/completions", json=payload, stream=True, timeout=120
        ) as model_response:
            print("With streaming:")
            for chunk in model_response:
                data = chunk[6:-2]
                if data != b"[DONE]":
                    content = json.loads(data)["choices"][0]["delta"].get("content", "")
                    print(f"{content}", end="", flush=True)
            print("\n")
    else:
        model_response = requests.post(
            "http://127.0.0.1:8000/v1/chat/completions", json=payload, timeout=120
        )
        print(f"\n{model_response.json()['choices'][0]['message']['content']}\n")


@pytest.mark.usefixtures("run_rest_server")
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("model", MODELS)
def test_rest_completions(model, stream):
    payload = {
        "model": model,
        "prompt": "What is the meaning of life?",
        "stream": stream,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "temperature": 1.0,
        "n": 3,
    }
    if stream:
        with requests.post(
            "http://127.0.0.1:8000/v1/completions", json=payload, stream=True, timeout=120
        ) as model_response:
            print("With streaming:")
            for chunk in model_response:
                data = chunk[6:-2]
                if data != b"[DONE]":
                    content = json.loads(data)["choices"][0]["text"]
                    print(f"{content}", end="", flush=True)
            print("\n")
    else:
        model_response = requests.post(
            "http://127.0.0.1:8000/v1/completions", json=payload, timeout=120
        )
        assert len(model_response.json()["choices"]) == 3
        print(f"\n{model_response.json()['choices'][0]['text']}\n")
