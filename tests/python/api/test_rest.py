# pylint: disable=missing-docstring
import os
import pytest
import requests
import subprocess
import time
import signal
import json

MODELS = ["Llama-2-7b-chat-hf-q4f16_1", "Mistral-7B-v0.1-q4f16_1"]


@pytest.fixture
def run_rest_server(model):
    cmd = f"python -m mlc_chat.rest --model {model}"
    os.environ["PYTHONPATH"] = "./python"
    server_proc = subprocess.Popen(cmd.split())
    # wait for server to start
    while True:
        try:
            _ = requests.get("http://localhost:8000/stats")
            break
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    yield
    server_proc.send_signal(signal.SIGINT)
    server_proc.wait()


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("model", MODELS)
def test_rest_api(run_rest_server, model, stream):
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
            "http://127.0.0.1:8000/v1/chat/completions", json=payload, stream=True
        ) as r:
            print(f"With streaming:")
            for chunk in r:
                content = json.loads(chunk[6:-2])["choices"][0]["delta"].get("content", "")
                print(f"{content}", end="", flush=True)
            print("\n")
    else:
        r = requests.post("http://127.0.0.1:8000/v1/chat/completions", json=payload)
        print(f"\n{r.json()['choices'][0]['message']['content']}\n")
