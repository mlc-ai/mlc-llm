import json

import requests


class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


# Get a response using a prompt without streaming
payload = {
    "model": "vicuna-v1-7b",
    "messages": [{"role": "user", "content": "Write a haiku"}],
    "stream": False,
}
r = requests.post("http://127.0.0.1:8000/v1/chat/completions", json=payload)
print(
    f"{color.BOLD}Without streaming:{color.END}\n{color.GREEN}{r.json()['choices'][0]['message']['content']}{color.END}\n"
)

# Reset the chat
r = requests.post("http://127.0.0.1:8000/chat/reset", json=payload)
print(f"{color.BOLD}Reset chat:{color.END} {str(r)}\n")

# Get a response using a prompt with streaming
payload = {
    "model": "vicuna-v1-7b",
    "messages": [{"role": "user", "content": "Write a haiku"}],
    "stream": True,
}
with requests.post("http://127.0.0.1:8000/v1/chat/completions", json=payload, stream=True) as r:
    print(f"{color.BOLD}With streaming:{color.END}")
    for chunk in r:
        if chunk[6:].decode("utf-8").strip() == "[DONE]":
            break
        content = json.loads(chunk[6:])["choices"][0]["delta"].get("content", "")
        print(f"{color.GREEN}{content}{color.END}", end="", flush=True)
    print("\n")

# Get the latest runtime stats
r = requests.get("http://127.0.0.1:8000/stats")
print(f"{color.BOLD}Runtime stats:{color.END} {r.json()}\n")
