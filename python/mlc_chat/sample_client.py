import requests
import json

# To launch the server, run
# $ python -m mlc_chat.rest

# List the models that are currently supported
r = requests.get("http://127.0.0.1:8000/models")
print(f"Supported models: {r.json()}\n")

# Get a response using a prompt without streaming
payload = {
    "prompt": "Write a haiku"
}
r = requests.post("http://127.0.0.1:8000/chat/completions", json=payload)
print(f"Without streaming: {r.json()}\n")

# Reset the chat
r = requests.post("http://127.0.0.1:8000/chat/reset", json=payload)
print(f"Reset chat: {str(r)}\n")

# Get a response using a prompt with streaming
payload = {
    "prompt": "Write a haiku",
    "stream": True
}
with requests.post("http://127.0.0.1:8000/chat/completions", json=payload, stream=True) as r:
    print("With streaming: ")
    try:
        for data in r.iter_content(chunk_size=1024):
            if data:
                print(json.loads(data))
    except requests.exceptions.ChunkedEncodingError as ex:
        print(f"Invalid chunk encoding {str(ex)}")
    print("\n")

# Get the latest runtime stats
r = requests.get("http://127.0.0.1:8000/stats")
print(f"Runtime stats: {r.json()}\n")
