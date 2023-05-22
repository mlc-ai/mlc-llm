import requests

# To launch the server, run
# $ uvicorn server:app --reload

# List the models that are currently supported
r = requests.get("http://127.0.0.1:8000/models")
print(f"Supported models: {r.json()}")

# Models are not initialized by default
r = requests.get("http://127.0.0.1:8000/models/init")
print(f"Model initialized: {r.json()}")

# Initialize a model
payload = {
    "model": "vicuna-v1-7b",
    "mlc_path": "/home/sudeepag/mlc-llm",
    "artifact_path": "/home/sudeepag/mlc-llm/dist",
    "device_name": "vulkan"
}
r = requests.post("http://127.0.0.1:8000/models/init", json=payload)
print(r)

# Verify that a model has now been initialized
r = requests.get("http://127.0.0.1:8000/models/init")
print(f"Model initialized: {r.json()}")

# Get a response using a prompt
payload = {
    "prompt": "Write a haiku"
}
r = requests.post("http://127.0.0.1:8000/chat/completions", json=payload)
print(r.json())