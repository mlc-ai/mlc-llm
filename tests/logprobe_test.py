import requests
import json

# Get a response using a prompt without streaming
payload = {
   "model": "Llama-2-7b-chat-hf-q0f16",
   "context": "Cheerleading: A group of cheerleaders are seen walking on stage while the audience cheers. The group",
   "continuation": " start performing cheer roping on a fake stage while many watch from the stands.",
}
response = requests.post("http://127.0.0.1:8000/v1/logprob", json=payload)

if response.status_code != 200:
   print(f"Error: {response.status_code} - {response.text}")

response = json.loads(response.text)
log_probes = response["logprob"]       # expected -84.03089141845703
is_greedy = response["is_greedy"]      # expected False
print(f"Logprobs: {log_probes}; Is_greedy: {is_greedy}")
