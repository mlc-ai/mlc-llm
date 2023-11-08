import requests
import json

# Get a response using a prompt without streaming
payload = {
   "model": "Llama-2-7b-chat-hf-q0f16",
   "prompt": "",    # required parameter, do it dummy
   "context": "Cheerleading: A group of cheerleaders are seen walking on stage while the audience cheers. The group",
   "continuation": " start performing cheer roping on a fake stage while many watch from the stands.",
   "stream": False,
   "logprobs": True
}
response = requests.post("http://127.0.0.1:8000/v1/completions", json=payload)

if response.status_code != 200:
   print(f"Error: {response.status_code} - {response.text}")

response = json.loads(response.text)
logprob_dict = json.loads(response['choices'][0]['text'])
log_probes = logprob_dict["logprobes"]
is_greedy = logprob_dict["is_greedy"]
print(f"Logprobs: {log_probes}; Is_greedy: {is_greedy}")

# Get the latest runtime stats
r = requests.get("http://127.0.0.1:8000/stats")
print(f"Runtime stats: {r.json()}\n")
