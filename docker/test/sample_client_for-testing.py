import requests
import json

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

# Get a response using a prompt without streaming
payload = {
#    "model": "HF://mlc-ai/gemma-2b-it-q4f16_1-MLC",
    "model": "HF://mlc-ai/Mistral-7B-Instruct-v0.2-q4f16_1-MLC",
    "messages": [{"role": "user", "content": "write a haiku"}],
    "stream": False
}
r = requests.post("http://127.0.0.1:8000/v1/chat/completions", json=payload)
print(f"{color.BOLD}Without streaming:{color.END}\n{color.GREEN}{r.json()['choices'][0]['message']['content']}{color.END}\n")


payload = {
#    "model": "HF://mlc-ai/gemma-2b-it-q4f16_1-MLC",
    "model": "HF://mlc-ai/Mistral-7B-Instruct-v0.2-q4f16_1-MLC",
    "messages": [{"role": "user", "content": "Write a 500 words essay about the civil war"}],
    "stream": True
}

print(f"{color.BOLD}With streaming:{color.END}")
with requests.post("http://127.0.0.1:8000/v1/chat/completions", json=payload, stream=True) as r:
   for chunk in r.iter_content(chunk_size=None):
      chunk = chunk.decode("utf-8")
      if "[DONE]" in chunk[6:]:
         break
      response = json.loads(chunk[6:])
      content = response["choices"][0]["delta"].get("content", "")
      print(f"{color.GREEN}{content}{color.END}", end="", flush=True)

print("\n")

