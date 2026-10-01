import requests
import json

tools = [
   {
      "type": "function",
      "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
               "type": "object",
               "properties": {
                  "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                  },
                  "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
               },
               "required": ["location"],
            },
      },
   }
]

payload = {
   "model": "HF://mlc-ai/gorilla-openfunctions-v2-q4f16_1-MLC",
#   "model": "HF://mlc-ai/gemma-2b-it-q4f16_1-MLC",
   "messages": [
      {
            "role": "user",
            "content": "What is the current weather in Pittsburgh, PA in fahrenheit?",
      }
   ],
   "stream": False,
   "tools": tools,
}

r = requests.post("http://127.0.0.1:8000/v1/chat/completions", json=payload)
print(f"{r.json()['choices'][0]['message']['tool_calls'][0]['function']}\n")

# Output: {'name': 'get_current_weather', 'arguments': {'location': 'Pittsburgh, PA', 'unit': 'fahrenheit'}}
