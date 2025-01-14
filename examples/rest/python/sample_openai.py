import openai

openai.api_key = "None"
openai.api_base = "http://127.0.0.1:8000/v1"

model = "vicuna-v1-7b"


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


# Chat completion example without streaming
print(f"{color.BOLD}OpenAI chat completion example without streaming:{color.END}\n")
completion = openai.ChatCompletion.create(
    model=model, messages=[{"role": "user", "content": "Write a poem about OpenAI"}]
)
print(f"{color.GREEN}{completion.choices[0].message.content}{color.END}\n\n")

# Chat completion example with streaming
print(f"{color.BOLD}OpenAI chat completion example with streaming:{color.END}\n")
res = openai.ChatCompletion.create(
    model=model, messages=[{"role": "user", "content": "Write a poem about OpenAI"}], stream=True
)
for chunk in res:
    content = chunk["choices"][0]["delta"].get("content", "")
    print(f"{color.GREEN}{content}{color.END}", end="", flush=True)
print("\n")

# Completion example
print(f"{color.BOLD}OpenAI completion example:{color.END}\n")
res = openai.Completion.create(prompt="Write a poem about OpenAI", model=model)
print(f"{color.GREEN}{res.choices[0].text}{color.END}\n\n")
