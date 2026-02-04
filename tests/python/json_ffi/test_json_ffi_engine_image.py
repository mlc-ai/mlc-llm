import base64
from typing import Dict, List, Optional

import requests

from mlc_llm.json_ffi import JSONFFIEngine
from mlc_llm.testing import require_test_model


def base64_encode_image(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()  # Ensure we got a successful response
    image_data = base64.b64encode(response.content)
    image_data_str = image_data.decode("utf-8")
    data_url = f"data:image/jpeg;base64,{image_data_str}"
    return data_url


image_prompts = [
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": f"{base64_encode_image('https://llava-vl.github.io/static/images/view.jpg')}",
                },
                {"type": "text", "text": "What does the image represent?"},
            ],
        }
    ]
]


def run_chat_completion(
    engine: JSONFFIEngine,
    model: str,
    prompts: List[List[Dict]] = image_prompts,
    tools: Optional[List[Dict]] = None,
):
    num_requests = 1
    max_tokens = 64
    n = 1
    output_texts: List[List[str]] = [["" for _ in range(n)] for _ in range(num_requests)]

    for rid in range(num_requests):
        print(f"chat completion for request {rid}")
        for response in engine.chat.completions.create(
            messages=prompts[rid],
            model=model,
            max_tokens=max_tokens,
            n=n,
            request_id=str(rid),
            tools=tools,
        ):
            for choice in response.choices:
                assert choice.delta.role == "assistant"
                assert isinstance(choice.delta.content[0], Dict)
                assert choice.delta.content[0]["type"] == "text"
                output_texts[rid][choice.index] += choice.delta.content[0]["text"]

    # Print output.
    print("Chat completion all finished")
    for req_id, outputs in enumerate(output_texts):
        print(f"Prompt {req_id}: {prompts[req_id]}")
        if len(outputs) == 1:
            print(f"Output {req_id}:{outputs[0]}\n")
        else:
            for i, output in enumerate(outputs):
                print(f"Output {req_id}({i}):{output}\n")


@require_test_model("llava-1.5-7b-hf-q4f16_1-MLC")
def test_chat_completion():
    # Create engine.
    engine = JSONFFIEngine(
        model,
        max_total_sequence_length=1024,
    )

    run_chat_completion(engine, model)

    # Test malformed requests.
    for response in engine._raw_chat_completion("malformed_string", n=1, request_id="123"):
        assert len(response.choices) == 1
        assert response.choices[0].finish_reason == "error"

    engine.terminate()


if __name__ == "__main__":
    test_chat_completion()
