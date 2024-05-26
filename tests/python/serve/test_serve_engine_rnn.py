# pylint: disable=chained-comparison,line-too-long,missing-docstring,
# pylint: disable=too-many-arguments,too-many-locals,unused-argument,unused-variable
from typing import List

from mlc_llm.protocol.generation_config import GenerationConfig
from mlc_llm.serve import EngineConfig, MLCEngine

prompts = [
    "What is the meaning of life?",
    "Introduce the history of Pittsburgh to me. Please elaborate in detail.",
    "Write a three-day Seattle travel plan. Please elaborate in detail.",
    "What is Alaska famous of? Please elaborate in detail.",
    "What is the difference between Lambda calculus and Turing machine? Please elaborate in detail.",
    "What are the necessary components to assemble a desktop computer? Please elaborate in detail.",
    "Why is Vitamin D important to human beings? Please elaborate in detail.",
    "Where is milk tea originated from? Please elaborate in detail.",
    "Where is the southernmost place in United States? Please elaborate in detail.",
    "Do you know AlphaGo? What capabilities does it have, and what achievements has it got? Please elaborate in detail.",
]


def test_engine_generate() -> None:
    engine = MLCEngine(
        model="dist/rwkv-6-world-1b6-q0f16-MLC",
        model_lib="dist/rwkv-6-world-1b6-q0f16-MLC/rwkv-6-world-1b6-q0f16-MLC-cuda.so",
        mode="server",
        engine_config=EngineConfig(
            max_num_sequence=8,
            max_history_size=1,
        ),
    )

    num_requests = 10
    max_tokens = 256
    generation_cfg = GenerationConfig(max_tokens=max_tokens, n=7)

    output_texts: List[List[str]] = [
        ["" for _ in range(generation_cfg.n)] for _ in range(num_requests)
    ]
    for rid in range(num_requests):
        print(f"generating for request {rid}")
        for delta_outputs in engine._generate(prompts[rid], generation_cfg, request_id=str(rid)):
            assert len(delta_outputs) == generation_cfg.n
            for i, delta_output in enumerate(delta_outputs):
                output_texts[rid][i] += delta_output.delta_text

    # Print output.
    print("All finished")
    for req_id, outputs in enumerate(output_texts):
        print(f"Prompt {req_id}: {prompts[req_id]}")
        if len(outputs) == 1:
            print(f"Output {req_id}:{outputs[0]}\n")
        else:
            for i, output in enumerate(outputs):
                print(f"Output {req_id}({i}):{output}\n")

    engine.terminate()
    del engine


if __name__ == "__main__":
    test_engine_generate()
