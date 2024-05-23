from mlc_llm.serve import DebugConfig, GenerationConfig
from mlc_llm.serve.sync_engine import SyncMLCEngine

prompts = [
    "The meaning of life is",
    "According to the history of Pittsburgh,",
    "I have a three-day Seattle travel plan. On the first day,",
    "Undoubtedly, Alaska is one of the most beautiful places on Earth,",
    "Explain difference between Lambda calculus and Turing machine is",
    "To assemble a desktop computer, we need the necessary components of",
    "Vitamin D is important to human beings, because",
    "Refer to history, the milk tea is originated from",
    "In the southernmost place in United States,",
    "AlphaGo has the capabilities of",
]


def test_engine_system_prompt(engine):
    system_prompt = "This is a system prompt"
    system_prompt_tokens = len(engine.tokenizer.encode(system_prompt))
    max_tokens = 8
    _, _ = engine.generate(
        system_prompt,
        GenerationConfig(
            temperature=0,
            max_tokens=max_tokens,
            debug_config=DebugConfig(pinned_system_prompt=True),
        ),
    )
    metrics = engine.metrics()
    assert metrics["sum_num_prefill_tokens"] == system_prompt_tokens
    sum_prefill_tokens = system_prompt_tokens

    input_token_lens = [len(engine.tokenizer.encode(prompt)) for prompt in prompts]

    generation_config = GenerationConfig(temperature=0, max_tokens=max_tokens)
    _, _ = engine.generate(prompts, generation_config)
    metrics = engine.metrics()
    assert metrics["sum_num_prefill_tokens"] == sum_prefill_tokens + sum(input_token_lens)
    sum_prefill_tokens = metrics["sum_num_prefill_tokens"]

    _, _ = engine.generate(system_prompt + " and why ?", generation_config)
    metrics = engine.metrics()
    # system prompt is reused entirely
    assert metrics["sum_num_prefill_tokens"] == sum_prefill_tokens + 3
    sum_prefill_tokens = metrics["sum_num_prefill_tokens"]

    _, _ = engine.generate(prompts[:4], generation_config)
    metrics = engine.metrics()
    # first 4 prompts are removed and need to prefill again
    assert metrics["sum_num_prefill_tokens"] == sum_prefill_tokens + sum(input_token_lens[:4])


def test_engine_multi_round(engine):
    num_requests = 10
    max_tokens = 8
    generation_config = GenerationConfig(temperature=0, max_tokens=max_tokens)
    input_token_lens = [len(engine.tokenizer.encode(prompt)) for prompt in prompts[:num_requests]]

    output_texts, _ = engine.generate(prompts[:num_requests], generation_config)
    metrics = engine.metrics()
    assert metrics["sum_num_prefill_tokens"] == sum(input_token_lens)
    sum_prefill_tokens = metrics["sum_num_prefill_tokens"]
    concat_prompt = []
    for i, output in enumerate(output_texts):
        concat_prompt.append(prompts[i] + " " + output[0] + " ?")
    output_texts, _ = engine.generate(concat_prompt[:num_requests], generation_config)
    metrics = engine.metrics()
    assert metrics["sum_num_prefill_tokens"] == sum_prefill_tokens + 2 * num_requests


def test_basic_engine_system_prompt():
    # Create engine
    model = "HF://mlc-ai/Llama-2-7b-chat-hf-q0f16-MLC"
    engine = SyncMLCEngine(
        model=model,
        mode="local",
        max_total_sequence_length=4096,
        prefix_cache_max_num_recycling_seqs=5,
    )
    test_engine_system_prompt(engine)


def test_basic_engine_multi_round():
    # Create engine
    model = "HF://mlc-ai/Llama-2-7b-chat-hf-q0f16-MLC"
    engine = SyncMLCEngine(
        model=model,
        mode="server",
        max_total_sequence_length=4096,
    )
    test_engine_multi_round(engine)


def test_engine_spec_multi_round():
    # Create engine
    model = "HF://mlc-ai/Llama-2-7b-chat-hf-q0f16-MLC"
    small_model = "HF://mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC"

    engine = SyncMLCEngine(
        model=model,
        mode="server",
        max_total_sequence_length=4096,
        additional_models=[small_model],
        speculative_mode="small_draft",
    )

    test_engine_multi_round(engine)


def test_engine_eagle_multi_round():
    # Create engine
    model = "HF://mlc-ai/Llama-2-7b-chat-hf-q0f16-MLC"
    small_model = "dist/Eagle-llama2-7b-chat-q0f16-MLC"
    small_model_lib = "dist/Eagle-llama2-7b-chat-q0f16-MLC/Eagle-llama2-7b-chat-q0f16-MLC-cuda.so"
    engine = SyncMLCEngine(
        model=model,
        mode="server",
        max_total_sequence_length=4096,
        additional_models=[small_model + ":" + small_model_lib],
        speculative_mode="eagle",
        max_batch_size=80,
    )

    test_engine_multi_round(engine)


if __name__ == "__main__":
    test_basic_engine_system_prompt()
    test_basic_engine_multi_round()
    test_engine_spec_multi_round()
    test_engine_eagle_multi_round()
