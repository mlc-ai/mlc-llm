import pytest

from mlc_llm.protocol.generation_config import GenerationConfig
from mlc_llm.serve import EngineConfig
from mlc_llm.serve.sync_engine import SyncMLCEngine
from mlc_llm.testing import require_test_model


@pytest.mark.parametrize("prefix_cache_mode", ["radix", "disable"])
@require_test_model("LFM2.5-350M-q4f16_1-MLC")
def test_hybrid_repeated_prefixes(model: str, prefix_cache_mode: str):
    engine = SyncMLCEngine(
        model=model,
        mode="interactive",
        engine_config=EngineConfig(
            max_num_sequence=1,
            max_total_sequence_length=4096,
            max_single_sequence_length=4096,
            prefill_chunk_size=1024,
            prefix_cache_mode=prefix_cache_mode,
        ),
    )
    context = "The blue bird sits beside a red house. A green tree grows nearby. " * 20
    prompt = context + "What color is the bird?"
    prompts = [prompt, prompt, context + "What color is the tree?"]
    outputs = []
    for request in prompts:
        result, _ = engine.generate(request, GenerationConfig(temperature=0, max_tokens=16))
        assert result[0][0].strip()
        outputs.append(result[0][0])
    assert outputs[0] == outputs[1]
    # Hybrid recurrent state must be computed from the full input for each request.
    assert engine.metrics()["prefill_tokens_sum"] == sum(
        len(engine.tokenizer.encode(request)) for request in prompts
    )


@require_test_model("LFM2.5-350M-q4f16_1-MLC")
def test_hybrid_speculative_decoding_rejected(model: str):
    with pytest.raises(
        RuntimeError, match="Speculative decoding is not supported for hybrid models"
    ):
        SyncMLCEngine(
            model=model,
            mode="interactive",
            engine_config=EngineConfig(
                additional_models=[model],
                speculative_mode="small_draft",
                max_num_sequence=1,
                max_total_sequence_length=4096,
                max_single_sequence_length=4096,
                prefill_chunk_size=1024,
            ),
        )
