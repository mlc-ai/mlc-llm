"""Gemma 3 keeps global context while applying local windows per layer."""

import pytest
from tvm import relax

from mlc_llm.model.gemma3.gemma3_model import Gemma3Config, Gemma3ForCausalLM
from mlc_llm.nn import RopeMode


@pytest.mark.parametrize("chunk_size,expected", [(128, 128), (1024, 512)])
def test_gemma3_layer_windows(chunk_size, expected):
    config = Gemma3Config.from_dict(
        {
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 6,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "vocab_size": 256,
            "sliding_window": 512,
            "prefill_chunk_size": chunk_size,
        }
    )
    assert config.sliding_window_size == -1
    assert config.prefill_chunk_size == expected
    model = Gemma3ForCausalLM(config)
    mod, _ = model.export_tvm(spec=model.get_default_spec())
    cache_calls = []

    def collect(expr):
        if (
            isinstance(expr, relax.Call)
            and getattr(expr.op, "name", None) == "relax.call_pure_packed"
            and getattr(expr.args[0], "global_symbol", None) == "mlc.create_paged_kv_cache_generic"
        ):
            cache_calls.append(expr)

    relax.analysis.post_order_visit(mod["create_paged_kv_cache"].body, collect)
    assert len(cache_calls) == 1
    args = cache_calls[0].args
    assert [kind.value for kind in args[1]] == ["mha_sliding"] * 5 + ["mha"]
    assert int(args[2].values[-1]) == 512
    assert int(args[11].value) == RopeMode.NONE
