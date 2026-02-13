# pylint: disable=missing-docstring
import tvm

from mlc_llm.compiler_pass.dispatch_kv_cache_creation import DispatchKVCacheCreation


def test_attach_kv_cache_metadata_records_dtype():
    metadata = {}
    dispatch = DispatchKVCacheCreation(
        tvm.target.Target("llvm"), flashinfer=False, metadata=metadata
    )
    kwargs = {
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "qk_head_dim": 128,
        "dtype": "int8",
    }
    dispatch.attach_kv_cache_metadata(kwargs)
    assert metadata["kv_cache"]["dtype"] == "int8"
