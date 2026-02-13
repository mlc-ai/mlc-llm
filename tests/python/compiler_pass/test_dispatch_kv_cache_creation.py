# pylint: disable=missing-docstring,protected-access
import pytest
import tvm

from mlc_llm.compiler_pass.dispatch_kv_cache_creation import DispatchKVCacheCreation


def test_apply_kv_cache_dtype_override():
    metadata = {"kv_cache_dtype": "int8"}
    dispatch = DispatchKVCacheCreation(
        tvm.target.Target("llvm"), flashinfer=False, metadata=metadata
    )
    kwargs = {"dtype": "float16"}
    dispatch._apply_kv_cache_dtype_override(kwargs)
    assert kwargs["dtype"] == "int8"


def test_apply_kv_cache_dtype_override_auto_no_change():
    metadata = {"kv_cache_dtype": "auto"}
    dispatch = DispatchKVCacheCreation(
        tvm.target.Target("llvm"), flashinfer=False, metadata=metadata
    )
    kwargs = {"dtype": "float16"}
    dispatch._apply_kv_cache_dtype_override(kwargs)
    assert kwargs["dtype"] == "float16"


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


def test_apply_kv_cache_dtype_override_fp8_on_cuda():
    metadata = {"kv_cache_dtype": "float8_e4m3fn"}
    dispatch = DispatchKVCacheCreation(
        tvm.target.Target("cuda"), flashinfer=False, metadata=metadata
    )
    kwargs = {"dtype": "float16"}
    dispatch._apply_kv_cache_dtype_override(kwargs)
    assert kwargs["dtype"] == "float8_e4m3fn"


def test_apply_kv_cache_dtype_override_fp8_requires_cuda():
    metadata = {"kv_cache_dtype": "float8_e4m3fn"}
    dispatch = DispatchKVCacheCreation(
        tvm.target.Target("llvm"), flashinfer=False, metadata=metadata
    )
    kwargs = {"dtype": "float16"}
    with pytest.raises(ValueError, match="only supported on CUDA target"):
        dispatch._apply_kv_cache_dtype_override(kwargs)
