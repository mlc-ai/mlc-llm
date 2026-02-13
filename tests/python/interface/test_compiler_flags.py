# pylint: disable=missing-docstring
import pytest

from mlc_llm.interface.compiler_flags import ModelConfigOverride


def test_model_config_override_parse_kv_cache_dtype_int8():
    overrides = ModelConfigOverride.from_str("tensor_parallel_shards=2;kv_cache_dtype=int8")
    assert overrides.tensor_parallel_shards == 2
    assert overrides.kv_cache_dtype == "int8"


def test_model_config_override_parse_kv_cache_dtype_auto():
    overrides = ModelConfigOverride.from_str("kv_cache_dtype=auto")
    assert overrides.kv_cache_dtype is None


@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
def test_model_config_override_parse_kv_cache_dtype_fp8(dtype: str):
    overrides = ModelConfigOverride.from_str(f"kv_cache_dtype={dtype}")
    assert overrides.kv_cache_dtype == dtype


def test_model_config_override_parse_kv_cache_dtype_reject_invalid():
    with pytest.raises(ValueError):
        ModelConfigOverride.from_str("kv_cache_dtype=uint8")
