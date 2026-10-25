"""Native embedding copies preserve the surrounding rows."""

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx

import mlc_llm  # noqa: F401

pytestmark = [pytest.mark.unittest]


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("offset", [0, 3, 6])
def test_copy_embedding_to_offset(dtype, offset):
    embedding = np.arange(3 * 16, dtype=dtype).reshape(3, 16)
    destination = np.full((9, 16), -1, dtype=dtype)
    source_tensor = tvm_tirx.runtime.tensor(embedding, tvm_tirx.cpu())
    destination_tensor = tvm_tirx.runtime.tensor(destination, tvm_tirx.cpu())

    tvm_ffi.get_global_func("mlc.copy_embedding_to_offset")(
        source_tensor, destination_tensor, offset
    )

    destination[offset : offset + 3] = embedding
    np.testing.assert_array_equal(destination_tensor.numpy(), destination)
