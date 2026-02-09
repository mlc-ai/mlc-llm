import numpy as np
import pytest
import scipy.special
import tvm
from tvm.s_tir import dlight

# test category "op_correctness"
pytestmark = [pytest.mark.op_correctness]


def test_two_stage_softmax():
    from mlc_llm.compiler_pass.rewrite_softmax import _get_lse_and_softmax_func

    chunk_size = 4096
    target = tvm.target.Target("cuda")
    f_chunk_lse, f_softmax_with_lse = _get_lse_and_softmax_func(target, chunk_size)
    mod = tvm.IRModule({"chunk_lse": f_chunk_lse, "softmax_with_chunked_lse": f_softmax_with_lse})
    with target:
        mod = dlight.ApplyDefaultSchedule(dlight.gpu.GeneralReduction())(mod)

    runtime_mod = tvm.build(mod, target=target)
    device = tvm.cuda()

    num_runs = 5
    vocab_size = 128256
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
        for _ in range(num_runs):
            x_np = np.random.uniform(low=-10, high=10, size=(batch_size, vocab_size)).astype(
                "float32"
            )
            y_np = scipy.special.softmax(x_np, axis=-1)

            x_nd = tvm.runtime.tensor(x_np, device=device)
            r_nd = tvm.runtime.empty(
                (batch_size, (vocab_size + chunk_size - 1) // chunk_size),
                x_np.dtype,
                device=device,
            )
            y_nd = tvm.runtime.empty(x_np.shape, x_np.dtype, device=device)

            runtime_mod["chunk_lse"](x_nd, r_nd)
            runtime_mod["softmax_with_chunked_lse"](x_nd, r_nd, y_nd)

            y_nd_arr = y_nd.numpy()
            np.testing.assert_allclose(y_nd_arr, y_np, atol=1e-6, rtol=1e-6)

        print(f"pass batch size {batch_size}")


if __name__ == "__main__":
    test_two_stage_softmax()
