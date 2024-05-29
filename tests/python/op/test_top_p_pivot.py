import numpy as np
import pytest
import tvm
import tvm.testing

from mlc_llm.op.top_p_pivot import top_p_pivot, top_p_renorm

# mypy: disable-error-code="var-annotated"

# test category "op_correctness"
pytestmark = [pytest.mark.op_correctness]


@pytest.mark.parametrize("batch_size", [32, 64])
@pytest.mark.parametrize("vocab", [3, 32, 64, 128])
def test_top_p_renorm(batch_size, vocab):
    top_p = 0.95
    init_pivots_np = np.array([1 - top_p, 0.02, 0.01]).astype(np.float32)
    top_p_np = np.array([top_p]).astype(np.float32)

    p_np = np.random.exponential(3, size=(batch_size, vocab)).astype(np.float32)
    p_np /= np.sum(p_np, axis=-1, keepdims=True)
    final_pivot_np = np.zeros(batch_size).astype(np.float32)
    final_lsum_np = np.zeros(batch_size).astype(np.float32)

    dev = tvm.cuda(0)
    var_prob = tvm.nd.array(p_np, dev)
    var_init_pivots = tvm.nd.array(init_pivots_np, dev)
    top_p_global = tvm.nd.array(top_p_np, dev)
    var_final_pivot = tvm.nd.array(final_pivot_np, dev)
    var_final_lsum = tvm.nd.array(final_lsum_np, dev)

    kernel = top_p_pivot(init_pivots_np.shape[0])
    mod = tvm.build(kernel, target="cuda")
    mod(var_prob, top_p_global, var_init_pivots, var_final_pivot, var_final_lsum)

    final_pivot = var_final_pivot.asnumpy()
    final_lsum = var_final_lsum.asnumpy()

    renorm_np = p_np.copy()
    var_renorm = tvm.nd.array(renorm_np, dev)

    kernel_renorm = top_p_renorm()
    mod_renorm = tvm.build(kernel_renorm, target="cuda")
    mod_renorm(var_prob, var_final_pivot, var_final_lsum, var_renorm)

    renorm = var_renorm.asnumpy()

    def verify_pivot(probs: np.ndarray, pivot: float, lsum: float, renorm: np.ndarray):
        sorted_probs = np.sort(probs, axis=-1)[::-1]
        num_larger_than_pivot = np.sum(sorted_probs >= pivot)
        filtered_sorted_probs = sorted_probs[:num_larger_than_pivot]
        min_larger_than_pivot = min(filtered_sorted_probs)

        sum_larger_than_pivot = np.sum(np.where(sorted_probs >= pivot, sorted_probs, 0))
        sum_larger_than_pivot_exclude_min = np.sum(
            np.where(filtered_sorted_probs != min_larger_than_pivot, filtered_sorted_probs, 0)
        )

        probs[probs < pivot] = 0
        renorm_prob = probs / np.sum(probs, axis=-1, keepdims=True)
        try:
            assert sum_larger_than_pivot >= top_p
            assert sum_larger_than_pivot_exclude_min < top_p
            assert abs(lsum - sum_larger_than_pivot) < 1e-6
            assert np.allclose(renorm, renorm_prob, atol=1e-6, rtol=1e-6)
        except AssertionError:
            print("Failed")
            print("probs:", repr(probs))
            print("pivot:", pivot)
            print("sorted_probs:", sorted_probs)
            print("num_larger_than_pivot:", num_larger_than_pivot)
            print("filtered_sorted_probs:", filtered_sorted_probs)
            print("min_larger_than_pivot:", min_larger_than_pivot)
            print("sum_larger_than_pivot:", sum_larger_than_pivot)
            print("sum_larger_than_pivot_exclude_min:", sum_larger_than_pivot_exclude_min)
            print("renom_prob:", renorm_prob)
            print("renorm:", renorm)
            raise

    for i in range(batch_size):
        verify_pivot(p_np[i], final_pivot[i], final_lsum[i], renorm[i])


if __name__ == "__main__":
    tvm.testing.main()
