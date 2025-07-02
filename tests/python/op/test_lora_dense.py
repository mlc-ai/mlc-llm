import numpy as np
import tvm
from tvm.relax.frontend import nn
from mlc_llm.op import lora_dense


def _np_lora_dense(x, w_base, w_delta, alpha):
    return x @ w_base.T + alpha * (x @ w_delta.T)


def test_lora_dense_numerical():
    """Compare Relax lora_dense vs NumPy reference on CPU."""

    rng = np.random.default_rng(0)
    batch, in_feat, out_feat = 2, 4, 3
    x_np = rng.standard_normal((batch, in_feat), dtype="float32")
    w_base_np = rng.standard_normal((out_feat, in_feat), dtype="float32")
    w_delta_np = rng.standard_normal((out_feat, in_feat), dtype="float32") * 0.1
    alpha = 0.5

    x = nn.const(x_np)
    w_base = nn.const(w_base_np)
    w_delta = nn.const(w_delta_np)

    y = lora_dense(x, w_base, w_delta, alpha)
    mod = tvm.IRModule.from_expr(y)

    target = tvm.target.Target("llvm")
    ex = tvm.relax.build(mod, target)
    vm = tvm.relax.VirtualMachine(ex, tvm.cpu())
    res = vm["main"]()

    np_expected = _np_lora_dense(x_np, w_base_np, w_delta_np, alpha)
    np.testing.assert_allclose(res.numpy(), np_expected, rtol=1e-5, atol=1e-5) 