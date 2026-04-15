import tvm
from tvm import relax

from mlc_llm.compiler_pass.attach_sampler import AttachGPUSamplingFunc


def _base_module():
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((1, 1), "float32"))

    with bb.function("main", [x]):
        with bb.dataflow():
            output = bb.emit_output(x)
        bb.emit_func_output(output)

    return bb.finalize()


def test_attach_sampler_adds_webgpu_functions():
    seq = tvm.transform.Sequential(
        [AttachGPUSamplingFunc(tvm.target.Target("webgpu"), {"batch_size": 4})]
    )
    mod = seq(_base_module())
    global_vars = {gv.name_hint for gv in mod.get_global_vars()}

    assert "argsort_probs" in global_vars
    assert "sample_with_top_p" in global_vars
    assert "multinomial_from_uniform" not in global_vars


def test_attach_sampler_skips_non_gpu_targets():
    seq = tvm.transform.Sequential(
        [AttachGPUSamplingFunc(tvm.target.Target("llvm"), {"batch_size": 4})]
    )
    mod = seq(_base_module())
    global_vars = {gv.name_hint for gv in mod.get_global_vars()}

    assert "argsort_probs" not in global_vars
    assert "sample_with_top_p" not in global_vars
