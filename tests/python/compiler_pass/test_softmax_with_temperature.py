"""The softmax_with_temperature reductions must use every thread of the workgroup."""

import pytest
import tvm_tirx
from tvm import s_tir
from tvm_tirx.ir.module import IRModule

from mlc_llm.compiler_pass.attach_softmax_with_temperature import (
    AttachSoftmaxWithTemperature,
)

pytestmark = [pytest.mark.unittest]


def _thread_tags(func: tvm_tirx.tirx.Function, block_name: str) -> list:
    sch = s_tir.Schedule(func)
    loops = [sch.get(loop) for loop in sch.get_loops(sch.get_sblock(block_name))]
    tags = [
        str(loop.annotations["thread_binding"])
        for loop in loops
        if "thread_binding" in loop.annotations
    ]
    return [tag for tag in tags if tag.startswith("threadIdx")]


@pytest.mark.parametrize("block_name", ["max", "sum_exp"])
def test_softmax_reductions_bind_all_threads(block_name):
    """softmax_with_chunked_sum runs in a TY x TX workgroup. Its max and sum reductions over the
    chunks used to bind threadIdx.x only, so without subgroups the TY rows ran the same reduction
    tree on one shared buffer, a data race. Binding threadIdx.y as well removes it.
    """
    target = tvm_tirx.target.Target("webgpu", host="llvm")
    with target:
        mod = AttachSoftmaxWithTemperature(target)(IRModule())
    softmax = mod["softmax_with_chunked_sum"]
    # The output loop that sets the workgroup shape binds both thread axes.
    assert _thread_tags(softmax, "log_pad") == ["threadIdx.y", "threadIdx.x"]
    assert _thread_tags(softmax, block_name) == ["threadIdx.y", "threadIdx.x"]
