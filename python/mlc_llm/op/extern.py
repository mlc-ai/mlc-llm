"""Potential externel modules managed by MLC compilation stack.

An externl module could contain one or multiple handcrafted kernels, as long as it is provided as
an object file (`.o`), a C++ source file (`.cc`), or a CUDA source file (`.cu`). It can be
integrated into the system pretty smoothly.

As examples, `flashinfer.py` contains such an example that instructs MLC to compile
"$tvm_home/3rdparty/flashinfer/src/tvm_wrapper.cu" with a specific set of compilation flags and then
link into the generated artifact of MLC LLM. TVM PR #16247
(https://github.com/apache/tvm/pull/16247/) provides more details of using TVM's
`nn.SourceModule` to integrate C++ and CUDA files, and `nn.ObjectModule` to integrate object files.

To conveniently use those externel modules, MLC LLM compilation pipeline manages an extra global
singleton `Store: ExternalModuleStore` to store the configured modules. It is supposed to be enabled
before any compilation happens, and configured during a model's `forward` method is invoked.
"""

import dataclasses
from typing import Optional

from tvm.target import Target


@dataclasses.dataclass
class ExternModuleStore:
    """Global store of external modules enabled during compilation."""

    configured: bool = False
    target: Optional[Target] = None
    flashinfer: bool = False
    faster_transformer: bool = False
    cutlass_group_gemm: bool = False
    cutlass_gemm: bool = False


STORE: ExternModuleStore = ExternModuleStore()
"""Singleton of `ExternModuleStore`."""


def enable(target: Target, flashinfer: bool, faster_transformer: bool, cutlass: bool) -> None:
    """Enable external modules. It should be called before any compilation happens."""
    global STORE  # pylint: disable=global-statement
    cutlass = cutlass and target.kind.name == "cuda" and target.attrs.get("arch", "") == "sm_90a"
    faster_transformer = False
    STORE = ExternModuleStore(
        configured=False,
        target=target,
        flashinfer=flashinfer,
        faster_transformer=faster_transformer,
        cutlass_group_gemm=cutlass,
        cutlass_gemm=cutlass,
    )


def get_store() -> ExternModuleStore:
    """Get the global store of external modules."""
    return STORE


def configure() -> None:
    """Configure external modules with extra parameters. It should be called during a model's
    `forward` method is invoked.

    Parameters
    ----------
    """
    store = get_store()
    if store.configured:
        return
    store.configured = True
    if store.flashinfer or store.faster_transformer:
        assert store.target.kind.name == "cuda"
