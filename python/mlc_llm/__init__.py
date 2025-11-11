"""MLC Chat python package.

MLC Chat is the app runtime of MLC LLM.
"""

import logging
import tvm

if hasattr(tvm, "register_func"):
    register_func = tvm.register_func  # type: ignore[attr-defined]
else:  # pragma: no cover
    from tvm_ffi.registry import register_global_func as register_func  # type: ignore

    setattr(tvm, "register_func", register_func)

AsyncMLCEngine = None  # type: ignore
MLCEngine = None  # type: ignore

try:
    from . import protocol as protocol  # type: ignore
except RuntimeError as err:  # pragma: no cover
    logging.getLogger(__name__).debug("MLC-LLM protocol unavailable: %s", err)
    protocol = None  # type: ignore

try:
    from . import serve as serve  # type: ignore
except RuntimeError as err:  # pragma: no cover
    logging.getLogger(__name__).debug("MLC-LLM serve unavailable: %s", err)
    serve = None  # type: ignore
else:
    AsyncMLCEngine = serve.AsyncMLCEngine
    MLCEngine = serve.MLCEngine

from .libinfo import __version__


@register_func("runtime.disco.create_socket_session_local_workers", override=True)
def _create_socket_session_local_workers(num_workers):
    """Create the local session for each distributed node over socket session."""
    from tvm.runtime.disco import (  # pylint: disable=import-outside-toplevel
        ProcessSession,
    )

    return ProcessSession(num_workers, num_groups=1, entrypoint="mlc_llm.cli.worker")
