"""MLC Chat python package.

MLC Chat is the app runtime of MLC LLM.
"""

from . import libinfo

# The imports below map the native libraries of MLC LLM into the process. On
# macOS, a library with a broken code signature is not mapped at all, the
# kernel kills the process instead. Checking the signatures first turns that
# silent crash into an error that can be acted upon.
libinfo.check_native_libraries()

from tvm import register_global_func

from . import protocol, serve
from .libinfo import __version__
from .serve import AsyncMLCEngine, MLCEngine


@register_global_func("runtime.disco.create_socket_session_local_workers", override=True)
def _create_socket_session_local_workers(num_workers):
    """Create the local session for each distributed node over socket session."""
    from tvm.runtime.disco import (
        ProcessSession,
    )

    return ProcessSession(num_workers, num_groups=1, entrypoint="mlc_llm.cli.worker")
