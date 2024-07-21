"""MLC Chat python package.

MLC Chat is the app runtime of MLC LLM.
"""

from tvm import register_func

from . import protocol, serve
from .libinfo import __version__
from .serve import AsyncMLCEngine, MLCEngine


@register_func("runtime.disco.create_socket_session_local_workers", override=True)
def _create_socket_session_local_workers(num_workers):
    """Create the local session for each distributed node over socket session."""
    from tvm.runtime.disco import (  # pylint: disable=import-outside-toplevel
        ProcessSession,
    )

    return ProcessSession(num_workers, num_groups=1, entrypoint="mlc_llm.cli.worker")
