"""MLC Chat python package.

MLC Chat is the app runtime of MLC LLM.
"""

import os

from tvm import register_global_func

from . import protocol
from .libinfo import __version__

# Compilation and weight conversion do not require the native serving library.
if os.environ.get("SKIP_LOADING_MLCLLM_SO", "0") == "0":
    from . import serve
    from .serve import AsyncMLCEngine, MLCEngine


@register_global_func("runtime.disco.create_socket_session_local_workers", override=True)
def _create_socket_session_local_workers(num_workers):
    """Create the local session for each distributed node over socket session."""
    from tvm.runtime.disco import (
        ProcessSession,
    )

    return ProcessSession(num_workers, num_groups=1, entrypoint="mlc_llm.cli.worker")
