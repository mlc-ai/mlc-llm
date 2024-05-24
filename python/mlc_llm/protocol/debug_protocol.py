"""Debug protocols in MLC LLM"""

from pydantic import BaseModel


class DebugConfig(BaseModel):
    """The class of debug options.

    These optionals are available to engine
    but won't be available to serving endpoint
    unless an explicit --enable-debug-config passed
    """

    ignore_eos: bool = False
    pinned_system_prompt: bool = False
