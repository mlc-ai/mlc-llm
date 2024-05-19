"""Debug protocols in MLC LLM"""

from pydantic import BaseModel


class DebugConfig(BaseModel):
    """The class of debug options."""

    pinned_system_prompt: bool = False
