"""Debug protocols in MLC LLM"""

from typing import Literal, Optional

from pydantic import BaseModel


class DebugConfig(BaseModel):
    """The class of debug options.

    These optionals are available to engine
    but won't be available to serving endpoint
    unless an explicit --enable-debug passed
    """

    ignore_eos: bool = False
    pinned_system_prompt: bool = False
    special_request: Optional[Literal["query_engine_metrics"]] = None
    """Special request indicators

    Special requests are handled by engine differently and do not go
    through the normal engine step flow.

    The results to these requests are returned as field of "usage"
    """
