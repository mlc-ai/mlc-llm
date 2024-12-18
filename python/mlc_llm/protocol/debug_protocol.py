"""Debug protocols in MLC LLM"""

from typing import Literal, Optional

from pydantic import BaseModel


class DisaggConfig(BaseModel):
    """The class of metadata used in microserving APIs."""

    kind: Optional[Literal["prepare_receive", "remote_send", "start_generation"]] = None
    # "kv_append_metadata" is base64-encoded and is thus a string.
    kv_append_metadata: Optional[str] = None
    # "kv_window_begin" and "kv_window_end" denote the KV interval of interests.
    # "kv_window_end" supports Python style negative indexing.
    # The concrete meaning varies for different special request kind:
    # - For "prepare_receive", the begin is always 0, and "[0:end]" denotes
    # the KV range to prefill on a prefill instance.
    # - For "remote_send", "[begin:end]" means the KV range to compute prefill
    # and send to the decode instance.
    # - For "start_generation", the end is always None, and "[begin:]" denotes
    # the KV range to prefill locally on the decode instance.
    kv_window_begin: Optional[int] = None
    kv_window_end: Optional[int] = None
    # KV data destination group offset
    dst_group_offset: Optional[int] = None


class DebugConfig(BaseModel):
    """The class of debug options.

    These optionals are available to engine
    but won't be available to serving endpoint
    unless an explicit --enable-debug passed
    """

    ignore_eos: bool = False
    pinned_system_prompt: bool = False
    special_request: Optional[Literal["query_engine_metrics"]] = None
    grammar_execution_mode: Literal["constraint", "jump_forward"] = "jump_forward"
    disagg_config: Optional[DisaggConfig] = None

    """Special request indicators

    Special requests are handled by engine differently and do not go
    through the normal engine step flow.

    The results to these requests are returned as field of "usage"
    """
