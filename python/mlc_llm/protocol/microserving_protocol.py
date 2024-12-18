"""Protocols in MLC LLM for MicroServing."""

from pydantic import BaseModel

from mlc_llm.protocol.openai_api_protocol import CompletionRequest


class PrepRecvRequest(CompletionRequest):
    """The extra request body for prep_recv request in MicroServing.

    Attributes
    ----------
    kv_window_end : int
        [0, kv_window_end] denotes the KV range of the prompt to prefill on
        a prefill instance.
        The entries of this KV range will be allocated on the decode instance.
    """

    kv_window_end: int


class PrepRecvResponse(BaseModel):
    """The response body for prep_recv request in MicroServing.

    Attributes
    ----------
    prompt_length : int
        The length of the request prompt in tokens.

    prefix_matched_length : int
        The matched common prefix length on the decode instance when
        prefix cache is enabled, or 0 if there is no prefix cache.

    kv_append_metadata : str
        The metadata of the KV range on the destination decode instance.
    """

    prompt_length: int
    prefix_matched_length: int
    kv_append_metadata: str


class RemoteSendRequest(CompletionRequest):
    """The extra request body for remote_send request in MicroServing.

    Attributes
    ----------
    kv_window_begin : int
        Denote the start of the KV range to prefill.

    kv_window_end : int
        Denote the end of the KV range to prefill.

    kv_append_metadata : str
        The metadata of the KV range on the destination decode instance.

    dst_group_offset : int
        The node group offset of the destination decode instance.
    """

    kv_window_begin: int
    kv_window_end: int
    kv_append_metadata: str
    dst_group_offset: int


class StartGenerateRequest(CompletionRequest):
    """The extra request body for start_generate request in MicroServing.

    Attributes
    ----------
    kv_window_begin : int
        Denote the start of the KV range to prefill on the decode instance.
    """

    kv_window_begin: int
