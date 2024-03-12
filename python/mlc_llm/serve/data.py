"""Classes denoting multi-modality data used in MLC LLM serving"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import tvm._ffi
from tvm.runtime import Object

from . import _ffi_api


@tvm._ffi.register_object("mlc.serve.Data")  # pylint: disable=protected-access
class Data(Object):
    """The base class of multi-modality data (text, tokens, embedding, etc)."""

    def __init__(self):
        pass


@tvm._ffi.register_object("mlc.serve.TextData")  # pylint: disable=protected-access
class TextData(Data):
    """The class of text data, containing a text string.

    Parameters
    ----------
    text : str
        The text string.
    """

    def __init__(self, text: str):
        self.__init_handle_by_constructor__(_ffi_api.TextData, text)  # type: ignore  # pylint: disable=no-member

    @property
    def text(self) -> str:
        """The text data in `str`."""
        return str(_ffi_api.TextDataGetTextString(self))  # type: ignore  # pylint: disable=no-member

    def __str__(self) -> str:
        return self.text


@tvm._ffi.register_object("mlc.serve.TokenData")  # type: ignore  # pylint: disable=protected-access
class TokenData(Data):
    """The class of token data, containing a list of token ids.

    Parameters
    ----------
    token_ids : List[int]
        The list of token ids.
    """

    def __init__(self, token_ids: List[int]):
        self.__init_handle_by_constructor__(_ffi_api.TokenData, *token_ids)  # type: ignore  # pylint: disable=no-member

    @property
    def token_ids(self) -> List[int]:
        """Return the token ids of the TokenData."""
        return list(_ffi_api.TokenDataGetTokenIds(self))  # type: ignore  # pylint: disable=no-member


@dataclass
class SingleRequestStreamOutput:
    """The request stream output of a single request.

    Attributes
    ----------
    delta_token_ids : List[int]
        The new generated tokens since the last callback invocation
        for the input request.

    delta_logprob_json_strs : Optional[List[str]]
        The logprobs JSON strings of the new generated tokens
        since last invocation.

    finish_reason : Optional[str]
        The finish reason of the request when it is finished,
        of None if the request has not finished yet.
    """

    delta_token_ids: List[int]
    delta_logprob_json_strs: Optional[List[str]]
    finish_reason: Optional[str]


@tvm._ffi.register_object("mlc.serve.RequestStreamOutput")  # pylint: disable=protected-access
class RequestStreamOutput(Object):
    """The generated delta request output that is streamed back
    through callback stream function.
    It contains four fields (in order):

    request_id : str
        The id of the request that the function is invoked for.

    stream_outputs : List[SingleRequestStreamOutput]
        The output instances, one for a request.

    Note
    ----
    We do not provide constructor, since in practice only C++ side
    instantiates this class.
    """

    def unpack(self) -> Tuple[str, List[SingleRequestStreamOutput]]:
        """Return the fields of the delta output in a tuple.

        Returns
        -------
        request_id : str
            The id of the request that the function is invoked for.

        stream_outputs : List[SingleRequestStreamOutput]
            The output instances, one for a request.
        """
        fields = _ffi_api.RequestStreamOutputUnpack(self)  # type: ignore  # pylint: disable=no-member
        request_id = str(fields[0])
        stream_outputs = []
        for i, (delta_token_ids, finish_reason) in enumerate(zip(fields[1], fields[3])):
            delta_logprob_json_strs = (
                [str(logprob_json_str) for logprob_json_str in fields[2][i]]
                if fields[2] is not None
                else None
            )
            stream_outputs.append(
                SingleRequestStreamOutput(
                    delta_token_ids=list(delta_token_ids),
                    delta_logprob_json_strs=delta_logprob_json_strs,
                    finish_reason=str(finish_reason) if finish_reason is not None else None,
                )
            )
        return request_id, stream_outputs
