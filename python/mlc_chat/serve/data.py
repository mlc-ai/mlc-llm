"""Classes denoting multi-modality data used in MLC LLM serving"""

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


@tvm._ffi.register_object("mlc.serve.RequestStreamOutput")  # pylint: disable=protected-access
class RequestStreamOutput(Object):
    """The generated delta request output that is streamed back
    through callback stream function.
    It contains four fields (in order):

    request_id : str
        The id of the request that the function is invoked for.

    delta_tokens : List[int]
        The new generated tokens since the last callback invocation
        for the input request.

    delta_logprob_json_strs : Optional[List[str]]
        The logprobs JSON strings of the new generated tokens
        since last invocation.

    finish_reason : Optional[str]
        The finish reason of the request when it is finished,
        of None if the request has not finished yet.

    Note
    ----
    We do not provide constructor, since in practice only C++ side
    instantiates this class.
    """

    def unpack(self) -> Tuple[str, List[int], Optional[List[str]], Optional[str]]:
        """Return the fields of the delta output in a tuple.

        Returns
        -------
        request_id : str
            The id of the request that the function is invoked for.

        delta_tokens : List[int]
            The new generated tokens since the last callback invocation
            for the input request.

        delta_logprob_json_strs : Optional[List[str]]
            The logprobs JSON strings of the new generated tokens
            since last invocation.

        finish_reason : Optional[str]
            The finish reason of the request when it is finished,
            of None if the request has not finished yet.
        """
        fields = _ffi_api.RequestStreamOutputUnpack(self)  # type: ignore  # pylint: disable=no-member
        return (
            str(fields[0]),
            list(fields[1]),
            (
                [str(logprob_json_str) for logprob_json_str in fields[2]]
                if fields[2] is not None
                else None
            ),
            str(fields[3]) if fields[3] is not None else None,
        )
