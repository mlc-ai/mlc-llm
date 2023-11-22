"""Classes denoting multi-modality data used in MLC LLM serving"""
from typing import List

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
