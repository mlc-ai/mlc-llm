"""Classes denoting multi-modality data used in MLC LLM serving"""
from typing import List

import tvm._ffi
from tvm.runtime import Object
from tvm.runtime.container import getitem_helper

from . import _ffi_api


@tvm._ffi.register_object("mlc.serve.Data")
class Data(Object):
    """The base class of multi-modality data (text, tokens, embedding, etc)."""

    def __init__(self):
        pass


@tvm._ffi.register_object("mlc.serve.TextData")
class TextData(Data):
    """The class of text data, containing a text string.

    Parameters
    ----------
    text : str
        The text string.
    """

    def __init__(self, text: str):
        self.__init_handle_by_constructor__(_ffi_api.TextData, text)

    @property
    def text(self) -> str:
        return str(_ffi_api.TextDataGetTextString(self))

    def __str__(self) -> str:
        return self.text


@tvm._ffi.register_object("mlc.serve.TokenData")
class TokenData(Data):
    """The class of token data, containing a list of token ids.

    Parameters
    ----------
    token_ids : List[int]
        The list of token ids.
    """

    def __init__(self, token_ids: List[int]):
        self.__init_handle_by_constructor__(_ffi_api.TokenData, *token_ids)

    def __len__(self) -> int:
        return _ffi_api.TokenDataGetLength(self)

    def __getitem__(self, idx) -> int:
        return getitem_helper(self, _ffi_api.TokenDataGetElem, len(self), idx)
