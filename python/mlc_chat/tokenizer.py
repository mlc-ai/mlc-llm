"""The tokenizer and related tools in MLC LLM.
This tokenizer essentially wraps and binds the HuggingFace tokenizer
library and sentencepiece.
Reference: https://github.com/mlc-ai/tokenizers-cpp
"""
from typing import List

import tvm
import tvm._ffi
from tvm.runtime import Object

from . import _ffi_api


@tvm._ffi.register_object("mlc.Tokenizer")  # pylint: disable=protected-access
class Tokenizer(Object):
    """The tokenizer class in MLC LLM."""

    def __init__(self, tokenizer_path: str) -> None:
        """Create the tokenizer from tokenizer directory path."""
        self.__init_handle_by_constructor__(
            _ffi_api.Tokenizer, tokenizer_path  # type: ignore  # pylint: disable=no-member
        )

    def encode(self, text: str) -> List[int]:
        """Encode text into ids.

        Parameters
        ----------
        text : str
            The text string to encode.

        Returns
        -------
        token_ids : List[int]
            The list of encoded token ids.
        """
        return list(_ffi_api.TokenizerEncode(self, text))  # type: ignore  # pylint: disable=no-member

    def decode(self, token_ids: List[int]) -> str:
        """Decode token ids into text.

        Parameters
        ----------
        token_ids : List[int]
            The token ids to decode to string.

        Returns
        -------
        text : str
            The decoded text string.
        """
        return _ffi_api.TokenizerDecode(  # type: ignore  # pylint: disable=no-member
            self, tvm.runtime.ShapeTuple(token_ids)
        )
