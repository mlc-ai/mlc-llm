"""The tokenizer in MLC LLM Serving."""

from typing import List

import tvm


class Tokenizer:
    """The tokenizer class in MLC LLM."""

    # Class variables for global functions.
    _tokenizer_create_func = tvm.get_global_func("mlc.Tokenizer")
    _encode_func = tvm.get_global_func("mlc.TokenizerEncode")
    _decode_func = tvm.get_global_func("mlc.TokenizerDecode")

    @staticmethod
    def from_path(path: str) -> "Tokenizer":
        """Create a tokenizer from the directory path on disk.

        Parameters
        ----------
        path : str
            The tokenizer directory path on the disk.

        Returns
        -------
        tokenizer : Tokenizer
            The created tokenizer.
        """
        assert Tokenizer._tokenizer_create_func is not None
        return Tokenizer(Tokenizer._tokenizer_create_func(path))

    def __init__(self, tokenizer: tvm.Object) -> None:
        # NOTE: This constructor is designed to be internally used only.
        # Please use `from_path` to create Tokenizer.
        self._tokenizer = tokenizer

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
        assert self._encode_func is not None
        return list(self._encode_func(self._tokenizer, text))

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
        assert self._decode_func is not None
        return self._decode_func(self._tokenizer, tvm.runtime.ShapeTuple(token_ids))
