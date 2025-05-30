"""The tokenizer and related tools in MLC LLM.
This tokenizer essentially wraps and binds the HuggingFace tokenizer
library and sentencepiece.
Reference: https://github.com/mlc-ai/tokenizers-cpp
"""

import json
from dataclasses import asdict, dataclass
from typing import List, Literal

import tvm
import tvm.ffi
from tvm.runtime import Object

from . import _ffi_api


@dataclass
class TokenizerInfo:  # pylint: disable=too-many-instance-attributes
    """Useful information of the tokenizer during generation.

    Attributes
    ----------
    token_postproc_method : Literal["byte_fallback", "byte_level"]
        The method to post-process the tokens to their original strings.
        Possible values (each refers to a kind of tokenizer):
        - "byte_fallback": The same as the byte-fallback BPE tokenizer, including LLaMA-2,
            Mixtral-7b, etc. E.g. "▁of" -> " of", "<0x1B>" -> "\x1b".
            This method:
            1) Transform tokens like <0x1B> to hex char byte 1B. (so-called byte-fallback)
            2) Replace \\u2581 "▁" with space.
        - "byte_level": The same as the byte-level BPE tokenizer, including LLaMA-3, GPT-2,
            Phi-2, etc. E.g. "Ġin" -> " in", "ě" -> "\x1b"
            This method inverses the bytes-to-unicode transformation in the encoding process in
            https://github.com/huggingface/transformers/blob/87be06ca77166e6a6215eee5a990ab9f07238a18/src/transformers/models/gpt2/tokenization_gpt2.py#L38-L59

    prepend_space_in_encode : bool
        Whether to prepend a space during encoding.

    strip_space_in_decode : bool
        Whether to strip the first space during decoding.
    """

    token_postproc_method: Literal["byte_fallback", "byte_level"] = "byte_fallback"
    prepend_space_in_encode: bool = False
    strip_space_in_decode: bool = False

    def asjson(self) -> str:
        """Return the config in string of JSON format."""
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(json_str: str) -> "TokenizerInfo":
        """Construct a config from JSON string."""
        return TokenizerInfo(**json.loads(json_str))


@tvm.ffi.register_object("mlc.Tokenizer")  # pylint: disable=protected-access
class Tokenizer(Object):
    """The tokenizer class in MLC LLM."""

    def __init__(self, tokenizer_path: str) -> None:  # pylint: disable=super-init-not-called
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

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode a batch of texts into ids.

        Parameters
        ----------
        texts : List[str]
            The list of text strings to encode.

        Returns
        -------
        token_ids : List[List[int]]
            The list of list of encoded token ids.
        """
        return list(_ffi_api.TokenizerEncodeBatch(self, texts))  # type: ignore  # pylint: disable=no-member

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

    @staticmethod
    def detect_tokenizer_info(tokenizer_path: str) -> TokenizerInfo:
        """Detect the tokenizer info from the given path of the tokenizer.

        Parameters
        ----------
        tokenizer_path : str
            The tokenizer directory path.

        Returns
        -------
        tokenizer_info : str
            The detected tokenizer info in JSON string.
        """
        return TokenizerInfo.from_json(_ffi_api.DetectTokenizerInfo(tokenizer_path))  # type: ignore  # pylint: disable=no-member
