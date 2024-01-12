"""Classes handling the grammar guided generation of MLC LLM serving"""
import tvm._ffi
from tvm.runtime import Object

from . import _ffi_api


@tvm._ffi.register_object("mlc.serve.BNFGrammar")  # pylint: disable=protected-access
class BNFGrammar(Object):
    """This class stores the abstract syntax tree (AST) of the Backus-Naur Form (BNF) grammar and
    provides utilities to parse and print the AST. User should provide a BNF/EBNF (Extended
    Backus-Naur Form) grammar, and use from_ebnf_string to parse and simplify the grammar into an
    AST of BNF grammar.
    """

    @staticmethod
    def from_ebnf_string(ebnf_string: str) -> "BNFGrammar":
        r"""Parse a BNF grammar from a string in BNF/EBNF format.

        This method accepts the EBNF notation from the W3C XML Specification
        (https://www.w3.org/TR/xml/#sec-notation), which is a popular standard, with the following
        changes:
        - Using # as comment mark instead of /**/
        - Using C-style unicode escape sequence \u01AB, \U000001AB, \xAB instead of #x0123
        - Do not support A-B (match A and not match B) yet

        See tests/python/serve/json.ebnf for an example.

        Parameters
        ----------
        ebnf_string : str
            The grammar string.

        Returns
        -------
        grammar : BNFGrammar
            The parsed BNF grammar.
        """
        return _ffi_api.BNFGrammarFromEBNFString(  # type: ignore  # pylint: disable=no-member
            ebnf_string
        )

    def to_string(self) -> str:
        """Print the BNF grammar to a string, in standard BNF format.

        Returns
        -------
        grammar_string : str
            The BNF grammar string.
        """
        return str(_ffi_api.BNFGrammarToString(self))  # type: ignore  # pylint: disable=no-member

    @staticmethod
    def from_json(json_string: str) -> "BNFGrammar":
        """Load a BNF grammar from the raw representation of the AST in JSON format.

        Parameters
        ----------
        json_string : str
            The JSON string.

        Returns
        -------
        grammar : BNFGrammar
            The loaded BNF grammar.
        """
        return _ffi_api.BNFGrammarFromJSON(json_string)  # type: ignore  # pylint: disable=no-member

    def to_json(self, prettify: bool = True) -> str:
        """Serialize the AST. Dump the raw representation of the AST to a JSON file.

        Parameters
        ----------
        prettify : bool
            Whether to format the JSON string. If False, all whitespaces will be removed.

        Returns
        -------
        json_string : str
            The JSON string.
        """
        return str(
            _ffi_api.BNFGrammarToJSON(self, prettify)  # type: ignore  # pylint: disable=no-member
        )
