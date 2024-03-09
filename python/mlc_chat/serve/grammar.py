"""Classes handling the grammar guided generation of MLC LLM serving"""
from typing import List, Union

import tvm._ffi
from tvm.runtime import Object

from ..tokenizer import Tokenizer
from . import _ffi_api


@tvm._ffi.register_object("mlc.serve.BNFGrammar")  # pylint: disable=protected-access
class BNFGrammar(Object):
    """This class stores the abstract syntax tree (AST) of the Backus-Naur Form (BNF) grammar and
    provides utilities to parse and print the AST. User should provide a BNF/EBNF (Extended
    Backus-Naur Form) grammar, and use from_ebnf_string to parse and simplify the grammar into an
    AST of BNF grammar.
    """

    @staticmethod
    def from_ebnf_string(
        ebnf_string: str, normalize: bool = True, simplify: bool = True
    ) -> "BNFGrammar":
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

        normalize : bool
            Whether to normalize the grammar. Default: true. Only set to false for the purpose of
            testing.

            In The normalized form of a BNF grammar, every rule is in the form:
            `rule_name ::= ("" | (element1_1 element1_2 ...) | (element2_1 element2_2 ...) | ...)`.

            I.e. a list of choices, each choice is a sequence of elements. Elements can be a
            character class or a rule reference. And if the rule can be empty, the first choice
            will be an empty string.

        simplify : bool
            Whether to simplify the grammar to make matching more efficient. Default: true. Not
            implemented yet.

        Returns
        -------
        grammar : BNFGrammar
            The parsed BNF grammar.
        """
        return _ffi_api.BNFGrammarFromEBNFString(  # type: ignore  # pylint: disable=no-member
            ebnf_string, normalize, simplify
        )

    def to_string(self) -> str:
        """Print the BNF grammar to a string, in standard BNF format.

        Returns
        -------
        grammar_string : str
            The BNF grammar string.
        """
        return str(_ffi_api.BNFGrammarToString(self))  # type: ignore  # pylint: disable=no-member

    def __str__(self) -> str:
        return self.to_string()

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

    @staticmethod
    def get_grammar_of_json() -> "BNFGrammar":
        """Get the grammar of standard JSON.

        Returns
        -------
        grammar : BNFGrammar
            The JSON grammar.
        """
        return _ffi_api.BNFGrammarGetGrammarOfJSON()  # type: ignore  # pylint: disable=no-member


@tvm._ffi.register_object("mlc.serve.GrammarStateMatcher")  # pylint: disable=protected-access
class GrammarStateMatcher(Object):
    """A stateful matcher to match tokens to the specified BNF grammar. This class is the core logic
    of the grammar-guided generation.

    This class implements the non-deterministic pushdown automaton (NPDA) matching algorithm to
    match characters to a BNF grammar. It keep track of the current state of the matching process by
    maintaining several stacks internally as possible paths in the NPDA. It also supports
    backtracking.

    It is particularly capable of finding the set of tokens that are acceptable for the next step
    and storing them in a bitmask. This aids in grammar-guided generation.

    Parameters
    ----------
    grammar : BNFGrammar
        The BNF grammar to match.

    tokenizer : Union[None, Tokenizer, List[str]]
        The tokenizer to use, or the list of tokens.

        (For debug purpose) If None, the matcher will use an empty token set, and can only accept
        and match characters. Default: None.

    max_rollback_steps : int
        The maximum number of steps to rollback when backtracking. Default: 0.
    """

    def __init__(
        self,
        grammar: BNFGrammar,
        tokenizer: Union[None, Tokenizer, List[str]] = None,
        max_rollback_steps: int = 0,
    ):
        if isinstance(tokenizer, list):
            self.__init_handle_by_constructor__(
                _ffi_api.GrammarStateMatcherFromTokenTable,  # type: ignore  # pylint: disable=no-member
                grammar,
                *tokenizer,
                max_rollback_steps,
            )
        else:
            self.__init_handle_by_constructor__(
                _ffi_api.GrammarStateMatcherFromTokenizer,  # type: ignore  # pylint: disable=no-member
                grammar,
                tokenizer,
                max_rollback_steps,
            )

    def accept_token(self, token_id: int) -> bool:
        """Accept one token and update the state of the matcher.

        Parameters
        ----------
        token_id : int
            The id of the token to accept.

        Returns
        -------
        accepted : bool
            Whether the token is accepted.

        Note
        ----
        Termination state.

        When the end of the main rule is reached, the matcher can only accept the stop token.
        The matcher is terminated after accepting the stop token, i.e. no accept_token or
        find_next_rejected_tokens operations can be performed. The termination state can be canceled
        using Rollback().
        """
        return _ffi_api.GrammarStateMatcherAcceptToken(self, token_id)  # type: ignore  # pylint: disable=no-member

    def find_next_rejected_tokens(self) -> List[int]:
        """Find the ids of the rejected tokens for the next step.

        Returns
        -------
        rejected_token_ids : List[int]
            A list of rejected token ids.
        """

        return _ffi_api.GrammarStateMatcherFindNextRejectedTokens(self)  # type: ignore  # pylint: disable=no-member

    def rollback(self, num_tokens: int) -> None:
        """Rollback the matcher to a previous state.

        Parameters
        ----------
        num_tokens : int
            The number of tokens to rollback. It cannot exceed the current number of steps, nor can
            it exceed the specified maximum number of rollback steps.
        """
        _ffi_api.GrammarStateMatcherRollback(self, num_tokens)  # type: ignore  # pylint: disable=no-member

    def max_rollback_steps(self) -> int:
        """Get the maximum number of rollback steps allowed.

        Returns
        -------
        max_rollback_steps : int
            The maximum number of rollback steps.
        """
        return _ffi_api.GrammarStateMatcherMaxRollbackSteps(self)  # type: ignore  # pylint: disable=no-member

    def reset_state(self) -> None:
        """Reset the matcher to the initial state."""
        _ffi_api.GrammarStateMatcherResetState(self)  # type: ignore  # pylint: disable=no-member

    def is_terminated(self) -> bool:
        """Check if the matcher has accepted the stop token and terminated. See also
        GrammarStateMatcher.accept_token.

        Returns
        -------
        terminated : bool
            Whether the matcher has terminated.
        """
        return _ffi_api.GrammarStateMatcherIsTerminated(self)  # type: ignore  # pylint: disable=no-member

    def debug_accept_char(self, codepoint: int) -> bool:
        """Accept one unicode codepoint to the current state. For test purposes.

        Parameters
        ----------
        codepoint : int
            The unicode codepoint of the character to be accepted.
        """
        return _ffi_api.GrammarStateMatcherDebugAcceptCodepoint(  # type: ignore  # pylint: disable=no-member
            self, codepoint
        )

    def debug_match_complete_string(self, string: str) -> bool:
        """Check if the matcher can accept the complete string, and then reach the end of the
        grammar. For test purposes.

        Parameters
        ----------
        string : str
            The string to be matched.
        """
        return _ffi_api.GrammarStateMatcherDebugMatchCompleteString(self, string)  # type: ignore  # pylint: disable=no-member
