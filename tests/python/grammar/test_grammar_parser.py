# pylint: disable=missing-module-docstring,missing-function-docstring
import json
import os

import pytest
import tvm.testing
from tvm import TVMError

from mlc_llm.grammar import BNFGrammar


def test_bnf_simple():
    before = """main ::= b c
b ::= "b"
c ::= "c"
"""
    expected = """main ::= ((b c))
b ::= (("b"))
c ::= (("c"))
"""
    bnf_grammar = BNFGrammar.from_ebnf_string(before, "main")
    after = bnf_grammar.to_string()
    print(after)
    print(expected)
    assert after == expected


def test_ebnf():
    before = """main ::= b c | b main
b ::= "ab"*
c ::= [acep-z]+
d ::= "d"?
"""
    expected = """main ::= ((b c) | (b main))
b ::= ((b_1))
c ::= ((c_1))
d ::= ((d_1))
b_1 ::= ("" | ("ab" b_1))
c_1 ::= (([acep-z] c_1) | ([acep-z]))
d_1 ::= ("" | ("d"))
"""
    bnf_grammar = BNFGrammar.from_ebnf_string(before, "main")
    after = bnf_grammar.to_string()
    assert after == expected


def test_star_quantifier():
    before = """main ::= b c d
b ::= [b]*
c ::= "b"*
d ::= ([b] [c] [d] | ([p] [q]))*
e ::= [e]* [f]* | [g]*
"""
    expected = """main ::= ((b c d))
b ::= (([b]*))
c ::= ((c_1))
d ::= ((d_1))
e ::= (([e]* [f]*) | ([g]*))
c_1 ::= ("" | ("b" c_1))
d_1 ::= ("" | (d_1_choice d_1))
d_1_choice ::= (("bcd") | ("pq"))
"""
    bnf_grammar = BNFGrammar.from_ebnf_string(before, "main")
    after = bnf_grammar.to_string()
    assert after == expected


def test_lookahead_assertion():
    before = """main ::= ((b c d))
b ::= (("abc" [a-z])) (=("abc"))
c ::= (("a") | ("b")) (=([a-z] "b"))
d ::= (("ac") | ("b" d_choice)) (=("abc"))
d_choice ::= (("e") | ("d"))
"""
    expected = """main ::= ((b c d))
b ::= (("abc" [a-z])) (=("abc"))
c ::= (("a") | ("b")) (=([a-z] "b"))
d ::= (("ac") | ("b" d_choice)) (=("abc"))
d_choice ::= (("e") | ("d"))
"""
    bnf_grammar = BNFGrammar.from_ebnf_string(before, "main")
    after = bnf_grammar.to_string()
    assert after == expected


def test_char():
    before = r"""main ::= [a-z] [A-z] "\u0234" "\U00000345\xff" [-A-Z] [--] [^a] rest
rest ::= [a-zA-Z0-9-] [\u0234-\U00000345] [测-试] [\--\]]  rest1
rest1 ::= "\?\"\'测试あc" "👀" "" [a-a] [b-b]
"""
    expected = r"""main ::= (([a-z] [A-z] "\u0234\u0345\xff" [\-A-Z] [\-\-] [^a] rest))
rest ::= (([a-zA-Z0-9\-] [\u0234-\u0345] [\u6d4b-\u8bd5] [\--\]] rest1))
rest1 ::= (("\?\"\'\u6d4b\u8bd5\u3042c\U0001f440ab"))
"""
    # Disable unwrap_nesting_rules to expose the result before unwrapping.
    bnf_grammar = BNFGrammar.from_ebnf_string(before, "main")
    after = bnf_grammar.to_string()
    assert after == expected


def test_space():
    before = """

main::="a"  "b" ("c""d"
"e") |

"f" | "g"
"""
    expected = """main ::= (("abcde") | ("f") | ("g"))
"""
    bnf_grammar = BNFGrammar.from_ebnf_string(before, "main")
    after = bnf_grammar.to_string()
    assert after == expected


def test_nest():
    before = """main::= "a" ("b" | "c" "d") | (("e" "f"))
"""
    expected = """main ::= (("a" main_choice) | ("ef"))
main_choice ::= (("b") | ("cd"))
"""
    bnf_grammar = BNFGrammar.from_ebnf_string(before, "main")
    after = bnf_grammar.to_string()
    assert after == expected


def test_flatten():
    before = """main ::= or_test sequence_test nested_test empty_test
or_test ::= ([a] | "b") | "de" | "" | or_test | [^a-z]
sequence_test ::= [a] "a" ("b" ("c" | "d")) ("d" "e") sequence_test ""
nested_test ::= ("a" ("b" ("c" "d"))) | ("a" | ("b" | "c")) | nested_rest
nested_rest ::= ("a" | ("b" "c" | ("d" | "e" "f"))) | ((("g")))
empty_test ::= "d" | (("" | "" "") "" | "a" "") | ("" ("" | "")) "" ""
"""
    expected = """main ::= ((or_test sequence_test nested_test empty_test))
or_test ::= ("" | ("a") | ("b") | ("de") | (or_test) | ([^a-z]))
sequence_test ::= (("aab" sequence_test_choice "de" sequence_test))
nested_test ::= (("abcd") | ("a") | ("b") | ("c") | (nested_rest))
nested_rest ::= (("a") | ("bc") | ("d") | ("ef") | ("g"))
empty_test ::= ("" | ("d") | ("a"))
sequence_test_choice ::= (("c") | ("d"))
"""
    bnf_grammar = BNFGrammar.from_ebnf_string(before, "main")
    after = bnf_grammar.to_string()
    print(after)
    assert after == expected


def test_json():
    # Adopted from https://www.crockford.com/mckeeman.html. Not optimized
    before = r"""main ::= element
value ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws "}" | "{" members "}"
members ::= member | member "," members
member ::= ws string ws ":" element
array ::= "[" ws "]" | "[" elements "]"
elements ::= element | element "," elements
element ::= ws value ws
string ::= "\"" characters "\""
characters ::= "" | character characters
character ::= [^"\\] | "\\" escape
escape ::= "\"" | "\\" | "/" | "b" | "f" | "n" | "r" | "t" | "u" hex hex hex hex
hex ::= [A-Fa-f0-9]
number ::= integer fraction exponent
integer ::= digit | onenine digits | "-" digit | "-" onenine digits
digits ::= digit | digit digits
digit ::= [0-9]
onenine ::= [1-9]
fraction ::= "" | "." digits
exponent ::= "" | ("e" | "E") ("" | "+" | "-") digits
ws ::= "" | "\u0020" ws | "\u000A" ws | "\u000D" ws | "\u0009" ws
"""

    expected = r"""main ::= ((element))
value ::= ((object) | (array) | (string) | (number) | ("true") | ("false") | ("null"))
object ::= (("{" ws "}") | ("{" members "}"))
members ::= ((member) | (member "," members))
member ::= ((ws string ws ":" element))
array ::= (("[" ws "]") | ("[" elements "]"))
elements ::= ((element) | (element "," elements))
element ::= ((ws value ws))
string ::= (("\"" characters "\""))
characters ::= ("" | (character characters))
character ::= (([^\"\\]) | ("\\" escape))
escape ::= (("\"") | ("\\") | ("/") | ("b") | ("f") | ("n") | ("r") | ("t") | ("u" hex hex hex hex))
hex ::= (([A-Fa-f0-9]))
number ::= ((integer fraction exponent))
integer ::= ((digit) | (onenine digits) | ("-" digit) | ("-" onenine digits))
digits ::= ((digit) | (digit digits))
digit ::= (([0-9]))
onenine ::= (([1-9]))
fraction ::= ("" | ("." digits))
exponent ::= ("" | (exponent_choice exponent_choice_1 digits))
ws ::= ("" | (" " ws) | ("\n" ws) | ("\r" ws) | ("\t" ws))
exponent_choice ::= (("e") | ("E"))
exponent_choice_1 ::= ("" | ("+") | ("-"))
"""

    bnf_grammar = BNFGrammar.from_ebnf_string(before, "main")
    after = bnf_grammar.to_string()
    assert after == expected


def test_to_string_roundtrip():
    """Checks the printed result can be parsed, and the parsing-printing process is idempotent."""

    before = r"""main ::= ((b c) | (b main))
b ::= ((b_1 d))
c ::= ((c_1))
d ::= ((d_1))
b_1 ::= ("" | ("b" b_1))
c_1 ::= ((c_2 c_1) | (c_2)) (=("abc" [a-z]))
c_2 ::= (([acep-z]))
d_1 ::= ("" | ("d"))
"""
    bnf_grammar_1 = BNFGrammar.from_ebnf_string(before, "main")
    output_string_1 = bnf_grammar_1.to_string()
    bnf_grammar_2 = BNFGrammar.from_ebnf_string(output_string_1, "main")
    output_string_2 = bnf_grammar_2.to_string()
    assert before == output_string_1
    assert output_string_1 == output_string_2


def test_error():
    with pytest.raises(
        TVMError, match='TVMError: EBNF parse error at line 1, column 11: Rule "a" is not defined'
    ):
        BNFGrammar.from_ebnf_string("main ::= a b")

    with pytest.raises(
        TVMError, match="TVMError: EBNF parse error at line 1, column 15: Expect element"
    ):
        BNFGrammar.from_ebnf_string('main ::= "a" |')

    with pytest.raises(TVMError, match='TVMError: EBNF parse error at line 1, column 15: Expect "'):
        BNFGrammar.from_ebnf_string('main ::= "a" "')

    with pytest.raises(
        TVMError, match="TVMError: EBNF parse error at line 1, column 1: Expect rule name"
    ):
        BNFGrammar.from_ebnf_string('::= "a"')

    with pytest.raises(
        TVMError,
        match="TVMError: EBNF parse error at line 1, column 12: Character class should not contain "
        "newline",
    ):
        BNFGrammar.from_ebnf_string("main ::= [a\n]")

    with pytest.raises(
        TVMError, match="TVMError: EBNF parse error at line 1, column 11: Invalid escape sequence"
    ):
        BNFGrammar.from_ebnf_string(r'main ::= "\@"')

    with pytest.raises(
        TVMError, match="TVMError: EBNF parse error at line 1, column 11: Invalid escape sequence"
    ):
        BNFGrammar.from_ebnf_string(r'main ::= "\uFF"')

    with pytest.raises(
        TVMError,
        match="TVMError: EBNF parse error at line 1, column 14: Invalid character class: "
        "lower bound is larger than upper bound",
    ):
        BNFGrammar.from_ebnf_string(r"main ::= [Z-A]")

    with pytest.raises(
        TVMError, match="TVMError: EBNF parse error at line 1, column 6: Expect ::="
    ):
        BNFGrammar.from_ebnf_string(r'main := "a"')

    with pytest.raises(
        TVMError,
        match='TVMError: EBNF parse error at line 2, column 9: Rule "main" is defined multiple '
        "times",
    ):
        BNFGrammar.from_ebnf_string('main ::= "a"\nmain ::= "b"')

    with pytest.raises(
        TVMError,
        match="TVMError: EBNF parse error at line 1, column 10: "
        'The main rule with name "main" is not found.',
    ):
        BNFGrammar.from_ebnf_string('a ::= "a"')

    with pytest.raises(
        TVMError,
        match="TVMError: EBNF parse error at line 1, column 21: Unexpected lookahead assertion",
    ):
        BNFGrammar.from_ebnf_string('main ::= "a" (="a") (="b")')


def test_to_json():
    before = """main ::= b c | b main
b ::= "bcd"
c ::= [a-z]
"""
    expected_obj = {
        "rules": [
            {"body_expr_id": 6, "name": "main"},
            {"body_expr_id": 9, "name": "b"},
            {"body_expr_id": 12, "name": "c"},
        ],
        "rule_expr_indptr": [0, 3, 6, 10, 13, 16, 20, 24, 29, 32, 35, 40, 43],
        "rule_expr_data": [
            # fmt: off
            4,1,1,4,1,2,5,2,0,1,4,1,1,4,1,0,5,2,3,4,6,2,2,5,0,3,98,99,
            100,5,1,7,6,1,8,1,3,0,97,122,5,1,10,6,1,11
            # fmt: on
        ],
    }
    bnf_grammar = BNFGrammar.from_ebnf_string(before, "main")
    print(bnf_grammar)
    after_str = bnf_grammar.to_json(False)
    after_obj = json.loads(after_str)
    assert after_obj == expected_obj


def test_to_json_roundtrip():
    before = r"""main ::= ((b c) | (b main))
b ::= ((b_1 d [a]*))
c ::= ((c_1))
d ::= ((d_1))
b_1 ::= ("" | ("b" b_1))
c_1 ::= ((c_2 c_1) | (c_2))
c_2 ::= (([acep-z]))
d_1 ::= ("" | ("d"))
"""
    bnf_grammar_1 = BNFGrammar.from_ebnf_string(before, "main")
    output_json_1 = bnf_grammar_1.to_json(False)
    bnf_grammar_2 = BNFGrammar.from_json(output_json_1)
    output_json_2 = bnf_grammar_2.to_json(False)
    output_str = bnf_grammar_2.to_string()
    assert output_json_1 == output_json_2
    assert output_str == before


if __name__ == "__main__":
    tvm.testing.main()
