# pylint: disable=missing-module-docstring,missing-function-docstring
import os

import pytest
import tvm.testing
from tvm import TVMError

from mlc_chat.serve import BNFGrammar


def test_bnf_simple():
    before = """main ::= b c
b ::= "b"
c ::= "c"
"""
    expected = """main ::= ((b c))
b ::= (([b]))
c ::= (([c]))
"""
    bnf_grammar = BNFGrammar.from_ebnf_string(before, True, False)
    after = bnf_grammar.to_string()
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
b_1 ::= ("" | ([a] [b] b_1))
c_1 ::= (([acep-z] c_1) | ([acep-z]))
d_1 ::= ("" | ([d]))
"""
    bnf_grammar = BNFGrammar.from_ebnf_string(before, True, False)
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
b ::= [b]*
c ::= ((c_1))
d ::= ((d_1))
e ::= ((e_star e_star_1) | (e_star_2))
c_1 ::= ("" | ([b] c_1))
d_1 ::= ("" | (d_1_choice d_1))
e_star ::= [e]*
e_star_1 ::= [f]*
e_star_2 ::= [g]*
d_1_choice ::= (([b] [c] [d]) | ([p] [q]))
"""
    bnf_grammar = BNFGrammar.from_ebnf_string(before, True, False)
    after = bnf_grammar.to_string()
    assert after == expected


def test_char():
    before = r"""main ::= [a-z] [A-z] "\u0234" "\U00000345\xff" [-A-Z] [--] [^a] rest
rest ::= [a-zA-Z0-9-] [\u0234-\U00000345] [测-试] [\--\]]  rest1
rest1 ::= "\?\"\'测试あc" "👀" ""
"""
    expected = r"""main ::= (([a-z] [A-z] ([\u0234]) ([\u0345] [\u00ff]) [\-A-Z] [\-\-] [^a] rest))
rest ::= (([a-zA-Z0-9\-] [\u0234-\u0345] [\u6d4b-\u8bd5] [\--\]] rest1))
rest1 ::= ((([\?] [\"] [\'] [\u6d4b] [\u8bd5] [\u3042] [c]) ([\U0001f440]) ""))
"""
    # Disable unwrap_nesting_rules to expose the result before unwrapping.
    bnf_grammar = BNFGrammar.from_ebnf_string(before, False, False)
    after = bnf_grammar.to_string()
    assert after == expected


def test_space():
    before = """

main::="a"  "b" ("c""d"
"e") |

"f" | "g"
"""
    expected = """main ::= (([a] [b] [c] [d] [e]) | ([f]) | ([g]))
"""
    bnf_grammar = BNFGrammar.from_ebnf_string(before, True, False)
    after = bnf_grammar.to_string()
    assert after == expected


def test_nest():
    before = """main::= "a" ("b" | "c" "d") | (("e" "f"))
"""
    expected = """main ::= (([a] main_choice) | ([e] [f]))
main_choice ::= (([b]) | ([c] [d]))
"""
    bnf_grammar = BNFGrammar.from_ebnf_string(before, True, False)
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
or_test ::= ("" | ([a]) | ([b]) | ([d] [e]) | (or_test) | ([^a-z]))
sequence_test ::= (([a] [a] [b] sequence_test_choice [d] [e] sequence_test))
nested_test ::= (([a] [b] [c] [d]) | ([a]) | ([b]) | ([c]) | (nested_rest))
nested_rest ::= (([a]) | ([b] [c]) | ([d]) | ([e] [f]) | ([g]))
empty_test ::= ("" | ([d]) | ([a]))
sequence_test_choice ::= (([c]) | ([d]))
"""
    bnf_grammar = BNFGrammar.from_ebnf_string(before, True, False)
    after = bnf_grammar.to_string()
    assert after == expected


def test_json():
    current_file_path = os.path.abspath(__file__)
    json_ebnf_path = os.path.join(os.path.dirname(current_file_path), "json.ebnf")

    with open(json_ebnf_path, "r", encoding="utf-8") as file:
        before = file.read()

    expected = r"""main ::= ((element))
value ::= ((object) | (array) | (string) | (number) | ([t] [r] [u] [e]) | ([f] [a] [l] [s] [e]) | ([n] [u] [l] [l]))
object ::= (([{] ws [}]) | ([{] members [}]))
members ::= ((member) | (member [,] members))
member ::= ((ws string ws [:] element))
array ::= (([[] ws [\]]) | ([[] elements [\]]))
elements ::= ((element) | (element [,] elements))
element ::= ((ws value ws))
string ::= (([\"] characters [\"]))
characters ::= ("" | (character characters))
character ::= (([^\"\\]) | ([\\] escape))
escape ::= (([\"]) | ([\\]) | ([/]) | ([b]) | ([f]) | ([n]) | ([r]) | ([t]) | ([u] hex hex hex hex))
hex ::= (([A-Fa-f0-9]))
number ::= ((integer fraction exponent))
integer ::= ((digit) | (onenine digits) | ([\-] digit) | ([\-] onenine digits))
digits ::= ((digit) | (digit digits))
digit ::= (([0-9]))
onenine ::= (([1-9]))
fraction ::= ("" | ([.] digits))
exponent ::= ("" | (exponent_choice exponent_choice_1 digits))
ws ::= ("" | ([ ] ws) | ([\n] ws) | ([\r] ws) | ([\t] ws))
exponent_choice ::= (([e]) | ([E]))
exponent_choice_1 ::= ("" | ([+]) | ([\-]))
"""

    bnf_grammar = BNFGrammar.from_ebnf_string(before, True, False)
    after = bnf_grammar.to_string()
    assert after == expected


def test_to_string_roundtrip():
    """Checks the printed result can be parsed, and the parsing-printing process is idempotent."""

    before = r"""main ::= (b c) | (b main)
b ::= b_1 d
c ::= c_1
d ::= d_1
b_1 ::= ([b] b_1) | ""
c_1 ::= (c_2 c_1) | c_2
c_2 ::= [acep-z]
d_1 ::= [d] | ""
"""
    bnf_grammar_1 = BNFGrammar.from_ebnf_string(before, True, False)
    output_string_1 = bnf_grammar_1.to_string()
    bnf_grammar_2 = BNFGrammar.from_ebnf_string(output_string_1, True, False)
    output_string_2 = bnf_grammar_2.to_string()
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
        match='TVMError: EBNF parse error at line 1, column 10: There must be a rule named "main"',
    ):
        BNFGrammar.from_ebnf_string('a ::= "a"')


def test_to_json():
    before = """main ::= b c | b main
b ::= "bcd"
c ::= [a-z]
"""
    expected = (
        '{"rule_expr_indptr":[0,3,6,10,13,16,20,24,28,32,36,41,44,48,51],"rule_expr_data"'
        ":[3,1,1,3,1,2,4,2,0,1,3,1,1,3,1,0,4,2,3,4,5,2,2,5,0,2,98,98,0,2,99,99,0,2,100,100,"
        '4,3,7,8,9,5,1,10,0,2,97,122,4,1,12,5,1,13],"rules":[{"body_expr_id":6,"name":"main"},'
        '{"body_expr_id":11,"name":"b"},{"body_expr_id":14,"name":"c"}]}'
    )
    bnf_grammar = BNFGrammar.from_ebnf_string(before, True, False)
    after = bnf_grammar.to_json(False)
    assert after == expected


def test_to_json_roundtrip():
    before = r"""main ::= ((b c) | (b main))
b ::= ((b_1 d))
c ::= ((c_1))
d ::= ((d_1))
b_1 ::= ("" | ([b] b_1))
c_1 ::= ((c_2 c_1) | (c_2))
c_2 ::= (([acep-z]))
d_1 ::= ("" | ([d]))
"""
    bnf_grammar_1 = BNFGrammar.from_ebnf_string(before, True, False)
    output_json_1 = bnf_grammar_1.to_json(False)
    bnf_grammar_2 = BNFGrammar.from_json(output_json_1)
    output_json_2 = bnf_grammar_2.to_json(False)
    output_str = bnf_grammar_2.to_string()
    assert output_json_1 == output_json_2
    assert output_str == before


if __name__ == "__main__":
    tvm.testing.main()
