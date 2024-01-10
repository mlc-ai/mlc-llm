# pylint: disable=missing-module-docstring,missing-function-docstring
import os

import pytest
import tvm.testing
from tvm._ffi.base import TVMError

from mlc_chat.serve import BNFGrammar


def test_bnf_simple():
    before = """main ::= b c
b ::= "b"
c ::= "c"
"""
    expected = """main ::= b c
b ::= [b]
c ::= [c]
"""
    bnf_grammar = BNFGrammar.from_ebnf_string(before)
    after = bnf_grammar.to_string()
    assert after == expected


def test_ebnf():
    before = """main ::= b c | b main
b ::= "b"* d
c ::= [acep-z]+
d ::= "d"?
"""
    expected = """main ::= (b c) | (b main)
b ::= b_1 d
c ::= c_2
d ::= d_1
b_1 ::= ([b] b_1) | ""
c_1 ::= [acep-z]
c_2 ::= (c_1 c_2) | c_1
d_1 ::= [d] | ""
"""
    bnf_grammar = BNFGrammar.from_ebnf_string(before)
    after = bnf_grammar.to_string()
    print(after)
    assert after == expected


def test_char():
    before = r"""main ::= [a-z] [A-z] "\u0234" "\U00000345\xff" [-A-Z] [--] rest
rest ::= [a-zA-Z0-9-] [\u0234-\U00000345] [测-试] [\--\]]  rest1
rest1 ::= "\?\"\'测试あc" "👀" ""
"""
    expected = r"""main ::= [a-z] [A-z] [\u0234] ([\u0345] [\u00ff]) [\-A-Z] [\-\-] rest
rest ::= [a-zA-Z0-9\-] [\u0234-\u0345] [\u6d4b-\u8bd5] [\--\]] rest1
rest1 ::= ([\?] [\"] [\'] [\u6d4b] [\u8bd5] [\u3042] [c]) [\U0001f440] ""
"""
    bnf_grammar = BNFGrammar.from_ebnf_string(before)
    after = bnf_grammar.to_string()
    assert after == expected


def test_space():
    before = """

main::="a"  "b" ("c""d"
"e") |

"f" | "g"
"""
    expected = """main ::= ([a] [b] ([c] [d] [e])) | [f] | [g]
"""
    bnf_grammar = BNFGrammar.from_ebnf_string(before)
    after = bnf_grammar.to_string()
    assert after == expected


def test_nest():
    before = """main::= "a" ("b" | "c" "d") | (("e" "f"))
"""
    expected = """main ::= ([a] ([b] | ([c] [d]))) | ([e] [f])
"""
    bnf_grammar = BNFGrammar.from_ebnf_string(before)
    after = bnf_grammar.to_string()
    assert after == expected


def test_json():
    current_file_path = os.path.abspath(__file__)
    json_ebnf_path = os.path.join(os.path.dirname(current_file_path), "json.ebnf")

    with open(json_ebnf_path, "r", encoding="utf-8") as file:
        before = file.read()

    expected = r"""main ::= element
value ::= object | array | string | number | ([t] [r] [u] [e]) | ([f] [a] [l] [s] [e]) | ([n] [u] [l] [l])
object ::= ([{] ws [}]) | ([{] members [}])
members ::= member | (member [,] members)
member ::= ws string ws [:] element
array ::= ([[] ws [\]]) | ([[] elements [\]])
elements ::= element | (element [,] elements)
element ::= ws value ws
string ::= [\"] characters [\"]
characters ::= "" | (character characters)
character ::= [\"\\] | ([\\] escape)
escape ::= [\"] | [\\] | [/] | [b] | [f] | [n] | [r] | [t] | ([u] hex hex hex hex)
hex ::= [A-Fa-f0-9]
number ::= integer fraction exponent
integer ::= digit | (onenine digits) | ([\-] digit) | ([\-] onenine digits)
digits ::= digit | (digit digits)
digit ::= [0-9]
onenine ::= [1-9]
fraction ::= "" | ([.] digits)
exponent ::= "" | (([e] | [E]) ("" | [+] | [\-]) digits)
ws ::= "" | ([ ] ws) | ([\n] ws) | ([\r] ws) | ([\t] ws)
"""

    bnf_grammar = BNFGrammar.from_ebnf_string(before)
    after = bnf_grammar.to_string()
    assert after == expected


def test_to_string_roundtrip():
    before = r"""main ::= (b c) | (b main)
b ::= b_1 d
c ::= c_1
d ::= d_1
b_1 ::= ([b] b_1) | ""
c_1 ::= (c_2 c_1) | c_2
c_2 ::= [acep-z]
d_1 ::= [d] | ""
"""
    bnf_grammar = BNFGrammar.from_ebnf_string(before)
    string = bnf_grammar.to_string()
    new_grammar = BNFGrammar.from_ebnf_string(string)
    new_string = new_grammar.to_string()
    assert string == new_string


def test_error():
    with pytest.raises(TVMError, match="Rule a is not defined at line 1, column 11"):
        BNFGrammar.from_ebnf_string("main ::= a b")

    with pytest.raises(TVMError, match="Expect element at line 1, column 15"):
        BNFGrammar.from_ebnf_string('main ::= "a" |')

    with pytest.raises(TVMError, match='Expect " at line 1, column 15'):
        BNFGrammar.from_ebnf_string('main ::= "a" "')

    with pytest.raises(TVMError, match="Expect rule name at line 1, column 1"):
        BNFGrammar.from_ebnf_string('::= "a"')

    with pytest.raises(
        TVMError, match="Character range should not contain newline at line 1, column 12"
    ):
        BNFGrammar.from_ebnf_string("main ::= [a\n]")

    with pytest.raises(TVMError, match="Invalid escape sequence at line 1, column 11"):
        BNFGrammar.from_ebnf_string(r'main ::= "\@"')

    with pytest.raises(TVMError, match="Invalid escape sequence at line 1, column 11"):
        BNFGrammar.from_ebnf_string(r'main ::= "\uFF"')

    with pytest.raises(
        TVMError,
        match="Invalid character range: lower bound is larger than upper bound at "
        "line 1, column 14",
    ):
        BNFGrammar.from_ebnf_string(r"main ::= [Z-A]")

    with pytest.raises(TVMError, match="Expect ::= at line 1, column 6"):
        BNFGrammar.from_ebnf_string(r'main := "a"')

    with pytest.raises(TVMError, match="Rule main is defined multiple times at line 2, column 9"):
        BNFGrammar.from_ebnf_string('main ::= "a"\nmain ::= "b"')

    with pytest.raises(TVMError, match="There must be a rule named main at line 1, column 10"):
        BNFGrammar.from_ebnf_string('a ::= "a"')


def test_to_json():
    before = """main ::= b c | b main
b ::= "bcd"
c ::= [a-z]
"""
    expected = (
        '{"rule_expr_indptr":[0,2,4,7,9,11,14,17,20,23,26,30,32,34,37,39],'
        '"rule_expr_data":[3,1,3,2,4,0,1,3,1,3,0,4,3,4,5,2,5,0,98,98,0,99,99,0,100,'
        "100,4,7,8,9,4,10,5,11,0,97,122,4,13,5,14],"
        '"rules":[{"rule_expr_id":6,"name":"main"},{"rule_expr_id":12,"name":"b"},'
        '{"rule_expr_id":15,"name":"c"}]}'
    )
    bnf_grammar = BNFGrammar.from_ebnf_string(before)
    after = bnf_grammar.to_json(False)
    assert after == expected


def test_to_json_roundtrip():
    before = r"""main ::= (b c) | (b main)
b ::= b_1 d
c ::= c_1
d ::= d_1
b_1 ::= ([b] b_1) | ""
c_1 ::= (c_2 c_1) | c_2
c_2 ::= [acep-z]
d_1 ::= [d] | ""
"""
    bnf_grammar = BNFGrammar.from_ebnf_string(before)
    json = bnf_grammar.to_json(False)
    new_grammar = BNFGrammar.from_json(json)
    new_json = new_grammar.to_json(False)
    after = new_grammar.to_string()
    assert json == new_json
    assert after == before


if __name__ == "__main__":
    tvm.testing.main()
