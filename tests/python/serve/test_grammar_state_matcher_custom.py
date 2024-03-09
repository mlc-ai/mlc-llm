# pylint: disable=missing-module-docstring,missing-function-docstring
# pylint: disable=redefined-outer-name,unbalanced-tuple-unpacking
"""This test is adopted from test_grammar_state_matcher_json.py, but the grammar is parsed from
a unoptimized, non-simplified EBNF string. This is to test the robustness of the grammar state
matcher."""
import sys
from typing import List, Optional

import pytest
import tvm
import tvm.testing

from mlc_chat.serve import BNFGrammar, GrammarStateMatcher
from mlc_chat.tokenizer import Tokenizer


def get_json_grammar():
    json_grammar_ebnf = r"""
main ::= basic_array | basic_object
basic_any ::= basic_integer | basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*) ".0"?
basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= (([\"] basic_string_1 [\"]))
basic_string_1 ::= "" | [^"\\\r\n] basic_string_1 | "\\" escape basic_string_1
escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= "[" ("" | ws basic_any (ws "," ws basic_any)*) ws "]"
basic_object ::= "{" ("" | ws basic_string ws ":" ws basic_any ( ws "," ws basic_string ws ":" ws basic_any)*) ws "}"
ws ::= [ \n\t]*
"""
    grammar = BNFGrammar.from_ebnf_string(json_grammar_ebnf)
    print(grammar)
    return grammar


@pytest.fixture(scope="function")
def json_grammar():
    return get_json_grammar()


(json_input_accepted,) = tvm.testing.parameters(
    ('{"name": "John"}',),
    ('{ "name" : "John" }',),
    ("{}",),
    ("[]",),
    ('{"name": "Alice", "age": 30, "city": "New York"}',),
    ('{"name": "Mike", "hobbies": ["reading", "cycling", "hiking"]}',),
    ('{"name": "Emma", "address": {"street": "Maple Street", "city": "Boston"}}',),
    ('[{"name": "David"}, {"name": "Sophia"}]',),
    (
        '{"name": "William", "age": null, "married": true, "children": ["Liam", "Olivia"],'
        ' "hasPets": false}',
    ),
    (
        '{"name": "Olivia", "contact": {"email": "olivia@example.com", "address": '
        '{"city": "Chicago", "zipcode": "60601"}}}',
    ),
    (
        '{"name": "Liam", "skills": ["Java", "Python"], "experience": '
        '[{"company": "CompanyA", "years": 5}, {"company": "CompanyB", "years": 3}]}',
    ),
    (
        '{"person": {"name": "Ethan", "age": 40}, "education": {"degree": "Masters", '
        '"university": "XYZ University"}, "work": [{"company": "ABC Corp", "position": '
        '"Manager"}, {"company": "DEF Corp", "position": "Senior Manager"}]}',
    ),
    (
        '{"name": "Charlotte", "details": {"personal": {"age": 35, "hobbies": ["gardening", '
        '"painting"]}, "professional": {"occupation": "Engineer", "skills": '
        '["CAD", "Project Management"], "projects": [{"name": "Project A", '
        '"status": "Completed"}, {"name": "Project B", "status": "In Progress"}]}}}',
    ),
)


def test_json_accept(json_grammar: BNFGrammar, json_input_accepted: str):
    assert GrammarStateMatcher(json_grammar).debug_match_complete_string(json_input_accepted)


(json_input_refused,) = tvm.testing.parameters(
    (r'{ name: "John" }',),
    (r'{ "name": "John" } ',),  # trailing space is not accepted
    (r'{ "name": "John", "age": 30, }',),
    (r'{ "name": "John", "address": { "street": "123 Main St", "city": "New York" }',),
    (r'{ "name": "John", "age": 30, "hobbies": ["reading", "traveling",], }',),
    (r'{ "name": "John", "age": 30.5.7 }',),
    (r'{ "name": "John, "age": 30, "hobbies": ["reading", "traveling"] }',),
    (
        r'{ "name": "John", "age": 30, "hobbies": ["reading", { "type": "outdoor", "list": '
        r'["hiking", "swimming",]}] }',
    ),
    (r'{ "name": "John", "age": 30, "status": "\P\J" }',),
    (
        r'{ "name": "John", "age": 30, "hobbies": ["reading", "traveling"], "address": '
        r'{ "street": "123 Main St", "city": "New York", "coordinates": { "latitude": 40.7128, '
        r'"longitude": -74.0060 }}}, "work": { "company": "Acme", "position": "developer" }}',
    ),
)


def test_json_refuse(json_grammar: BNFGrammar, json_input_refused):
    assert not GrammarStateMatcher(json_grammar).debug_match_complete_string(json_input_refused)


(input_find_rejected_tokens, expected_rejected_sizes) = tvm.testing.parameters(
    (
        # short test
        '{"id": 1,"name": "Example"}',
        [
            # fmt: off
            31989, 31912, 299, 299, 299, 31973, 31846, 31846, 31948, 31915, 299, 299, 299, 299,
            299, 31973, 31846, 31846, 292, 292, 292, 292, 292, 292, 292, 292, 31974, 31999
            # fmt: on
        ],
    ),
    (
        # long test
        """{
"id": 1,
"na": "ex",
"ac": true,
"t": ["t1", "t2"],
"ne": {"lv2": {"val": "dp"}, "arr": [1, 2, 3]},
"res": "res"
}""",
        [
            # fmt: off
            31989, 31912, 31912, 299, 299, 299, 31973, 31846, 31846, 31948, 31915, 31915, 299, 299,
            299, 31973, 31846, 31846, 292, 292, 292, 31974, 31915, 31915, 299, 299, 299, 31973,
            31846, 31846, 31997, 31997, 31998, 31974, 31915, 31915, 299, 299, 31973, 31846, 31846,
            31840, 291, 291, 291, 31969, 31846, 31846, 291, 291, 291, 31969, 31974, 31915, 31915,
            299, 299, 299, 31973, 31846, 31846, 31908, 299, 299, 299, 299, 31973, 31846, 31846,
            31906, 299, 299, 299, 299, 31973, 31846, 31846, 291, 291, 291, 31968, 31970, 31915,
            31915, 299, 299, 299, 299, 31973, 31846, 31846, 31840, 31943, 31846, 31846, 31943,
            31846, 31846, 31943, 31970, 31974, 31915, 31915, 299, 299, 299, 299, 31973, 31846,
            31846, 292, 292, 292, 292, 31974, 31974, 31999
            # fmt: on
        ],
    ),
)


def test_find_next_rejected_tokens(
    json_grammar: BNFGrammar,
    input_find_rejected_tokens: str,
    expected_rejected_sizes: Optional[List[int]] = None,
):
    tokenizer_path = "dist/Llama-2-7b-chat-hf-q4f16_1-MLC"
    tokenizer = Tokenizer(tokenizer_path)
    grammar_state_matcher = GrammarStateMatcher(json_grammar, tokenizer)

    real_sizes = []
    for c in input_find_rejected_tokens:
        rejected_token_ids = grammar_state_matcher.find_next_rejected_tokens()
        real_sizes.append(len(rejected_token_ids))
        print("Accepting char:", c, file=sys.stderr)
        assert grammar_state_matcher.debug_accept_char(ord(c))
    rejected_token_ids = grammar_state_matcher.find_next_rejected_tokens()
    real_sizes.append(len(rejected_token_ids))

    if expected_rejected_sizes is not None:
        assert real_sizes == expected_rejected_sizes


def test_token_based_operations(json_grammar: BNFGrammar):
    """Test accepting token and finding the next token mask."""
    token_table = [
        # fmt: off
        "<s>", "</s>", "a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", "\n", " ", '"a":true',
        # fmt: on
    ]
    input_splitted = ["{", '"', "abc", 'b"', ":", "6", ", ", " ", '"a":true', "}"]
    input_ids = [token_table.index(t) for t in input_splitted]

    grammar_state_matcher = GrammarStateMatcher(json_grammar, token_table)

    expected = [
        ["{"],
        ['"', "}", "\n", " ", '"a":true'],
        ["a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", " "],
        ["a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", " "],
        [":", "\n", " ", ':"'],
        ['"', "{", "6", "\n", " "],
        ["}", ", ", "6", "\n", " "],
        [" ", "\n", '"', '"a":true'],
        [" ", "\n", '"', '"a":true'],
        ["}", ", ", "\n", " "],
        ["</s>"],
    ]

    result = []

    for id in input_ids:
        rejected = grammar_state_matcher.find_next_rejected_tokens()
        accepted = list(set(range(len(token_table))) - set(rejected))
        accepted_tokens = [token_table[i] for i in accepted]
        result.append(accepted_tokens)
        assert id in accepted
        assert grammar_state_matcher.accept_token(id)

    rejected = grammar_state_matcher.find_next_rejected_tokens()
    accepted = list(set(range(len(token_table))) - set(rejected))
    accepted_tokens = [token_table[i] for i in accepted]
    result.append(accepted_tokens)

    assert result == expected


if __name__ == "__main__":
    # Run a benchmark to show the performance before running tests
    test_find_next_rejected_tokens(get_json_grammar(), '{"id": 1,"name": "Example"}')

    tvm.testing.main()
