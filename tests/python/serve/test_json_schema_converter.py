import json
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import tvm.testing
from pydantic import BaseModel, TypeAdapter

from mlc_llm.serve.grammar import BNFGrammar, GrammarStateMatcher
from mlc_llm.serve.json_schema_converter import json_schema_to_ebnf


def check_schema_with_grammar(schema: Dict[str, Any], expected_grammar: str):
    schema_str = json.dumps(schema, indent=2)
    grammar = json_schema_to_ebnf(schema_str, separators=(",", ":"))
    print(grammar)
    print(expected_grammar)
    assert grammar == expected_grammar


def check_schema_with_json(schema: Dict[str, Any], json_str: str, check_accepted=True):
    schema_str = json.dumps(schema, indent=2)

    ebnf_grammar_str = json_schema_to_ebnf(schema_str, separators=(",", ":"))
    ebnf_grammar = BNFGrammar.from_ebnf_string(ebnf_grammar_str)
    matcher = GrammarStateMatcher(ebnf_grammar)

    print("json str:", json_str)

    if check_accepted:
        assert matcher.debug_match_complete_string(json_str)
    else:
        assert not matcher.debug_match_complete_string(json_str)


def check_schema_with_instance(schema: Dict[str, Any], instance: BaseModel):
    check_schema_with_json(schema, instance.model_dump_json(round_trip=True))


def test_basic():
    class MainModel(BaseModel):
        integer_field: int
        number_field: float
        boolean_field: bool
        any_array_field: List
        array_field: List[str]
        tuple_field: Tuple[str, int, List[str]]
        object_field: Dict[str, int]
        nested_object_field: Dict[str, Dict[str, int]]

    ebnf_grammar = r"""basic_ws ::= [ \n\t]*
basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= "" | [^"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*) ".0"?
basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub ["]
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= "[" ("" | basic_any ("," basic_any)*) "]"
basic_object ::= "{" ("" | basic_string ":" basic_any ("," basic_string ":" basic_any)*) "}"
main_array_field ::= "[" ("" | basic_string ("," basic_string)*) "]"
main_tuple_field ::= "[" basic_string "," basic_integer "," main_array_field ("," basic_any)* "]"
main_object_field ::= "{" ("" | basic_string ":" basic_integer ("," basic_string ":" basic_integer)*) "}"
main_nested_object_field ::= "{" ("" | basic_string ":" main_object_field ("," basic_string ":" main_object_field)*) "}"
main ::= "{" "\"integer_field\"" ":" basic_integer "," "\"number_field\"" ":" basic_number "," "\"boolean_field\"" ":" basic_boolean "," "\"array_field\"" ":" main_array_field "," "\"tuple_field\"" ":" main_tuple_field "," "\"object_field\"" ":" main_object_field "," "\"nested_object_field\"" ":" main_nested_object_field ("," basic_string ":" basic_any)* "}"
"""

    instance = MainModel(
        integer_field=42,
        number_field=3.14,
        boolean_field=True,
        any_array_field=[3.14, "foo", [None, True]],
        array_field=["foo", "bar"],
        tuple_field=("foo", 42, ["bar", "baz"]),
        object_field={"foo": 42, "bar": 43},
        nested_object_field={"foo": {"bar": 42}},
    )
    # print(instance.model_dump_json(round_trip=True))
    # print(instance.model_dump_json(round_trip=True, indent=2))
    # print(
    #     json.dumps(
    #         json.loads(instance.model_dump_json(round_trip=True)), indent=2, separators=(",", ":")
    #     )
    # )
    # print(
    #     json.dumps(
    #         json.loads(instance.model_dump_json(round_trip=True)),
    #         indent=None,
    #         separators=(",", ":"),
    #     )
    # )

    # check_schema_with_grammar(MainModel.model_json_schema(), ebnf_grammar)
    check_schema_with_instance(MainModel.model_json_schema(), instance)


test_basic()
exit()


def test_enum_const():
    class Field(Enum):
        FOO = "foo"
        BAR = "bar"

    class MainModel(BaseModel):
        bars: Literal["a"]
        str_values: Literal['a\n\r"']
        foo: Literal["a", "b", "c"]
        values: Literal[1, "a", True]
        field: Field

    instance = MainModel(foo="a", values=1, bars="a", str_values='a\n\r"', field=Field.FOO)
    check_schema_with_instance(MainModel.model_json_schema(), instance)


def test_optional():
    class MainModel(BaseModel):
        size: Optional[float]
        name: str = None

    instance = MainModel(size=3.14)
    check_schema_with_instance(MainModel.model_json_schema(), instance)

    instance = MainModel()
    check_schema_with_instance(MainModel.model_json_schema(), instance)

    check_schema_with_json(MainModel.model_json_schema(), "{}")


def test_reference():
    class Foo(BaseModel):
        count: int
        size: Optional[float] = None

    class Bar(BaseModel):
        apple: str = "x"
        banana: str = "y"

    class MainModel(BaseModel):
        foo: Foo
        bars: List[Bar]

    instance = MainModel(
        foo=Foo(count=42, size=3.14),
        bars=[Bar(apple="a", banana="b"), Bar(apple="c", banana="d")],
    )

    check_schema_with_instance(MainModel.model_json_schema(), instance)


def test_union():
    class Cat(BaseModel):
        name: str
        color: str

    class Dog(BaseModel):
        name: str
        breed: str

    ta = TypeAdapter(Union[Cat, Dog])

    model_schema = ta.json_schema()

    check_schema_with_instance(model_schema, Cat(name="kitty", color="black"))
    check_schema_with_instance(model_schema, Dog(name="doggy", breed="bulldog"))
    check_schema_with_json(model_schema, '{"name":"kitty","test":"black"}', False)


def test_alias():
    pass


if __name__ == "__main__":
    tvm.testing.main()
