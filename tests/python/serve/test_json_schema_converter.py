import json
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import tvm.testing
from pydantic import BaseModel, TypeAdapter

from mlc_llm.serve.grammar import BNFGrammar, GrammarStateMatcher
from mlc_llm.serve.json_schema_converter import json_schema_to_ebnf


def check_schema_with_grammar(schema: Dict[str, Any], expected_grammar: str):
    schema_str = json.dumps(schema, indent=2)
    print(schema_str)
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

    ebnf_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= "" | [^"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*) ".0"?
basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub ["]
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= "[" ("" | "" basic_any ("," basic_any)*) "]"
basic_object ::= "{" ("" | "" basic_string ":" basic_any ("," basic_string ":" basic_any)*) "}"
main_any_array_field ::= "[" ("" | "" basic_any ("," basic_any)*) "]"
main_array_field ::= "[" ("" | "" basic_string ("," basic_string)*) "]"
main_tuple_field_2 ::= "[" ("" | "" basic_string ("," basic_string)*) "]"
main_tuple_field ::= "[" "" basic_string "," basic_integer "," main_tuple_field_2 ("" | "," basic_any ("," basic_any)*) "]"
main_object_field ::= "{" ("" | "" basic_string ":" basic_integer ("," basic_string ":" basic_integer)*) "}"
main_nested_object_field_add ::= "{" ("" | "" basic_string ":" basic_integer ("," basic_string ":" basic_integer)*) "}"
main_nested_object_field ::= "{" ("" | "" basic_string ":" main_nested_object_field_add ("," basic_string ":" main_nested_object_field_add)*) "}"
main ::= "{" "" "\"integer_field\"" ":" basic_integer "," "\"number_field\"" ":" basic_number "," "\"boolean_field\"" ":" basic_boolean "," "\"any_array_field\"" ":" main_any_array_field "," "\"array_field\"" ":" main_array_field "," "\"tuple_field\"" ":" main_tuple_field "," "\"object_field\"" ":" main_object_field "," "\"nested_object_field\"" ":" main_nested_object_field ("" | "," basic_string ":" basic_any ("," basic_string ":" basic_any)*) "}"
"""

    instance = MainModel(
        integer_field=42,
        number_field=3.14e5,
        boolean_field=True,
        any_array_field=[3.14, "foo", [None, True]],
        array_field=["foo", "bar"],
        tuple_field=("foo", 42, ["bar", "baz"]),
        object_field={"foo": 42, "bar": 43},
        nested_object_field={"foo": {"bar": 42}},
    )

    check_schema_with_grammar(MainModel.model_json_schema(), ebnf_grammar)
    check_schema_with_instance(MainModel.model_json_schema(), instance)


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

    ebnf_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= "" | [^"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*) ".0"?
basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub ["]
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= "[" ("" | "" basic_any ("," basic_any)*) "]"
basic_object ::= "{" ("" | "" basic_string ":" basic_any ("," basic_string ":" basic_any)*) "}"
main_bars ::= "\"a\""
main_str_values ::= "\"a\\n\\r\\\"\""
main_foo ::= ("\"a\"") | ("\"b\"") | ("\"c\"")
main_values ::= ("1") | ("\"a\"") | ("true")
main_field ::= ("\"foo\"") | ("\"bar\"")
main ::= "{" "" "\"bars\"" ":" main_bars "," "\"str_values\"" ":" main_str_values "," "\"foo\"" ":" main_foo "," "\"values\"" ":" main_values "," "\"field\"" ":" main_field ("" | "," basic_string ":" basic_any ("," basic_string ":" basic_any)*) "}"
"""

    instance = MainModel(foo="a", values=1, bars="a", str_values='a\n\r"', field=Field.FOO)

    check_schema_with_instance(MainModel.model_json_schema(), instance)
    check_schema_with_grammar(MainModel.model_json_schema(), ebnf_grammar)


def test_optional():
    class MainModel(BaseModel):
        num: int = 0
        opt_bool: Optional[bool] = None
        size: Optional[float]
        name: str = ""

    ebnf_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= "" | [^"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*) ".0"?
basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub ["]
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= "[" ("" | "" basic_any ("," basic_any)*) "]"
basic_object ::= "{" ("" | "" basic_string ":" basic_any ("," basic_string ":" basic_any)*) "}"
main_opt_bool ::= basic_boolean | basic_null
main_size ::= basic_number | basic_null
main ::= "{" "" ("\"num\"" ":" basic_integer ",")? ("\"opt_bool\"" ":" main_opt_bool ",")? "\"size\"" ":" main_size ("," "\"name\"" ":" basic_string)? ("" | "," basic_string ":" basic_any ("," basic_string ":" basic_any)*) "}"
"""

    check_schema_with_grammar(MainModel.model_json_schema(), ebnf_grammar)

    instance = MainModel(num=42, opt_bool=True, size=3.14, name="foo")
    check_schema_with_instance(MainModel.model_json_schema(), instance)

    instance = MainModel(size=None)
    check_schema_with_instance(MainModel.model_json_schema(), instance)

    check_schema_with_json(MainModel.model_json_schema(), '{"size":null}')
    check_schema_with_json(MainModel.model_json_schema(), '{"size":null,"name":"foo"}')
    check_schema_with_json(MainModel.model_json_schema(), '{"num":1,"size":null,"name":"foo"}')


def test_all_optional():
    class MainModel(BaseModel):
        size: int = 0
        state: bool = False
        num: float = 0

    ebnf_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= "" | [^"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*) ".0"?
basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub ["]
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= "[" ("" | "" basic_any ("," basic_any)*) "]"
basic_object ::= "{" ("" | "" basic_string ":" basic_any ("," basic_string ":" basic_any)*) "}"
main ::= "{" ("" ((("\"size\"" ":" basic_integer) ("" | "," "\"state\"" ":" basic_boolean) | "\"state\"" ":" basic_boolean) ("" | "," "\"num\"" ":" basic_number) | "\"num\"" ":" basic_number) | "") ("" | "," basic_string ":" basic_any ("," basic_string ":" basic_any)*) "}"
"""

    check_schema_with_grammar(MainModel.model_json_schema(), ebnf_grammar)

    instance = MainModel(size=42, state=True, num=3.14)
    check_schema_with_instance(MainModel.model_json_schema(), instance)

    check_schema_with_json(MainModel.model_json_schema(), '{"state":false}')
    check_schema_with_json(MainModel.model_json_schema(), '{"size":1,"num":1.5}')
    check_schema_with_json(MainModel.model_json_schema(), '{"other": null}')


test_all_optional()
exit()


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
