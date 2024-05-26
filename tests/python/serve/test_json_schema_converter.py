import json
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import tvm.testing
from pydantic import BaseModel, Field, TypeAdapter

from mlc_llm.serve import BNFGrammar, GrammarStateMatcher


def check_schema_with_grammar(
    schema: Dict[str, Any],
    expected_grammar: str,
    indent: Optional[int] = None,
    separators: Optional[Tuple[str, str]] = None,
    strict_mode: bool = True,
):
    schema_str = json.dumps(schema, indent=2)
    grammar = BNFGrammar.debug_json_schema_to_ebnf(
        schema_str, indent=indent, separators=separators, strict_mode=strict_mode
    )
    assert grammar == expected_grammar


def check_schema_with_json(
    schema: Dict[str, Any],
    json_str: str,
    check_accepted: bool = True,
    indent: Optional[int] = None,
    separators: Optional[Tuple[str, str]] = None,
    strict_mode: bool = True,
):
    ebnf_grammar = BNFGrammar.from_schema(
        json.dumps(schema, indent=2), indent=indent, separators=separators, strict_mode=strict_mode
    )
    matcher = GrammarStateMatcher(ebnf_grammar)

    if check_accepted:
        assert matcher.debug_match_complete_string(json_str)
    else:
        assert not matcher.debug_match_complete_string(json_str)


def check_schema_with_instance(
    schema: Dict[str, Any],
    instance: BaseModel,
    check_accepted: bool = True,
    indent: Optional[int] = None,
    separators: Optional[Tuple[str, str]] = None,
    strict_mode: bool = True,
):
    instance_obj = instance.model_dump(mode="json", round_trip=True)
    instance_str = json.dumps(instance_obj, indent=indent, separators=separators)
    check_schema_with_json(schema, instance_str, check_accepted, indent, separators, strict_mode)


def test_basic() -> None:
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
basic_array ::= ("[" "" basic_any (", " basic_any)* "" "]") | "[]"
basic_object ::= ("{" "" basic_string ": " basic_any (", " basic_string ": " basic_any)* "" "}") | "{}"
main_prop_3 ::= ("[" "" basic_any (", " basic_any)* "" "]") | "[]"
main_prop_4 ::= ("[" "" basic_string (", " basic_string)* "" "]") | "[]"
main_prop_5_item_2 ::= ("[" "" basic_string (", " basic_string)* "" "]") | "[]"
main_prop_5 ::= "[" "" basic_string ", " basic_integer ", " main_prop_5_item_2 "" "]"
main_prop_6 ::= ("{" "" basic_string ": " basic_integer (", " basic_string ": " basic_integer)* "" "}") | "{}"
main_prop_7_addl ::= ("{" "" basic_string ": " basic_integer (", " basic_string ": " basic_integer)* "" "}") | "{}"
main_prop_7 ::= ("{" "" basic_string ": " main_prop_7_addl (", " basic_string ": " main_prop_7_addl)* "" "}") | "{}"
main ::= "{" "" "\"integer_field\"" ": " basic_integer ", " "\"number_field\"" ": " basic_number ", " "\"boolean_field\"" ": " basic_boolean ", " "\"any_array_field\"" ": " main_prop_3 ", " "\"array_field\"" ": " main_prop_4 ", " "\"tuple_field\"" ": " main_prop_5 ", " "\"object_field\"" ": " main_prop_6 ", " "\"nested_object_field\"" ": " main_prop_7 "" "}"
"""

    schema = MainModel.model_json_schema()
    check_schema_with_grammar(schema, ebnf_grammar)

    instance = MainModel(
        integer_field=42,
        number_field=3.14e5,
        boolean_field=True,
        any_array_field=[3.14, "foo", None, True],
        array_field=["foo", "bar"],
        tuple_field=("foo", 42, ["bar", "baz"]),
        object_field={"foo": 42, "bar": 43},
        nested_object_field={"foo": {"bar": 42}},
    )
    check_schema_with_instance(schema, instance)

    instance_empty = MainModel(
        integer_field=42,
        number_field=3.14e5,
        boolean_field=True,
        any_array_field=[],
        array_field=[],
        tuple_field=("foo", 42, []),
        object_field={},
        nested_object_field={},
    )

    schema = MainModel.model_json_schema()
    check_schema_with_instance(schema, instance_empty)


def test_indent() -> None:
    class MainModel(BaseModel):
        array_field: List[str]
        tuple_field: Tuple[str, int, List[str]]
        object_field: Dict[str, int]

    ebnf_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= "" | [^"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*) ".0"?
basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub ["]
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= ("[" "" basic_any ("," basic_any)* "" "]") | "[]"
basic_object ::= ("{" "" basic_string ": " basic_any ("," basic_string ": " basic_any)* "" "}") | "{}"
main_prop_0 ::= ("[" "\n    " basic_string (",\n    " basic_string)* "\n  " "]") | "[]"
main_prop_1_item_2 ::= ("[" "\n      " basic_string (",\n      " basic_string)* "\n    " "]") | "[]"
main_prop_1 ::= "[" "\n    " basic_string ",\n    " basic_integer ",\n    " main_prop_1_item_2 "\n  " "]"
main_prop_2 ::= ("{" "\n    " basic_string ": " basic_integer (",\n    " basic_string ": " basic_integer)* "\n  " "}") | "{}"
main ::= "{" "\n  " "\"array_field\"" ": " main_prop_0 ",\n  " "\"tuple_field\"" ": " main_prop_1 ",\n  " "\"object_field\"" ": " main_prop_2 "\n" "}"
"""

    instance = MainModel(
        array_field=["foo", "bar"],
        tuple_field=("foo", 42, ["bar", "baz"]),
        object_field={"foo": 42, "bar": 43},
    )

    schema = MainModel.model_json_schema()
    check_schema_with_grammar(schema, ebnf_grammar, indent=2)
    check_schema_with_instance(schema, instance, indent=2)
    check_schema_with_instance(schema, instance, indent=None, separators=(",", ":"))


def test_non_strict() -> None:
    class Foo(BaseModel):
        pass

    class MainModel(BaseModel):
        tuple_field: Tuple[str, Tuple[int, int]]
        foo_field: Foo

    ebnf_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= "" | [^"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*) ".0"?
basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub ["]
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= ("[" "" basic_any ("," basic_any)* "" "]") | "[]"
basic_object ::= ("{" "" basic_string ": " basic_any ("," basic_string ": " basic_any)* "" "}") | "{}"
main_prop_0_item_1 ::= "[" "\n      " basic_integer ",\n      " basic_integer (",\n      " basic_any)* "\n    " "]"
main_prop_0 ::= "[" "\n    " basic_string ",\n    " main_prop_0_item_1 (",\n    " basic_any)* "\n  " "]"
main_prop_1 ::= ("{" "\n    " basic_string ": " basic_any (",\n    " basic_string ": " basic_any)* "\n  " "}") | "{}"
main ::= "{" "\n  " "\"tuple_field\"" ": " main_prop_0 ",\n  " "\"foo_field\"" ": " main_prop_1 (",\n  " basic_string ": " basic_any)* "\n" "}"
"""

    instance_json = """{
  "tuple_field": [
    "foo",
    [
      12,
      13,
      "ext"
    ],
    "extra"
  ],
  "foo_field": {
    "tmp": "str"
  },
  "extra": "field"
}"""

    schema = MainModel.model_json_schema()
    check_schema_with_grammar(schema, ebnf_grammar, indent=2, strict_mode=False)
    check_schema_with_json(schema, instance_json, indent=2, strict_mode=False)


def test_enum_const() -> None:
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
basic_array ::= ("[" "" basic_any (", " basic_any)* "" "]") | "[]"
basic_object ::= ("{" "" basic_string ": " basic_any (", " basic_string ": " basic_any)* "" "}") | "{}"
main_prop_0 ::= "\"a\""
main_prop_1 ::= "\"a\\n\\r\\\"\""
main_prop_2 ::= ("\"a\"") | ("\"b\"") | ("\"c\"")
main_prop_3 ::= ("1") | ("\"a\"") | ("true")
main_prop_4 ::= ("\"foo\"") | ("\"bar\"")
main ::= "{" "" "\"bars\"" ": " main_prop_0 ", " "\"str_values\"" ": " main_prop_1 ", " "\"foo\"" ": " main_prop_2 ", " "\"values\"" ": " main_prop_3 ", " "\"field\"" ": " main_prop_4 "" "}"
"""

    schema = MainModel.model_json_schema()
    instance = MainModel(foo="a", values=1, bars="a", str_values='a\n\r"', field=Field.FOO)
    check_schema_with_grammar(schema, ebnf_grammar)
    check_schema_with_instance(schema, instance)


def test_optional() -> None:
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
basic_array ::= ("[" "" basic_any (", " basic_any)* "" "]") | "[]"
basic_object ::= ("{" "" basic_string ": " basic_any (", " basic_string ": " basic_any)* "" "}") | "{}"
main_prop_1 ::= basic_boolean | basic_null
main_prop_2 ::= basic_number | basic_null
main ::= "{" "" ("\"num\"" ": " basic_integer ", ")? ("\"opt_bool\"" ": " main_prop_1 ", ")? "\"size\"" ": " main_prop_2 (", " "\"name\"" ": " basic_string)? "" "}"
"""

    schema = MainModel.model_json_schema()
    check_schema_with_grammar(schema, ebnf_grammar)

    instance = MainModel(num=42, opt_bool=True, size=3.14, name="foo")
    check_schema_with_instance(schema, instance)

    instance = MainModel(size=None)
    check_schema_with_instance(schema, instance)

    check_schema_with_json(schema, '{"size": null}')
    check_schema_with_json(schema, '{"size": null, "name": "foo"}')
    check_schema_with_json(schema, '{"num": 1, "size": null, "name": "foo"}')


def test_all_optional() -> None:
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
basic_array ::= ("[" "" basic_any (", " basic_any)* "" "]") | "[]"
basic_object ::= ("{" "" basic_string ": " basic_any (", " basic_string ": " basic_any)* "" "}") | "{}"
main_part_1 ::= "" | ", " "\"num\"" ": " basic_number ""
main_part_0 ::= main_part_1 | ", " "\"state\"" ": " basic_boolean main_part_1
main ::= ("{" "" (("\"size\"" ": " basic_integer main_part_0) | ("\"state\"" ": " basic_boolean main_part_1) | ("\"num\"" ": " basic_number "")) "" "}") | "{}"
"""

    schema = MainModel.model_json_schema()
    check_schema_with_grammar(schema, ebnf_grammar)

    instance = MainModel(size=42, state=True, num=3.14)
    check_schema_with_instance(schema, instance)

    check_schema_with_json(schema, '{"state": false}')
    check_schema_with_json(schema, '{"size": 1, "num": 1.5}')

    ebnf_grammar_non_strict = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= "" | [^"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*) ".0"?
basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub ["]
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= ("[" "" basic_any (", " basic_any)* "" "]") | "[]"
basic_object ::= ("{" "" basic_string ": " basic_any (", " basic_string ": " basic_any)* "" "}") | "{}"
main_part_2 ::= (", " basic_string ": " basic_any)*
main_part_1 ::= main_part_2 | ", " "\"num\"" ": " basic_number main_part_2
main_part_0 ::= main_part_1 | ", " "\"state\"" ": " basic_boolean main_part_1
main ::= ("{" "" (("\"size\"" ": " basic_integer main_part_0) | ("\"state\"" ": " basic_boolean main_part_1) | ("\"num\"" ": " basic_number main_part_2) | basic_string ": " basic_any main_part_2) "" "}") | "{}"
"""

    check_schema_with_grammar(schema, ebnf_grammar_non_strict, strict_mode=False)

    check_schema_with_json(schema, '{"size": 1, "num": 1.5, "other": false}', strict_mode=False)
    check_schema_with_json(schema, '{"other": false}', strict_mode=False)


def test_empty() -> None:
    class MainModel(BaseModel):
        pass

    ebnf_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= "" | [^"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*) ".0"?
basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub ["]
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= ("[" "" basic_any (", " basic_any)* "" "]") | "[]"
basic_object ::= ("{" "" basic_string ": " basic_any (", " basic_string ": " basic_any)* "" "}") | "{}"
main ::= "{" "}"
"""

    schema = MainModel.model_json_schema()
    check_schema_with_grammar(schema, ebnf_grammar)

    instance = MainModel()
    check_schema_with_instance(schema, instance)

    check_schema_with_json(schema, '{"tmp": 123}', strict_mode=False)


def test_reference() -> None:
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

    ebnf_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= "" | [^"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*) ".0"?
basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub ["]
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= ("[" "" basic_any (", " basic_any)* "" "]") | "[]"
basic_object ::= ("{" "" basic_string ": " basic_any (", " basic_string ": " basic_any)* "" "}") | "{}"
main_prop_0_prop_1 ::= basic_number | basic_null
main_prop_0 ::= "{" "" "\"count\"" ": " basic_integer (", " "\"size\"" ": " main_prop_0_prop_1)? "" "}"
main_prop_1_items_part_0 ::= "" | ", " "\"banana\"" ": " basic_string ""
main_prop_1_items ::= ("{" "" (("\"apple\"" ": " basic_string main_prop_1_items_part_0) | ("\"banana\"" ": " basic_string "")) "" "}") | "{}"
main_prop_1 ::= ("[" "" main_prop_1_items (", " main_prop_1_items)* "" "]") | "[]"
main ::= "{" "" "\"foo\"" ": " main_prop_0 ", " "\"bars\"" ": " main_prop_1 "" "}"
"""

    schema = MainModel.model_json_schema()
    check_schema_with_grammar(schema, ebnf_grammar)
    check_schema_with_instance(schema, instance)


def test_union() -> None:
    class Cat(BaseModel):
        name: str
        color: str

    class Dog(BaseModel):
        name: str
        breed: str

    ta = TypeAdapter(Union[Cat, Dog])

    model_schema = ta.json_schema()

    ebnf_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= "" | [^"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*) ".0"?
basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub ["]
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= ("[" "" basic_any (", " basic_any)* "" "]") | "[]"
basic_object ::= ("{" "" basic_string ": " basic_any (", " basic_string ": " basic_any)* "" "}") | "{}"
main_case_0 ::= "{" "" "\"name\"" ": " basic_string ", " "\"color\"" ": " basic_string "" "}"
main_case_1 ::= "{" "" "\"name\"" ": " basic_string ", " "\"breed\"" ": " basic_string "" "}"
main ::= main_case_0 | main_case_1
"""

    check_schema_with_grammar(model_schema, ebnf_grammar)

    check_schema_with_instance(model_schema, Cat(name="kitty", color="black"))
    check_schema_with_instance(model_schema, Dog(name="doggy", breed="bulldog"))
    check_schema_with_json(model_schema, '{"name": "kitty", "test": "black"}', False)


def test_alias() -> None:
    class MainModel(BaseModel):
        test: str = Field(..., alias="name")

    ebnf_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= "" | [^"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*) ".0"?
basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub ["]
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= ("[" "" basic_any (", " basic_any)* "" "]") | "[]"
basic_object ::= ("{" "" basic_string ": " basic_any (", " basic_string ": " basic_any)* "" "}") | "{}"
main ::= "{" "" "\"name\"" ": " basic_string "" "}"
"""

    check_schema_with_grammar(MainModel.model_json_schema(), ebnf_grammar)

    instance = MainModel(name="kitty")
    instance_str = json.dumps(instance.model_dump(mode="json", round_trip=True, by_alias=False))
    check_schema_with_json(MainModel.model_json_schema(by_alias=False), instance_str)

    instance_str = json.dumps(instance.model_dump(mode="json", round_trip=True, by_alias=True))
    check_schema_with_json(MainModel.model_json_schema(by_alias=True), instance_str)

    # property name contains space
    class MainModelSpace(BaseModel):
        test: Literal["abc"] = Field(..., alias="name 1")

    ebnf_grammar_space = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= "" | [^"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*) ".0"?
basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub ["]
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= ("[" "" basic_any (", " basic_any)* "" "]") | "[]"
basic_object ::= ("{" "" basic_string ": " basic_any (", " basic_string ": " basic_any)* "" "}") | "{}"
main_prop_0 ::= "\"abc\""
main ::= "{" "" "\"name 1\"" ": " main_prop_0 "" "}"
"""

    check_schema_with_grammar(MainModelSpace.model_json_schema(), ebnf_grammar_space)

    instance_space = MainModelSpace(**{"name 1": "abc"})
    instance_space_str = json.dumps(
        instance_space.model_dump(mode="json", round_trip=True, by_alias=True)
    )
    check_schema_with_json(MainModelSpace.model_json_schema(by_alias=True), instance_space_str)


if __name__ == "__main__":
    tvm.testing.main()
