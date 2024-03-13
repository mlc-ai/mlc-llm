import json
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import tvm.testing
from pydantic import BaseModel, TypeAdapter

from mlc_chat.serve.grammar import BNFGrammar, GrammarStateMatcher
from mlc_chat.serve.json_schema_converter import JSONSchemaConverter


def check_schema_with_grammar(schema: Dict[str, Any], grammar: str):
    schema_str = json.dumps(schema, indent=2)
    bnf_grammar_str = JSONSchemaConverter.to_ebnf(schema_str, separators=(",", ":"))
    assert bnf_grammar_str == grammar


def check_schema_with_data(schema: Dict[str, Any], data: str, accepted=True):
    schema_str = json.dumps(schema, indent=2)

    bnf_grammar_str = JSONSchemaConverter.to_ebnf(schema_str, separators=(",", ":"))
    bnf_grammar = BNFGrammar.from_ebnf_string(bnf_grammar_str)
    matcher = GrammarStateMatcher(bnf_grammar)

    if accepted:
        assert matcher.debug_match_complete_string(data)
    else:
        assert not matcher.debug_match_complete_string(data)


def check_schema_with_instance(schema: Dict[str, Any], instance: BaseModel):
    check_schema_with_data(schema, instance.model_dump_json(round_trip=True))


def test_basic():
    class MainModel(BaseModel):
        integer_field: int
        number_field: float
        boolean_field: bool
        array_field: List[str]
        tuple_field: Tuple[str, int, List[str]]
        object_field: Dict[str, int]
        nested_object_field: Dict[str, Dict[str, int]]

    instance = MainModel(
        integer_field=42,
        number_field=3.14,
        boolean_field=True,
        array_field=["foo", "bar"],
        tuple_field=("foo", 42, ["bar", "baz"]),
        object_field={"foo": 42, "bar": 43},
        nested_object_field={"foo": {"bar": 42}},
    )

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

    instance = MainModel(foo="a", values=1, bars="a", str_values='a\n\r"', field=Field.FOO)
    check_schema_with_instance(MainModel.model_json_schema(), instance)


def test_optional():
    class MainModel(BaseModel):
        size: Optional[float] = None

    instance = MainModel(size=3.14)
    check_schema_with_instance(MainModel.model_json_schema(), instance)

    instance = MainModel()
    check_schema_with_instance(MainModel.model_json_schema(), instance)

    check_schema_with_data(MainModel.model_json_schema(), "{}")


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
    check_schema_with_data(model_schema, '{"name":"kitty","test":"black"}', False)


def test_alias():
    

if __name__ == "__main__":
    tvm.testing.main()
