import json
from enum import Enum
from typing import Dict, List, Tuple, Union
from mlc_chat.serve.grammar import BNFGrammar
from mlc_chat.serve.json_schema_converter import JSONSchemaToBNF
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict


def test_simple():
    class MainModel(BaseModel):
        integer_field: int
        number_field: float
        boolean_field: bool
        array_field: List[str]
        tuple_field: Tuple[str, int, List[str]]
        object_field: Dict[str, int]
        nested_object_field: Dict[str, Dict[str, int]]

    main_model_schema = MainModel.model_json_schema()
    main_model_schema_str = json.dumps(main_model_schema, indent=2)
    print(main_model_schema_str)

    instance = MainModel(
        integer_field=42,
        number_field=3.14,
        boolean_field=True,
        array_field=["foo", "bar"],
        tuple_field=("foo", 42, ["bar", "baz"]),
        object_field={"foo": 42, "bar": 43},
        nested_object_field={"foo": {"bar": 42}},
    )
    instance_json = instance.model_dump_json(round_trip=True)
    print(instance_json)

    bnf_grammar = JSONSchemaToBNF.to_bnf(main_model_schema_str, separators=(",", ":"))
    print(bnf_grammar)

    print(BNFGrammar.from_ebnf_string(bnf_grammar))



if __name__ == "__main__":
    test_simple()
