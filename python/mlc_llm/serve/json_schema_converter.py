import json
import logging
from typing import Any, Dict, List, Tuple, Union

from numpy import sort


SchemaType = Union[Dict[str, Any], bool]


class JSONSchemaToBNF:
    @staticmethod
    def vto_bnf(
        json_schema: str,
        *,
        indent: Union[int, None] = None,
        separators: Union[Tuple[str, str], None] = None,
    ) -> str:
        json_schema_schema = json.loads(json_schema)
        return JSONSchemaToBNF(json_schema_schema, indent, separators).get_bnf_grammar()

    def __init__(
        self,
        json_schema: SchemaType,
        indent: Union[int, None] = None,
        separators: Union[Tuple[str, str], None] = None,
    ):
        # todo: allow_additional_properties
        # todo: allow_additional_items
        self.json_schema = json_schema
        self.indent = indent

        if separators is None:
            separators = (", ", ": ") if indent is None else (",", ": ")
        else:
            assert len(separators) == 2
        self.separators = separators

        self.rules: List[Tuple[str, str]] = []
        self.cache_schema_to_rule: Dict[str, str] = {}
        self.init_basic_rules()

    def get_bnf_grammar(self) -> str:
        self.schema_to_rule(self.json_schema, "main")
        res = ""
        for rule_name, rule in self.rules:
            res += f"{rule_name} ::= {rule}\n"
        return res

    def init_basic_rules(self):
        self.schema_to_rule(True, "basic_any")
        self.schema_to_rule({"type": "integer"}, "basic_integer")
        self.schema_to_rule({"type": "number"}, "basic_number")
        self.schema_to_rule({"type": "string"}, "basic_string")
        self.schema_to_rule({"type": "boolean"}, "basic_boolean")
        self.schema_to_rule({"type": "null"}, "basic_null")
        self.schema_to_rule({"type": "array"}, "basic_array")
        self.schema_to_rule({"type": "object"}, "basic_object")

    @staticmethod
    def warn_unsupported_keywords(schema: SchemaType, keywords: Union[str, List[str]]):
        if isinstance(keywords, str):
            keywords = [keywords]
        for keyword in keywords:
            if keyword in schema:
                # todo: test and output format
                logging.warning(f"Keyword {keyword} is not supported in schema {schema}")

    def schema_to_rule(self, schema: SchemaType, rule_name: str) -> str:
        schema_key = self.schema_to_str_key(schema)
        if schema_key in self.cache_schema_to_rule:
            return self.cache_schema_to_rule[schema_key]

        self.cache_schema_to_rule[schema_key] = rule_name
        self.rules.append((rule_name, self.convert(schema, rule_name)))
        return rule_name

    SKIPPED_KEYS = [
        "title",
        "description",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        "$comment",
    ]

    @staticmethod
    def remove_skipped_keys_recursive(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: JSONSchemaToBNF.remove_skipped_keys_recursive(v)
                for k, v in obj.items()
                if k not in JSONSchemaToBNF.SKIPPED_KEYS
            }
        elif isinstance(obj, list):
            return [JSONSchemaToBNF.remove_skipped_keys_recursive(v) for v in obj]
        else:
            return obj

    def schema_to_str_key(self, schema: SchemaType) -> str:
        return json.dumps(
            JSONSchemaToBNF.remove_skipped_keys_recursive(schema), sort_keys=True, indent=None
        )

    def convert(self, schema: SchemaType, rule_name: str) -> str:
        assert schema is not False
        if schema is True:
            return self.convert_any(schema, rule_name)
        if "type" in schema:
            match schema["type"]:
                case "integer":
                    return self.convert_integer(schema, rule_name)
                case "number":
                    return self.convert_number(schema, rule_name)
                case "string":
                    return self.convert_string(schema, rule_name)
                case "boolean":
                    return self.convert_boolean(schema, rule_name)
                case "null":
                    return self.convert_null(schema, rule_name)
                case "array":
                    return self.convert_array(schema, rule_name)
                case "object":
                    return self.convert_object(schema, rule_name)
                case _:
                    raise ValueError(f"Unsupported type {schema['type']}")
        else:
            return self.convert_union(schema, rule_name)

    def convert_any(self, schema: SchemaType, rule_name: str) -> str:
        return "basic_integer | basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object"

    def convert_union(self, schema: SchemaType, rule_name: str) -> str:
        return ""

    def convert_integer(self, schema: SchemaType, rule_name: str) -> str:
        assert schema["type"] == "integer"
        JSONSchemaToBNF.warn_unsupported_keywords(
            schema, ["multipleOf", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]
        )
        return '("0" | "-"? [1-9] [0-9]*) ".0"?'

    def convert_number(self, schema: SchemaType, rule_name: str) -> str:
        assert schema["type"] == "number"
        JSONSchemaToBNF.warn_unsupported_keywords(
            schema, ["multipleOf", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]
        )
        return '("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?'

    def convert_string(self, schema: SchemaType, rule_name: str) -> str:
        assert schema["type"] == "string"
        JSONSchemaToBNF.warn_unsupported_keywords(
            schema, ["minLength", "maxLength", "pattern", "format"]
        )

        return (
            '"\\"" ([^"\\\\\\u0000-\\u001F] | "\\\\" ([\\\\"/bfnrt] | '
            '"u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\\""'
        )

    def convert_boolean(self, schema: SchemaType, rule_name: str) -> str:
        assert schema["type"] == "boolean"

        return '"true" | "false"'

    def convert_null(self, schema: SchemaType, rule_name: str) -> str:
        assert schema["type"] == "null"

        return '"null"'

    def convert_array(self, schema: SchemaType, rule_name: str) -> str:
        assert schema["type"] == "array"
        JSONSchemaToBNF.warn_unsupported_keywords(
            schema,
            ["uniqueItems", "contains", "minContains", "maxContains", "minItems", "maxItems"],
        )

        res = '"["'

        separator = f'"{self.separators[0]}"'

        if "prefixItems" in schema:
            for i, prefix_item in enumerate(schema["prefixItems"]):
                assert prefix_item is not False
                if i != 0:
                    res += separator
                res += self.schema_to_rule(prefix_item, f"{rule_name}_{i}")

        if "items" in schema and schema["items"] is not False:
            item_rule_name = self.schema_to_rule(schema["items"], f"{rule_name}__item")
            additional_separator = separator if res != '"["' else ""
            res += f'("" | {additional_separator} {item_rule_name} ({separator} {item_rule_name})*)'

        disallow_other_items = "items" in schema or (
            "unevaluatedItems" in schema and schema["unevaluatedItems"] is False
        )

        if not disallow_other_items:
            rest_schema = schema.get("unevaluatedItems", True)
            rest_schema_rule = self.schema_to_rule(rest_schema, f"{rule_name}_rest")
            additional_separator = separator if res != '"["' else ""
            res += f'("" | {additional_separator} {rest_schema_rule} ({separator} {rest_schema_rule})*)'

        res += '"]"'
        return res

    def convert_object(self, schema: SchemaType, rule_name: str) -> str:
        assert schema["type"] == "object"
        JSONSchemaToBNF.warn_unsupported_keywords(
            schema, ["patternProperties", "minProperties", "maxProperties", "propertyNames"]
        )

        res = '"{"'

        separator = f'"{self.separators[0]}"'
        colon = f'"{self.separators[1]}"'

        # Now we only consider the required list for the properties field
        required = schema.get("required", None)

        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                assert prop_schema is not False
                prop_rule_name = self.schema_to_rule(prop_schema, f"{rule_name}_{prop_name}")
                additional_separator = separator if res != '"{"' else ""
                if required is not None and prop_name in required:
                    res += f'{additional_separator} "{prop_name}" {colon} {prop_rule_name}'
                else:
                    res += f'{additional_separator} ("{prop_name}" {colon} {prop_rule_name})?'

        if "additionalProperties" in schema and schema["additionalProperties"] is not False:
            additional_properties_rule = self.schema_to_rule(
                schema["additionalProperties"], f"{rule_name}__addi"
            )
            property_with_name = f"basic_string {colon} {additional_properties_rule}"

            if res != '"{"':
                res += f"({separator} {property_with_name})*"
            else:
                res += f'("" | {property_with_name} ({separator} {property_with_name})*)'

        disallow_other_items = "additionalProperties" in schema or (
            "unevaluatedProperties" in schema and schema["unevaluatedProperties"] is False
        )

        if not disallow_other_items:
            uneval_schema = schema.get("unevaluatedProperties", True)
            unevaluated_properties_rule = self.schema_to_rule(uneval_schema, f"{rule_name}__uneval")
            property_with_name = f"basic_string {colon} {unevaluated_properties_rule}"

            if res != '"{"':
                res += f"({separator} {property_with_name})*"
            else:
                res += f'("" | {property_with_name} ({separator} {property_with_name})*)'

        res += '"}"'
        return res
