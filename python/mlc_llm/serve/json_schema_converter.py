import json
import logging
from typing import Any, Dict, List, Tuple, Union


SchemaType = Union[Dict[str, Any], bool]


class JSONSchemaConverter:
    @staticmethod
    def to_ebnf(
        json_schema: str,
        *,
        indent: Union[int, None] = None,
        separators: Union[Tuple[str, str], None] = None,
    ) -> str:
        json_schema_schema = json.loads(json_schema)
        return JSONSchemaConverter(json_schema_schema, indent, separators).convert()

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

    def convert(self) -> str:
        self.schema_to_rule(self.json_schema, "main")
        res = ""
        for rule_name, rule in self.rules:
            res += f"{rule_name} ::= {rule}\n"
        return res

    # Basic rules
    BASIC_ANY = "basic_any"
    BASIC_INTEGER = "basic_integer"
    BASIC_NUMBER = "basic_number"
    BASIC_STRING = "basic_string"
    BASIC_BOOLEAN = "basic_boolean"
    BASIC_NULL = "basic_null"
    BASIC_ARRAY = "basic_array"
    BASIC_OBJECT = "basic_object"

    # Helper rules to construct basic rules
    BASIC_DIGITS = "basic_digits"
    BASIC_DECIMAL = "basic_decimal"
    BASIC_EXP = "basic_exp"
    BASIC_WS = "basic_ws"
    BASIC_ESCAPE = "basic_escape"
    BASIC_STRING_SUB = "basic_string_sub"

    def init_basic_rules(self):
        self.init_helper_rules()
        self.schema_to_rule(True, self.BASIC_ANY)
        self.schema_to_rule({"type": "integer"}, self.BASIC_INTEGER)
        self.schema_to_rule({"type": "number"}, self.BASIC_NUMBER)
        self.schema_to_rule({"type": "string"}, self.BASIC_STRING)
        self.schema_to_rule({"type": "boolean"}, self.BASIC_BOOLEAN)
        self.schema_to_rule({"type": "null"}, self.BASIC_NULL)
        self.schema_to_rule({"type": "array"}, self.BASIC_ARRAY)
        self.schema_to_rule({"type": "object"}, self.BASIC_OBJECT)

    def init_helper_rules(self):
        self.rules.append((self.BASIC_DIGITS, "[0-9]*"))
        self.rules.append((self.BASIC_DECIMAL, f'"" | "." [0-9] {self.BASIC_DIGITS}'))
        self.rules.append(
            (
                self.BASIC_EXP,
                f'"" | [eE] [0-9] {self.BASIC_DIGITS} | [eE] [+-] [0-9] {self.BASIC_DIGITS}',
            )
        )
        self.rules.append((self.BASIC_WS, "[ \\n\\t]*"))
        self.rules.append(
            (
                self.BASIC_ESCAPE,
                '["\\\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]',
            )
        )
        self.rules.append(
            (
                self.BASIC_STRING_SUB,
                f'"" | [^"\\\\\\r\\n] {self.BASIC_STRING_SUB} | '
                f'"\\\\" {self.BASIC_ESCAPE} {self.BASIC_STRING_SUB}',
            )
        )

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
        self.rules.append((rule_name, self.visit_schema(schema, rule_name)))
        return rule_name

    SKIPPED_KEYS = [
        "title",
        "default",
        "description",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        "$comment",
        "$schema",
    ]

    @staticmethod
    def remove_skipped_keys_recursive(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: JSONSchemaConverter.remove_skipped_keys_recursive(v)
                for k, v in obj.items()
                if k not in JSONSchemaConverter.SKIPPED_KEYS
            }
        elif isinstance(obj, list):
            return [JSONSchemaConverter.remove_skipped_keys_recursive(v) for v in obj]
        else:
            return obj

    def schema_to_str_key(self, schema: SchemaType) -> str:
        return json.dumps(
            JSONSchemaConverter.remove_skipped_keys_recursive(schema), sort_keys=True, indent=None
        )

    def visit_schema(self, schema: SchemaType, rule_name: str) -> str:
        assert schema is not False
        if schema is True:
            return self.visit_any(schema, rule_name)

        JSONSchemaConverter.warn_unsupported_keywords(
            schema,
            [
                "allof",
                "oneof",
                "not",
                "if",
                "then",
                "else",
                "dependentRequired",
                "dependentSchemas",
            ],
        )

        if "$ref" in schema:
            return self.visit_ref(schema, rule_name)
        elif "const" in schema:
            return self.visit_const(schema, rule_name)
        elif "enum" in schema:
            return self.visit_enum(schema, rule_name)
        elif "anyOf" in schema:
            return self.visit_anyof(schema, rule_name)
        elif "type" in schema:
            match schema["type"]:
                case "integer":
                    return self.visit_integer(schema, rule_name)
                case "number":
                    return self.visit_number(schema, rule_name)
                case "string":
                    return self.visit_string(schema, rule_name)
                case "boolean":
                    return self.visit_boolean(schema, rule_name)
                case "null":
                    return self.visit_null(schema, rule_name)
                case "array":
                    return self.visit_array(schema, rule_name)
                case "object":
                    return self.visit_object(schema, rule_name)
                case _:
                    raise ValueError(f"Unsupported type {schema['type']}")
        else:
            # no keyword is detected, we treat it as any
            return self.visit_any(schema, rule_name)

    def visit_ref(self, schema: SchemaType, rule_name: str) -> str:
        assert "$ref" in schema
        new_schema = self.uri_to_schema(schema["$ref"]).copy()
        if not (isinstance(new_schema, bool)):
            new_schema.update({k: v for k, v in schema.items() if k != "$ref"})
        return self.visit_schema(new_schema, rule_name)

    def uri_to_schema(self, uri: str) -> SchemaType:
        if uri.startswith("#/$defs/"):
            return self.json_schema["$defs"][uri[len("#/$defs/") :]]
        else:
            logging.warning(f"Now only support URI starting with '#/$defs/' but got {uri}")
            return True

    def visit_const(self, schema: SchemaType, rule_name: str) -> str:
        assert "const" in schema
        return '"' + self.json_str_to_printable_str(json.dumps(schema["const"])) + '"'

    def visit_enum(self, schema: SchemaType, rule_name: str) -> str:
        assert "enum" in schema
        res = ""
        for i, enum_value in enumerate(schema["enum"]):
            if i != 0:
                res += " | "
            res += '("' + self.json_str_to_printable_str(json.dumps(enum_value)) + '")'
        return res

    REPLACE_MAPPING = {
        "\\": "\\\\",
        '"': '\\"',
    }

    def json_str_to_printable_str(self, json_str: str) -> str:
        for k, v in self.REPLACE_MAPPING.items():
            json_str = json_str.replace(k, v)
        return json_str

    def visit_anyof(self, schema: SchemaType, rule_name: str) -> str:
        assert "anyOf" in schema
        res = ""
        for i, anyof_schema in enumerate(schema["anyOf"]):
            if i != 0:
                res += " | "
            res += self.schema_to_rule(anyof_schema, f"{rule_name}_{i}")
        return res

    def visit_any(self, schema: SchemaType, rule_name: str) -> str:
        # note integer is included in number
        return (
            f"{self.BASIC_NUMBER} | {self.BASIC_STRING} | {self.BASIC_BOOLEAN} | "
            f"{self.BASIC_NULL} | {self.BASIC_ARRAY} | {self.BASIC_OBJECT}"
        )

    def visit_integer(self, schema: SchemaType, rule_name: str) -> str:
        assert schema["type"] == "integer"
        JSONSchemaConverter.warn_unsupported_keywords(
            schema, ["multipleOf", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]
        )
        return '("0" | "-"? [1-9] [0-9]*) ".0"?'

    def visit_number(self, schema: SchemaType, rule_name: str) -> str:
        assert schema["type"] == "number"
        JSONSchemaConverter.warn_unsupported_keywords(
            schema, ["multipleOf", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]
        )
        return '("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?'

    def visit_string(self, schema: SchemaType, rule_name: str) -> str:
        assert schema["type"] == "string"
        JSONSchemaConverter.warn_unsupported_keywords(
            schema, ["minLength", "maxLength", "pattern", "format"]
        )
        return f'["] {self.BASIC_STRING_SUB} ["]'

    def visit_boolean(self, schema: SchemaType, rule_name: str) -> str:
        assert schema["type"] == "boolean"

        return '"true" | "false"'

    def visit_null(self, schema: SchemaType, rule_name: str) -> str:
        assert schema["type"] == "null"

        return '"null"'

    def visit_array(self, schema: SchemaType, rule_name: str) -> str:
        assert schema["type"] == "array"
        JSONSchemaConverter.warn_unsupported_keywords(
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

    def visit_object(self, schema: SchemaType, rule_name: str) -> str:
        assert schema["type"] == "object"
        JSONSchemaConverter.warn_unsupported_keywords(
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
                    res += f'{additional_separator} "\\"{prop_name}\\"" {colon} {prop_rule_name}'
                else:
                    res += f'({additional_separator} "\\"{prop_name}\\"" {colon} {prop_rule_name})?'

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
