import json
import logging
from typing import Any, Dict, List, Tuple, Union


SchemaType = Union[Dict[str, Any], bool]


class IndentManager:
    def __init__(self, indent: Union[int, None], separator: str):
        self.indent = indent or 0
        self.separator = separator
        self.total_indent = 0
        self.is_first = [True]

    def __enter__(self):
        self.total_indent += self.indent
        self.is_first.append(True)

    def __exit__(self, exc_type, exc_value, traceback):
        self.total_indent -= self.indent
        self.is_first.pop()

    def get_sep(self) -> str:
        res = self.total_indent * " "
        if not self.is_first[-1]:
            res += self.separator
        self.is_first[-1] = False
        return f'"{res}"'


class JSONSchemaToEBNFConverter:
    def __init__(
        self,
        json_schema: SchemaType,
        indent: Union[int, None] = None,
        separators: Union[Tuple[str, str], None] = None,
        strict_mode: bool = False,
    ):
        self.json_schema = json_schema
        self.indent = indent
        self.strict_mode = strict_mode

        if separators is None:
            separators = (", ", ": ") if indent is None else (",", ": ")
        else:
            assert len(separators) == 2
        self.indent_manager = IndentManager(indent, separators[0])
        self.colon = separators[1]

        self.rules: List[Tuple[str, str]] = []
        self.basic_rules_cache: Dict[str, str] = {}
        self.init_basic_rules()

    def convert(self) -> str:
        self.create_rule_with_schema("main", self.json_schema)
        res = ""
        for rule_name, rule in self.rules:
            res += f"{rule_name} ::= {rule}\n"
        return res

    # Basic rules
    BASIC_INTEGER = "basic_integer"
    BASIC_NUMBER = "basic_number"
    BASIC_STRING = "basic_string"
    BASIC_BOOLEAN = "basic_boolean"
    BASIC_NULL = "basic_null"

    # Helper rules to construct basic rules
    BASIC_ESCAPE = "basic_escape"
    BASIC_STRING_SUB = "basic_string_sub"

    def init_basic_rules(self):
        past_strict_mode = self.strict_mode
        self.strict_mode = False

        self.init_helper_rules()
        self.create_basic_rule(self.BASIC_INTEGER, {"type": "integer"})
        self.create_basic_rule(self.BASIC_NUMBER, {"type": "number"})
        self.create_basic_rule(self.BASIC_STRING, {"type": "string"})
        self.create_basic_rule(self.BASIC_BOOLEAN, {"type": "boolean"})
        self.create_basic_rule(self.BASIC_NULL, {"type": "null"})

        self.strict_mode = past_strict_mode

    def init_helper_rules(self):
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

    def create_basic_rule(self, name: str, schema: SchemaType) -> str:
        rule_name = self.create_rule_with_schema(name, schema)
        self.basic_rules_cache[self.get_schema_cache_index(schema)] = rule_name

    def get_sep(self):
        return self.indent_manager.get_sep()

    @staticmethod
    def warn_unsupported_keywords(schema: SchemaType, keywords: Union[str, List[str]]):
        if isinstance(keywords, str):
            keywords = [keywords]
        for keyword in keywords:
            if keyword in schema:
                logging.warning(f"Keyword {keyword} is not supported in schema {schema}")

    def create_rule_with_schema(self, rule_name_hint: str, schema: SchemaType) -> str:
        idx = self.get_schema_cache_index(schema)
        if idx in self.basic_rules_cache:
            return self.basic_rules_cache[idx]

        self.rules.append((rule_name_hint, self.visit_schema(schema, rule_name_hint)))
        return rule_name_hint

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
                k: JSONSchemaToEBNFConverter.remove_skipped_keys_recursive(v)
                for k, v in obj.items()
                if k not in JSONSchemaToEBNFConverter.SKIPPED_KEYS
            }
        elif isinstance(obj, list):
            return [JSONSchemaToEBNFConverter.remove_skipped_keys_recursive(v) for v in obj]
        else:
            return obj

    def get_schema_cache_index(self, schema: SchemaType) -> str:
        return json.dumps(
            JSONSchemaToEBNFConverter.remove_skipped_keys_recursive(schema),
            sort_keys=True,
            indent=None,
        )

    def visit_schema(self, schema: SchemaType, rule_name: str) -> str:
        assert schema is not False
        if schema is True:
            return self.visit_any(schema, rule_name)

        JSONSchemaToEBNFConverter.warn_unsupported_keywords(
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
            res += self.create_rule_with_schema(f"{rule_name}_{i}", anyof_schema)
        return res

    def visit_any(self, schema: SchemaType, rule_name: str) -> str:
        # note integer is a subset of number, so we don't need to add integer here
        return self.visit_schema(
            {
                "anyOf": [
                    {"type": "number"},
                    {"type": "string"},
                    {"type": "boolean"},
                    {"type": "null"},
                    {"type": "array"},
                    {"type": "object"},
                ]
            },
            rule_name,
        )

    def visit_integer(self, schema: SchemaType, rule_name: str) -> str:
        assert schema["type"] == "integer"
        JSONSchemaToEBNFConverter.warn_unsupported_keywords(
            schema, ["multipleOf", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]
        )
        return '("0" | "-"? [1-9] [0-9]*) ".0"?'

    def visit_number(self, schema: SchemaType, rule_name: str) -> str:
        assert schema["type"] == "number"
        JSONSchemaToEBNFConverter.warn_unsupported_keywords(
            schema, ["multipleOf", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]
        )
        return '("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?'

    def visit_string(self, schema: SchemaType, rule_name: str) -> str:
        assert schema["type"] == "string"
        JSONSchemaToEBNFConverter.warn_unsupported_keywords(
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
        JSONSchemaToEBNFConverter.warn_unsupported_keywords(
            schema,
            ["uniqueItems", "contains", "minContains", "maxContains", "minItems", "maxItems"],
        )

        res = '"["'

        with self.indent_manager:
            if "prefixItems" in schema:
                for i, prefix_item in enumerate(schema["prefixItems"]):
                    assert prefix_item is not False
                    res += " " + self.get_partial_rule_for_item(prefix_item, f"{rule_name}_{i}")

            items = schema.get("items", False)
            if items is not False:
                res += " " + self.get_partial_rule_for_repetitive_item(
                    schema["items"], f"{rule_name}_item"
                )

            unevaluated = schema.get("unevaluatedItems", not self.strict_mode)
            # if items is in the schema, we don't need to consider unevaluatedItems
            if "items" not in schema and unevaluated is not False:
                rest_schema = schema.get("unevaluatedItems", True)
                res += " " + self.get_partial_rule_for_repetitive_item(
                    rest_schema, f"{rule_name}_rest"
                )

        res += ' "]"'
        return res

    def get_partial_rule_for_item(self, schema: SchemaType, sub_rule_name: str) -> str:
        item = self.create_rule_with_schema(sub_rule_name, schema)
        return f"{self.get_sep()} {item}"

    def get_partial_rule_for_repetitive_item(self, schema: SchemaType, sub_rule_name: str) -> str:
        item = self.create_rule_with_schema(sub_rule_name, schema)
        return f'("" | {self.get_sep()} {item} ({self.get_sep()} {item})*)'

    def visit_object(self, schema: SchemaType, rule_name: str) -> str:
        assert schema["type"] == "object"
        JSONSchemaToEBNFConverter.warn_unsupported_keywords(
            schema, ["patternProperties", "minProperties", "maxProperties", "propertyNames"]
        )

        res = '"{"'
        # Now we only consider the required list for the properties field
        required = schema.get("required", [])

        with self.indent_manager:
            if "properties" in schema:
                properties = list(schema["properties"].items())

                first_required_idx = len(properties)
                for i, (prop_name, _) in enumerate(properties):
                    if prop_name in required:
                        first_required_idx = i
                        break

                if first_required_idx != 0:
                    res += " " + self.get_partial_rule_for_properties_all_optional(
                        properties[:first_required_idx], rule_name
                    )

                for prop_name, prop_schema in properties[first_required_idx:]:
                    assert prop_schema is not False
                    res += " " + self.get_partial_rule_for_property_may_required(
                        prop_name,
                        prop_schema,
                        sub_rule_name=f"{rule_name}_{prop_name}",
                        required=prop_name in required,
                    )

            additional = schema.get("additionalProperties", False)
            if additional is not False:
                res += " " + self.get_partial_rule_for_property_repetitive(
                    "basic_string",
                    schema["additionalProperties"],
                    sub_rule_name=rule_name + "_add",
                )

            unevaluated = schema.get("unevaluatedProperties", not self.strict_mode)
            # if additionalProperties is in the schema, we don't need to consider
            # unevaluatedProperties
            if "additionalProperties" not in schema and unevaluated is not False:
                res += " " + self.get_partial_rule_for_property_repetitive(
                    "basic_string",
                    unevaluated,
                    sub_rule_name=f"{rule_name}_uneval",
                )

        res += ' "}"'
        return res

    def get_partial_rule_for_properties_all_optional(
        self, properties: List[Tuple[str, SchemaType]], rule_name: str
    ) -> str:
        assert len(properties) >= 1
        colon = f'"{self.colon}"'
        # ("," (("a" ("" | "," b) | b) ("" | "," c) | c)) | ""

        first_sep = self.get_sep()

        res = ""

        for i, (prop_name, prop_schema) in enumerate(properties):
            key = f'"\\"{prop_name}\\""'
            value = self.create_rule_with_schema(f"{rule_name}_{prop_name}", prop_schema)
            key_value = f"{key} {colon} {value}"

            if i == 0:
                res = key_value
            else:
                res = f'({res}) ("" | {self.get_sep()} {key_value}) | {key_value}'

        res = f'{first_sep} ({res}) | ""'
        return res

    def get_partial_rule_for_property_may_required(
        self, prop_name: str, schema: str, sub_rule_name: str, required: bool
    ) -> str:
        colon = f'"{self.colon}"'
        # the outer quote is for the string in EBNF grammar, and the inner quote is for
        # the string in JSON
        key = f'"\\"{prop_name}\\""'
        value = self.create_rule_with_schema(sub_rule_name, schema)
        res = f"{self.get_sep()} {key} {colon} {value}"
        if not required:
            res = f"({res})?"
        return res

    def get_partial_rule_for_property_repetitive(
        self, key: str, schema: str, sub_rule_name: str
    ) -> str:
        colon = f'"{self.colon}"'
        value = self.create_rule_with_schema(sub_rule_name, schema)
        key_value = f"{key} {colon} {value}"

        return f'("" | {self.get_sep()} {key_value} ({self.get_sep()} {key_value})*)'


def json_schema_to_ebnf(
    json_schema: str,
    *,
    indent: Union[int, None] = None,
    separators: Union[Tuple[str, str], None] = None,
    strict_mode: bool = False,
) -> str:
    json_schema_schema = json.loads(json_schema)
    return JSONSchemaToEBNFConverter(json_schema_schema, indent, separators, strict_mode).convert()
