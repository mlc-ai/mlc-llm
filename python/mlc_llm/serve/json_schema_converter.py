import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union


SchemaType = Union[Dict[str, Any], bool]


class IndentManager:
    def __init__(self, indent: Union[int, None], separator: str):
        self.enable_newline = indent is not None
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
        res = ""
        if self.enable_newline:
            res += "\\n"
        res += self.total_indent * " "
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
        self.strict_mode = strict_mode

        if separators is None:
            separators = (", ", ": ") if indent is None else (",", ": ")
        assert len(separators) == 2
        self.indent_manager = IndentManager(indent, separators[0])
        self.colon = separators[1]

        self.rules: List[Tuple[str, str]] = []
        self.basic_rules_cache: Dict[str, str] = {}
        self.add_basic_rules()

    def convert(self) -> str:
        self.create_rule_with_schema(self.json_schema, "main")
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
    BASIC_ESCAPE = "basic_escape"
    BASIC_STRING_SUB = "basic_string_sub"

    def add_basic_rules(self):
        past_strict_mode = self.strict_mode
        self.strict_mode = False
        past_indent_manager = self.indent_manager
        self.indent_manager = IndentManager(None, past_indent_manager.separator)

        self.add_helper_rules()
        self.create_basic_rule(True, self.BASIC_ANY)
        self.basic_rules_cache[self.get_schema_cache_index({})] = self.BASIC_ANY
        self.create_basic_rule({"type": "integer"}, self.BASIC_INTEGER)
        self.create_basic_rule({"type": "number"}, self.BASIC_NUMBER)
        self.create_basic_rule({"type": "string"}, self.BASIC_STRING)
        self.create_basic_rule({"type": "boolean"}, self.BASIC_BOOLEAN)
        self.create_basic_rule({"type": "null"}, self.BASIC_NULL)
        self.create_basic_rule({"type": "array"}, self.BASIC_ARRAY)
        self.create_basic_rule({"type": "object"}, self.BASIC_OBJECT)

        self.strict_mode = past_strict_mode
        self.indent_manager = past_indent_manager

    def add_helper_rules(self):
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

    def create_basic_rule(self, schema: SchemaType, name: str) -> str:
        rule_name = self.create_rule_with_schema(schema, name)
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

    def create_rule_with_schema(self, schema: SchemaType, rule_name_hint: str) -> str:
        idx = self.get_schema_cache_index(schema)
        if idx in self.basic_rules_cache:
            return self.basic_rules_cache[idx]

        assert isinstance(rule_name_hint, str)

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
            res += self.create_rule_with_schema(anyof_schema, f"{rule_name}_{i}")
        return res

    def visit_any(self, schema: SchemaType, rule_name: str) -> str:
        # note integer is a subset of number, so we don't need to add integer here
        return (
            f"{self.BASIC_NUMBER} | {self.BASIC_STRING} | {self.BASIC_BOOLEAN} | "
            f"{self.BASIC_NULL} | {self.BASIC_ARRAY} | {self.BASIC_OBJECT}"
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
        item = self.create_rule_with_schema(schema, sub_rule_name)
        return f"{self.get_sep()} {item}"

    def get_partial_rule_for_repetitive_item(self, schema: SchemaType, sub_rule_name: str) -> str:
        item = self.create_rule_with_schema(schema, sub_rule_name)
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
            properties_all_optional = None
            if "properties" in schema and len(schema["properties"]) > 0:
                properties = list(schema["properties"].items())

                first_required_idx = len(properties)
                for i, (prop_name, _) in enumerate(properties):
                    if prop_name in required:
                        first_required_idx = i
                        break

                if first_required_idx == len(properties):
                    res += " " + self.get_partial_rule_for_properties_all_optional(
                        properties, rule_name
                    )
                else:
                    res += " " + self.get_partial_rule_for_properties_contain_required(
                        properties, rule_name, first_required_idx, required
                    )

            additional = schema.get("additionalProperties", False)
            if additional is not False:
                res += " " + self.get_partial_rule_for_repetitive_property(
                    "basic_string",
                    schema["additionalProperties"],
                    sub_rule_name=rule_name + "_add",
                )

            unevaluated = schema.get("unevaluatedProperties", not self.strict_mode)
            # if additionalProperties is in the schema, we don't need to consider
            # unevaluatedProperties
            if "additionalProperties" not in schema and unevaluated is not False:
                res += " " + self.get_partial_rule_for_repetitive_property(
                    self.BASIC_STRING,
                    unevaluated,
                    sub_rule_name=f"{rule_name}_uneval",
                )

        res += ' "}"'
        return res

    def get_property_pattern(self, prop_name: str, prop_schema: SchemaType, rule_name: str) -> str:
        # the outer quote is for the string in EBNF grammar, and the inner quote is for
        # the string in JSON
        colon = f'"{self.colon}"'
        key = f'"\\"{prop_name}\\""'
        value = self.create_rule_with_schema(prop_schema, f"{rule_name}_{prop_name}")
        return f"{key} {colon} {value}"

    def get_partial_rule_for_properties_all_optional(
        self, properties: List[Tuple[str, SchemaType]], rule_name: str
    ) -> str:
        assert len(properties) >= 1
        # ("," (("a" ("" | "," b) | b) ("" | "," c) | c)) | ""

        first_sep = self.get_sep()

        res = ""

        for i, (prop_name, prop_schema) in enumerate(properties):
            prop_pattern = self.get_property_pattern(prop_name, prop_schema, rule_name)

            if i == 0:
                res = prop_pattern
            else:
                res = f'({res}) ("" | {self.get_sep()} {prop_pattern}) | {prop_pattern}'

        res = f'({first_sep} ({res}) | "")'
        return res

    def get_partial_rule_for_properties_contain_required(
        self,
        properties: List[Tuple[str, SchemaType]],
        rule_name: str,
        first_required_idx: int,
        required: List[str],
    ) -> str:
        assert first_required_idx < len(properties)
        res = self.get_sep()

        for prop_name, prop_schema in properties[:first_required_idx]:
            assert prop_schema is not False
            property_pattern = self.get_property_pattern(prop_name, prop_schema, rule_name)
            res += f" ({property_pattern} {self.get_sep()})?"

        property_pattern = self.get_property_pattern(
            properties[first_required_idx][0], properties[first_required_idx][1], rule_name
        )
        res += f" {property_pattern}"

        for prop_name, prop_schema in properties[first_required_idx + 1 :]:
            assert prop_schema is not False
            property_pattern = self.get_property_pattern(prop_name, prop_schema, rule_name)
            if prop_name in required:
                res += f" {self.get_sep()} {property_pattern}"
            else:
                res += f" ({self.get_sep()} {property_pattern})?"

        return res

    def get_partial_rule_for_repetitive_property(
        self, key_pattern: str, schema: str, sub_rule_name: str
    ) -> str:
        colon = f'"{self.colon}"'
        value = self.create_rule_with_schema(schema, sub_rule_name)
        property_pattern = f"{key_pattern} {colon} {value}"
        return f'("" | {self.get_sep()} {property_pattern} ({self.get_sep()} {property_pattern})*)'


def json_schema_to_ebnf(
    json_schema: str,
    *,
    indent: Union[int, None] = None,
    separators: Union[Tuple[str, str], None] = None,
    strict_mode: bool = False,
) -> str:
    json_schema_schema = json.loads(json_schema)
    return JSONSchemaToEBNFConverter(json_schema_schema, indent, separators, strict_mode).convert()
