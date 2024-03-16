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


class JSONSchemaConverter:
    @staticmethod
    def to_ebnf(
        json_schema: str,
        *,
        indent: Union[int, None] = None,
        separators: Union[Tuple[str, str], None] = None,
        strict_mode: bool = False,
    ) -> str:
        json_schema_schema = json.loads(json_schema)
        return JSONSchemaConverter(json_schema_schema, indent, separators, strict_mode).convert()

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
        self.cache_schema_to_rule: Dict[str, str] = {}
        self.init_basic_rules()

    def convert(self) -> str:
        self.create_rule_with_schema("main", self.json_schema)
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
    BASIC_WS = "basic_ws"
    BASIC_ESCAPE = "basic_escape"
    BASIC_STRING_SUB = "basic_string_sub"

    def init_basic_rules(self):
        past_strict_mode = self.strict_mode
        self.strict_mode = False

        self.init_helper_rules()
        self.create_rule_with_schema(self.BASIC_ANY, True)
        self.create_rule_with_schema(self.BASIC_INTEGER, {"type": "integer"})
        self.create_rule_with_schema(self.BASIC_NUMBER, {"type": "number"})
        self.create_rule_with_schema(self.BASIC_STRING, {"type": "string"})
        self.create_rule_with_schema(self.BASIC_BOOLEAN, {"type": "boolean"})
        self.create_rule_with_schema(self.BASIC_NULL, {"type": "null"})
        self.create_rule_with_schema(self.BASIC_ARRAY, {"type": "array"})
        self.create_rule_with_schema(self.BASIC_OBJECT, {"type": "object"})

        self.strict_mode = past_strict_mode

    def init_helper_rules(self):
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

    def get_sep(self):
        return self.indent_manager.get_sep()

    @staticmethod
    def warn_unsupported_keywords(schema: SchemaType, keywords: Union[str, List[str]]):
        if isinstance(keywords, str):
            keywords = [keywords]
        for keyword in keywords:
            if keyword in schema:
                # todo: test and output format
                logging.warning(f"Keyword {keyword} is not supported in schema {schema}")

    def create_rule_with_schema(self, rule_name: str, schema: SchemaType) -> str:
        schema_index = self.get_schema_cache_index(schema)
        if schema_index in self.cache_schema_to_rule:
            return self.cache_schema_to_rule[schema_index]

        self.cache_schema_to_rule[schema_index] = rule_name
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

    def get_schema_cache_index(self, schema: SchemaType) -> str:
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
            res += self.create_rule_with_schema(f"{rule_name}_{i}", anyof_schema)
        return res

    def visit_any(self, schema: SchemaType, rule_name: str) -> str:
        # note integer is a subset of number, so we don't need to add integer here
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
        is_first_item = True

        if "prefixItems" in schema:
            for i, prefix_item in enumerate(schema["prefixItems"]):
                assert prefix_item is not False
                res += " " + self.get_partial_rule_for_item(
                    prefix_item, f"{rule_name}_{i}", is_first_item
                )
                is_first_item = False

        items = schema.get("items", False)
        if items is not False:
            res += " " + self.get_partial_rule_for_repetitive_item(
                schema["items"], f"{rule_name}_item", is_first_item
            )
            is_first_item = False

        unevaluated = schema.get("unevaluatedItems", not self.strict_mode)
        # if items is in the schema, we don't need to consider unevaluatedItems
        if "items" not in schema and unevaluated is not False:
            rest_schema = schema.get("unevaluatedItems", True)
            res += " " + self.get_partial_rule_for_repetitive_item(
                rest_schema, f"{rule_name}_rest", is_first_item
            )
            is_first_item = False

        res += ' "]"'
        return res

    def get_partial_rule_for_item(
        self, schema: SchemaType, sub_rule_name: str, is_first_item: bool
    ) -> str:
        item = self.create_rule_with_schema(sub_rule_name, schema)
        return f"{self.get_sep()} {item}"

    def get_partial_rule_for_repetitive_item(
        self, schema: SchemaType, sub_rule_name: str, is_first_item: bool
    ) -> str:
        item = self.create_rule_with_schema(sub_rule_name, schema)
        return f'("" | {self.get_sep()} {item} ({self.get_sep()} {item})*)'

    def visit_object(self, schema: SchemaType, rule_name: str) -> str:
        assert schema["type"] == "object"
        JSONSchemaConverter.warn_unsupported_keywords(
            schema, ["patternProperties", "minProperties", "maxProperties", "propertyNames"]
        )

        res = '"{"'
        # Now we only consider the required list for the properties field
        required = schema.get("required", [])
        is_first_property = True

        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                assert prop_schema is not False
                # the outer quote is for the string in EBNF grammar, and the inner quote is for
                # the string in JSON
                key = f'"\\"{prop_name}\\""'
                res += " " + self.get_partial_rule_for_optional_property(
                    key,
                    prop_schema,
                    sub_rule_name=f"{rule_name}_{prop_name}",
                    required=prop_name in required,
                    is_first_property=is_first_property,
                )
                is_first_property = False

        additional = schema.get("additionalProperties", False)
        if additional is not False:
            res += " " + self.get_partial_rule_for_repetitive_property(
                "basic_string",
                schema["additionalProperties"],
                sub_rule_name=rule_name + "_add",
                is_first_property=is_first_property,
            )
            is_first_property = False

        unevaluated = schema.get("unevaluatedProperties", not self.strict_mode)
        # if additionalProperties is in the schema, we don't need to consider unevaluatedProperties
        if "additionalProperties" not in schema and unevaluated is not False:
            res += " " + self.get_partial_rule_for_repetitive_property(
                "basic_string",
                unevaluated,
                sub_rule_name=f"{rule_name}_uneval",
                is_first_property=is_first_property,
            )
            is_first_property = False

        res += ' "}"'
        return res

    def get_partial_rule_for_optional_property(
        self, key: str, schema: str, sub_rule_name: str, required: bool, is_first_property: bool
    ) -> str:
        colon = f'"{self.colon}"'
        value = self.create_rule_with_schema(sub_rule_name, schema)
        res = f"{self.get_sep()} {key} {colon} {value}"
        if not required:
            res = f"({res})?"
        return res

    def get_partial_rule_for_repetitive_property(
        self, key: str, schema: str, sub_rule_name: str, is_first_property: bool
    ) -> str:
        separator = f'"{self.separators[0]}"'
        colon = f'"{self.separators[1]}"'
        value = self.create_rule_with_schema(sub_rule_name, schema)
        property = f"{key} {colon} {value}"

        if is_first_property:
            return f'("" | {property} ({separator} {property})*)'
        else:
            return f"({separator} {property})*"
