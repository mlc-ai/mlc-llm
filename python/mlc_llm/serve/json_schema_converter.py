# mypy: disable-error-code="operator,union-attr,index"
"""Utility to convert JSON schema to EBNF grammar. Helpful for the grammar-guided generation."""
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

SchemaType = Union[Dict[str, Any], bool]
"""
JSON schema specification defines the schema type could be a dictionary or a boolean value.
"""


class _IndentManager:
    """Manage the indent and separator for the generation of EBNF grammar.

    Parameters
    ----------
    indent : Optional[int]
        The number of spaces for each indent. If it is None, there will be no indent or newline.

    separator : str
        The separator between different elements in json. Examples include "," and ", ".
    """

    def __init__(self, indent: Optional[int], separator: str):
        self.enable_newline = indent is not None
        self.indent = indent or 0
        self.separator = separator
        self.total_indent = 0
        self.is_first = [True]

    def __enter__(self):
        """Enter a new indent level."""
        self.total_indent += self.indent
        self.is_first.append(True)

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the current indent level."""
        self.total_indent -= self.indent
        self.is_first.pop()

    def get_sep(self, is_end: bool = False) -> str:
        """Get the separator according to the current state. When first called in the current level,
        the starting separator will be returned. When called again, the middle separator will be
        returned. When called with `is_end=True`, the ending separator will be returned.

        Parameters
        ----------
        is_end : bool
            Get the separator for the end of the current level.

        Examples
        --------
        >>> indent_manager = IndentManager(2, ", ")
        >>> with indent_manager:
        ...     print(indent_manager.get_sep()) # get the start separator
        ...     print(indent_manager.get_sep()) # get the middle separator
        ...     print(indent_manager.get_sep(is_end=True)) # get the end separator

        Output: (double quotes are included in the string for EBNF construction)
        '"\n  "'
        '",\n  "'
        '"\n"'
        """
        res = ""

        if not self.is_first[-1] and not is_end:
            res += self.separator
        self.is_first[-1] = False

        if self.enable_newline:
            res += "\\n"

        if not is_end:
            res += self.total_indent * " "
        else:
            res += (self.total_indent - self.indent) * " "

        return f'"{res}"'


# pylint: disable=unused-argument,too-few-public-methods
class _JSONSchemaToEBNFConverter:
    """Convert JSON schema string to EBNF grammar string. The parameters follow
    `json_schema_to_ebnf()`.
    """

    def __init__(
        self,
        json_schema: SchemaType,
        indent: Optional[int] = None,
        separators: Optional[Tuple[str, str]] = None,
        strict_mode: bool = False,
    ):
        self.json_schema = json_schema
        self.strict_mode = strict_mode

        if separators is None:
            separators = (", ", ": ") if indent is None else (",", ": ")
        assert len(separators) == 2
        self.indent_manager = _IndentManager(indent, separators[0])
        self.colon = separators[1]

        self.rules: List[Tuple[str, str]] = []
        self.basic_rules_cache: Dict[str, str] = {}
        self._add_basic_rules()

    def convert(self) -> str:
        """Main method. Convert the JSON schema to EBNF grammar string."""
        self._create_rule_with_schema(self.json_schema, "main")
        res = ""
        for rule_name, rule in self.rules:
            res += f"{rule_name} ::= {rule}\n"
        return res

    # The name of the basic rules
    BASIC_ANY = "basic_any"
    BASIC_INTEGER = "basic_integer"
    BASIC_NUMBER = "basic_number"
    BASIC_STRING = "basic_string"
    BASIC_BOOLEAN = "basic_boolean"
    BASIC_NULL = "basic_null"
    BASIC_ARRAY = "basic_array"
    BASIC_OBJECT = "basic_object"

    # The name of the helper rules to construct basic rules
    BASIC_ESCAPE = "basic_escape"
    BASIC_STRING_SUB = "basic_string_sub"

    def _add_basic_rules(self):
        """Add the basic rules to the rules list and the basic_rules_cache."""
        past_strict_mode = self.strict_mode
        self.strict_mode = False
        past_indent_manager = self.indent_manager
        self.indent_manager = _IndentManager(None, past_indent_manager.separator)

        self._add_helper_rules()
        self._create_basic_rule(True, self.BASIC_ANY)
        self.basic_rules_cache[self._get_schema_cache_index({})] = self.BASIC_ANY
        self._create_basic_rule({"type": "integer"}, self.BASIC_INTEGER)
        self._create_basic_rule({"type": "number"}, self.BASIC_NUMBER)
        self._create_basic_rule({"type": "string"}, self.BASIC_STRING)
        self._create_basic_rule({"type": "boolean"}, self.BASIC_BOOLEAN)
        self._create_basic_rule({"type": "null"}, self.BASIC_NULL)
        self._create_basic_rule({"type": "array"}, self.BASIC_ARRAY)
        self._create_basic_rule({"type": "object"}, self.BASIC_OBJECT)

        self.strict_mode = past_strict_mode
        self.indent_manager = past_indent_manager

    def _add_helper_rules(self):
        """Add helper rules for the basic rules."""
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

    def _create_basic_rule(self, schema: SchemaType, name: str):
        """Create a rule for the given schema and name, and add it to the basic_rules_cache."""
        rule_name = self._create_rule_with_schema(schema, name)
        self.basic_rules_cache[self._get_schema_cache_index(schema)] = rule_name

    def _get_sep(self, is_end: bool = False):
        """Get the separator from the indent manager."""
        return self.indent_manager.get_sep(is_end)

    @staticmethod
    def _warn_unsupported_keywords(schema: SchemaType, keywords: Union[str, List[str]]):
        """Warn if any keyword is existing in the schema but not supported."""
        if isinstance(schema, bool):
            return
        if isinstance(keywords, str):
            keywords = [keywords]
        for keyword in keywords:
            if keyword in schema:
                logging.warning("Keyword %s is not supported in schema %s", keyword, schema)

    def _create_rule_with_schema(self, schema: SchemaType, rule_name_hint: str) -> str:
        """Create a rule with the given schema and rule name hint.

        Returns
        -------
        The name of the rule will be returned. That is not necessarily the same as the
        rule_name_hint due to the caching mechanism.
        """
        idx = self._get_schema_cache_index(schema)
        if idx in self.basic_rules_cache:
            return self.basic_rules_cache[idx]

        assert isinstance(rule_name_hint, str)

        self.rules.append((rule_name_hint, self._visit_schema(schema, rule_name_hint)))
        return rule_name_hint

    # The keywords that will be ignored when finding the cached rule for a schema
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
    def _remove_skipped_keys_recursive(obj: Any) -> Any:
        """Remove the skipped keys from the schema recursively."""
        if isinstance(obj, dict):
            return {
                k: _JSONSchemaToEBNFConverter._remove_skipped_keys_recursive(v)
                for k, v in obj.items()
                if k not in _JSONSchemaToEBNFConverter.SKIPPED_KEYS
            }
        if isinstance(obj, list):
            return [_JSONSchemaToEBNFConverter._remove_skipped_keys_recursive(v) for v in obj]
        return obj

    def _get_schema_cache_index(self, schema: SchemaType) -> str:
        """Get the index for the schema in the cache."""
        return json.dumps(
            _JSONSchemaToEBNFConverter._remove_skipped_keys_recursive(schema),
            sort_keys=True,
            indent=None,
        )

    # pylint: disable=too-many-return-statements,too-many-branches
    def _visit_schema(self, schema: SchemaType, rule_name: str) -> str:
        """Visit the schema and return the rule body for later constructing the rule."""
        assert schema is not False
        if schema is True:
            return self._visit_any(schema, rule_name)

        _JSONSchemaToEBNFConverter._warn_unsupported_keywords(
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
            return self._visit_ref(schema, rule_name)
        if "const" in schema:
            return self._visit_const(schema, rule_name)
        if "enum" in schema:
            return self._visit_enum(schema, rule_name)
        if "anyOf" in schema:
            return self._visit_anyof(schema, rule_name)
        if "type" in schema:
            type_obj = schema["type"]
            if type_obj == "integer":
                return self._visit_integer(schema, rule_name)
            if type_obj == "number":
                return self._visit_number(schema, rule_name)
            if type_obj == "string":
                return self._visit_string(schema, rule_name)
            if type_obj == "boolean":
                return self._visit_boolean(schema, rule_name)
            if type_obj == "null":
                return self._visit_null(schema, rule_name)
            if type_obj == "array":
                return self._visit_array(schema, rule_name)
            if type_obj == "object":
                return self._visit_object(schema, rule_name)
            raise ValueError(f"Unsupported type {schema['type']}")
        # no keyword is detected, we treat it as any
        return self._visit_any(schema, rule_name)

    def _visit_ref(self, schema: SchemaType, rule_name: str) -> str:
        """Visit a reference schema."""
        assert "$ref" in schema
        new_schema = self._uri_to_schema(schema["$ref"]).copy()
        if not isinstance(new_schema, bool):
            new_schema.update({k: v for k, v in schema.items() if k != "$ref"})
        return self._visit_schema(new_schema, rule_name)

    def _uri_to_schema(self, uri: str) -> SchemaType:
        """Get the schema from the URI."""
        if uri.startswith("#/$defs/"):
            return self.json_schema["$defs"][uri[len("#/$defs/") :]]
        logging.warning("Now only support URI starting with '#/$defs/' but got %s", uri)
        return True

    def _visit_const(self, schema: SchemaType, rule_name: str) -> str:
        """Visit a const schema."""
        assert "const" in schema
        return '"' + self._json_str_to_printable_str(json.dumps(schema["const"])) + '"'

    def _visit_enum(self, schema: SchemaType, rule_name: str) -> str:
        """Visit an enum schema."""
        assert "enum" in schema
        res = ""
        for i, enum_value in enumerate(schema["enum"]):
            if i != 0:
                res += " | "
            res += '("' + self._json_str_to_printable_str(json.dumps(enum_value)) + '")'
        return res

    REPLACE_MAPPING = {
        "\\": "\\\\",
        '"': '\\"',
    }

    def _json_str_to_printable_str(self, json_str: str) -> str:
        """Convert the JSON string to a printable string in BNF."""
        for k, v in self.REPLACE_MAPPING.items():
            json_str = json_str.replace(k, v)
        return json_str

    def _visit_anyof(self, schema: SchemaType, rule_name: str) -> str:
        """Visit an anyOf schema."""
        assert "anyOf" in schema
        res = ""
        for i, anyof_schema in enumerate(schema["anyOf"]):
            if i != 0:
                res += " | "
            res += self._create_rule_with_schema(anyof_schema, f"{rule_name}_{i}")
        return res

    def _visit_any(self, schema: SchemaType, rule_name: str) -> str:
        """Visit a true schema that can match anything."""
        # note integer is a subset of number, so we don't need to add integer here
        return (
            f"{self.BASIC_NUMBER} | {self.BASIC_STRING} | {self.BASIC_BOOLEAN} | "
            f"{self.BASIC_NULL} | {self.BASIC_ARRAY} | {self.BASIC_OBJECT}"
        )

    def _visit_integer(self, schema: SchemaType, rule_name: str) -> str:
        """Visit an integer schema."""
        assert schema["type"] == "integer"
        _JSONSchemaToEBNFConverter._warn_unsupported_keywords(
            schema, ["multipleOf", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]
        )
        return '("0" | "-"? [1-9] [0-9]*) ".0"?'

    def _visit_number(self, schema: SchemaType, rule_name: str) -> str:
        """Visit a number schema."""
        assert schema["type"] == "number"
        _JSONSchemaToEBNFConverter._warn_unsupported_keywords(
            schema, ["multipleOf", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]
        )
        return '("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?'

    def _visit_string(self, schema: SchemaType, rule_name: str) -> str:
        """Visit a string schema."""
        assert schema["type"] == "string"
        _JSONSchemaToEBNFConverter._warn_unsupported_keywords(
            schema, ["minLength", "maxLength", "pattern", "format"]
        )
        return f'["] {self.BASIC_STRING_SUB} ["]'

    def _visit_boolean(self, schema: SchemaType, rule_name: str) -> str:
        """Visit a boolean schema."""
        assert schema["type"] == "boolean"

        return '"true" | "false"'

    def _visit_null(self, schema: SchemaType, rule_name: str) -> str:
        """Visit a null schema."""
        assert schema["type"] == "null"

        return '"null"'

    def _visit_array(self, schema: SchemaType, rule_name: str) -> str:
        """Visit an array schema.

        Examples
        --------
        Schema:
        {
            "type": "array",
            "prefixItems": [
                {"type": "boolean"},
                {"type": "integer"}
            ],
            "items": {
                "type": "string"
            }
        }

        Rule (not considering the indent):
        main ::= "[" basic_boolean ", " basic_integer (", " basic_string)* "]"
        """
        assert schema["type"] == "array"
        _JSONSchemaToEBNFConverter._warn_unsupported_keywords(
            schema,
            ["uniqueItems", "contains", "minContains", "maxContains", "minItems", "maxItems"],
        )

        res = '"["'

        with self.indent_manager:
            # 1. Handle prefix items
            have_prefix_items = False
            if "prefixItems" in schema:
                for i, prefix_item in enumerate(schema["prefixItems"]):
                    assert prefix_item is not False
                    item = self._create_rule_with_schema(prefix_item, f"{rule_name}_{i}")
                    res += f" {self._get_sep()} {item}"
                    have_prefix_items = True

            # 2. Find additional items
            additional_item = None
            additional_suffix = ""

            items = schema.get("items", False)
            if items is not False:
                additional_item = items
                additional_suffix = "item"

            # if items is in the schema, we don't need to consider unevaluatedItems
            unevaluated = schema.get("unevaluatedItems", not self.strict_mode)
            if "items" not in schema and unevaluated is not False:
                additional_item = unevaluated
                additional_suffix = "uneval"

            # 3. Handle additional items and the end separator
            if additional_item is None:
                res += f" {self._get_sep(is_end=True)}"
            else:
                additional_pattern = self._create_rule_with_schema(
                    additional_item, f"{rule_name}_{additional_suffix}"
                )
                if have_prefix_items:
                    res += (
                        f' ("" | ({self._get_sep()} {additional_pattern})*)'
                        f" {self._get_sep(is_end=True)}"
                    )
                else:
                    res += (
                        f' ("" | {self._get_sep()} {additional_pattern} ({self._get_sep()} '
                        f"{additional_pattern})* {self._get_sep(is_end=True)})"
                    )

        res += ' "]"'
        return res

    def _visit_object(self, schema: SchemaType, rule_name: str) -> str:
        """Visit an object schema.

        Examples
        --------
        Schema:
        {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"}
            },
            "required": ["a"],
            "additionalProperties": true
        }

        Rule (not considering the indent):
        main ::= "{" "a" ":" basic_string (", " "b" ":" basic_integer)*
                 (", " basic_string ": " basic_any)* "}"

        We need special handling when all properties are optional, since the handling of separators
        is tricky in this case. E.g.

        Schema:
        {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"},
                "c": {"type": "boolean"}
            },
            "additionalProperties": true
        }

        Rule (indent=2):
        main ::= "{" ("\n  " (a main_sub_1 | b main_sub_2 | c main_sub_3 | d main_sub_3)
                 "\n" | "") "}"
        main_sub_1 ::= ",\n  " b r2 | r2
        main_sub_2 ::= ",\n  " c r3 | r3
        main_sub_3 ::= (",\n  " d)*
        """
        assert schema["type"] == "object"
        _JSONSchemaToEBNFConverter._warn_unsupported_keywords(
            schema, ["patternProperties", "minProperties", "maxProperties", "propertyNames"]
        )

        res = '"{"'
        # Now we only consider the required list for the properties field
        required = schema.get("required", [])

        with self.indent_manager:
            # 1. Find additional properties
            additional_property = None
            additional_suffix = ""

            additional = schema.get("additionalProperties", False)
            if additional is not False:
                additional_property = additional
                additional_suffix = "add"

            unevaluated = schema.get("unevaluatedProperties", not self.strict_mode)
            if "additionalProperties" not in schema and unevaluated is not False:
                additional_property = unevaluated
                additional_suffix = "uneval"

            # 2. Handle properties
            properties_obj = schema.get("properties", {})
            properties = list(properties_obj.items())

            properties_all_optional = all(prop_name not in required for prop_name, _ in properties)
            if properties_all_optional and len(properties) > 0:
                # 3.1 Case 1: properties are defined and all properties are optional
                res += " " + self._get_partial_rule_for_properties_all_optional(
                    properties, additional_property, rule_name, additional_suffix
                )
            elif len(properties) > 0:
                # 3.2 Case 2: properties are defined and some properties are required
                res += " " + self._get_partial_rule_for_properties_contain_required(
                    properties, required, rule_name
                )
                if additional_property is not None:
                    other_property_pattern = self._get_other_property_pattern(
                        self.BASIC_STRING, additional_property, rule_name, additional_suffix
                    )
                    res += f" ({self._get_sep()} {other_property_pattern})*"
                res += " " + self._get_sep(is_end=True)
            elif additional_property is not None:
                # 3.3 Case 3: no properties are defined and additional properties are allowed
                other_property_pattern = self._get_other_property_pattern(
                    self.BASIC_STRING, additional_property, rule_name, additional_suffix
                )
                res += (
                    f" ({self._get_sep()} {other_property_pattern} ({self._get_sep()} "
                    f'{other_property_pattern})* {self._get_sep(is_end=True)} | "")'
                )

        res += ' "}"'
        return res

    def _get_property_pattern(self, prop_name: str, prop_schema: SchemaType, rule_name: str) -> str:
        """Get the pattern for a property in the object schema."""
        # the outer quote is for the string in EBNF grammar, and the inner quote is for
        # the string in JSON
        key = f'"\\"{prop_name}\\""'
        colon = f'"{self.colon}"'
        value = self._create_rule_with_schema(prop_schema, rule_name + "_" + prop_name)
        return f"{key} {colon} {value}"

    def _get_other_property_pattern(
        self, key_pattern: str, prop_schema: SchemaType, rule_name: str, rule_name_suffix: str
    ) -> str:
        """Get the pattern for the additional/unevaluated properties in the object schema."""
        colon = f'"{self.colon}"'
        value = self._create_rule_with_schema(prop_schema, rule_name + "_" + rule_name_suffix)
        return f"{key_pattern} {colon} {value}"

    # pylint: disable=too-many-locals
    def _get_partial_rule_for_properties_all_optional(
        self,
        properties: List[Tuple[str, SchemaType]],
        additional: Optional[SchemaType],
        rule_name: str,
        additional_suffix: str = "",
    ) -> str:
        """Get the partial rule for the properties when all properties are optional. See the
        above example."""
        assert len(properties) >= 1

        first_sep = self._get_sep()
        mid_sep = self._get_sep()
        last_sep = self._get_sep(is_end=True)

        res = ""

        prop_patterns = [
            self._get_property_pattern(prop_name, prop_schema, rule_name)
            for prop_name, prop_schema in properties
        ]

        rule_names = [None] * len(properties)

        # construct the last rule
        if additional is not None:
            additional_prop_pattern = self._get_other_property_pattern(
                self.BASIC_STRING, additional, rule_name, additional_suffix
            )
            last_rule_body = f"({mid_sep} {additional_prop_pattern})*"
            last_rule_name = f"{rule_name}_sub_{len(properties)-1}"
            self.rules.append((last_rule_name, last_rule_body))
            rule_names[-1] = last_rule_name  # type: ignore
        else:
            rule_names[-1] = '""'  # type: ignore

        # construct 0~(len(properties) - 2) rules
        for i in reversed(range(0, len(properties) - 1)):
            prop_pattern = prop_patterns[i + 1]
            last_rule_name = rule_names[i + 1]
            cur_rule_body = f"{last_rule_name} | {mid_sep} {prop_pattern} {last_rule_name}"
            cur_rule_name = f"{rule_name}_sub_{i}"
            self.rules.append((cur_rule_name, cur_rule_body))
            rule_names[i] = cur_rule_name  # type: ignore

        # construct the main rule
        for i, prop_pattern in enumerate(prop_patterns):
            if i != 0:
                res += " | "
            res += f"({prop_pattern} {rule_names[i]})"

        if additional is not None:
            res += f" | {additional_prop_pattern} {rule_names[-1]}"

        # add separators and the empty string option
        res = f'({first_sep} ({res}) {last_sep} | "")'
        return res

    def _get_partial_rule_for_properties_contain_required(
        self,
        properties: List[Tuple[str, SchemaType]],
        required: List[str],
        rule_name: str,
    ) -> str:
        """Get the partial rule for the properties when some properties are required. See the
        above example.

        The constructed rule should be:

        start_separator (optional_property separator)? (optional_property separator)? ...
        first_required_property (separator optional_property)? separator required_property ...
        end_separator

        i.e. Before the first required property, all properties are in the form
        (property separator); and after the first required property, all properties are in the form
        (separator property).
        """

        # Find the index of the first required property
        first_required_idx = next(
            (i for i, (prop_name, _) in enumerate(properties) if prop_name in required),
            len(properties),
        )
        assert first_required_idx < len(properties)

        res = self._get_sep()

        # Handle the properties before the first required property
        for prop_name, prop_schema in properties[:first_required_idx]:
            assert prop_schema is not False
            property_pattern = self._get_property_pattern(prop_name, prop_schema, rule_name)
            res += f" ({property_pattern} {self._get_sep()})?"

        # Handle the first required property
        property_pattern = self._get_property_pattern(
            properties[first_required_idx][0], properties[first_required_idx][1], rule_name
        )
        res += f" {property_pattern}"

        # Handle the properties after the first required property
        for prop_name, prop_schema in properties[first_required_idx + 1 :]:
            assert prop_schema is not False
            property_pattern = self._get_property_pattern(prop_name, prop_schema, rule_name)
            if prop_name in required:
                res += f" {self._get_sep()} {property_pattern}"
            else:
                res += f" ({self._get_sep()} {property_pattern})?"

        return res


def json_schema_to_ebnf(
    json_schema: str,
    *,
    indent: Optional[int] = None,
    separators: Optional[Tuple[str, str]] = None,
    strict_mode: bool = True,
) -> str:
    """Convert JSON schema string to EBNF grammar string.

    Parameters
    ----------
    json_schema : str
        The JSON schema string.

    indent : Optional[int]
        The number of spaces for each indent. If it is None, there will be no indent or newline.
        The indent and separators parameters follow the same convention as
        `json.dumps()`.

    separators : Optional[Tuple[str, str]]
        The separator between different elements in json. Examples include "," and ", ".

    strict_mode : bool
        Whether to use strict mode. In strict mode, the generated grammar will not allow
        unevaluatedProperties and unevaluatedItems, i.e. these will be set to false by default.
        This helps LLM to generate accurate output in the grammar-guided generation with JSON
        schema.
    """
    json_schema_schema = json.loads(json_schema)
    return _JSONSchemaToEBNFConverter(json_schema_schema, indent, separators, strict_mode).convert()
