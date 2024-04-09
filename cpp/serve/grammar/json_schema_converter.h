/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/json_grammar_converter.h
 * \brief The header for translating JSON schema to EBNF grammar.
 */

#ifndef MLC_LLM_SERVE_GRAMMAR_JSON_SCHEMA_CONVERTER_H_
#define MLC_LLM_SERVE_GRAMMAR_JSON_SCHEMA_CONVERTER_H_

#include <optional>
#include <string>
#include <utility>

namespace mlc {
namespace llm {
namespace serve {

/*!
 * \brief Convert JSON schema string to EBNF grammar string.
 * \param json_schema The JSON schema string.
 * \param indent The number of spaces for indentation. If set to std::nullopt, the output will be
 * in one line. Default: std::nullopt.
 * \param separators Two separators used in the schema: comma and colon. Examples: {",", ":"},
 * {", ", ": "}. If std::nullopt, the default separators will be used: {",", ": "} when the
 * indent is not -1, and {", ", ": "} otherwise. This follows the convention in python json.dumps().
 * Default: std::nullopt.
 * \param strict_mode Whether to use strict mode. In strict mode, the generated grammar will not
 * allow properties and items that is not specified in the schema. This is equivalent to
 * setting unevaluatedProperties and unevaluatedItems to false.
 *
 * This helps LLM to generate accurate output in the grammar-guided generation with JSON
 * schema. Default: true.
 * \returns The EBNF grammar string.
 */
std::string JSONSchemaToEBNF(
    std::string schema, std::optional<int> indent = std::nullopt,
    std::optional<std::pair<std::string, std::string>> separators = std::nullopt,
    bool strict_mode = true);

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_JSON_SCHEMA_CONVERTER_H_
