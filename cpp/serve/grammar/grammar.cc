/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar.cc
 */

#include "grammar.h"

#include "grammar_parser.h"
#include "grammar_serializer.h"
#include "grammar_simplifier.h"
#include "json_schema_converter.h"

namespace mlc {
namespace llm {
namespace serve {

TVM_REGISTER_OBJECT_TYPE(BNFGrammarNode);

std::ostream& operator<<(std::ostream& os, const BNFGrammar& grammar) {
  os << BNFGrammarPrinter(grammar).ToString();
  return os;
}

BNFGrammar BNFGrammar::FromEBNFString(const std::string& ebnf_string, const std::string& main_rule,
                                      bool normalize, bool simplify) {
  auto grammar = EBNFParser::Parse(ebnf_string, main_rule);
  if (normalize) {
    grammar = NestedRuleUnwrapper(grammar).Apply();
  }
  return grammar;
}

TVM_REGISTER_GLOBAL("mlc.serve.BNFGrammarFromEBNFString")
    .set_body_typed([](String ebnf_string, String main_rule, bool normalize, bool simplify) {
      return BNFGrammar::FromEBNFString(ebnf_string, main_rule, normalize, simplify);
    });

BNFGrammar BNFGrammar::FromJSON(const std::string& json_string) {
  return BNFJSONParser::Parse(json_string);
}

TVM_REGISTER_GLOBAL("mlc.serve.BNFGrammarFromJSON").set_body_typed([](String json_string) {
  return BNFGrammar::FromJSON(json_string);
});

BNFGrammar BNFGrammar::FromSchema(const std::string& schema, std::optional<int> indent,
                                  std::optional<std::pair<std::string, std::string>> separators,
                                  bool strict_mode) {
  return FromEBNFString(JSONSchemaToEBNF(schema, indent, separators, strict_mode));
}

TVM_REGISTER_GLOBAL("mlc.serve.BNFGrammarFromSchema").set_body([](TVMArgs args, TVMRetValue* rv) {
  std::optional<int> indent;
  if (args[1].type_code() != kTVMNullptr) {
    indent = args[1];
  } else {
    indent = std::nullopt;
  }

  std::optional<std::pair<std::string, std::string>> separators;
  if (args[2].type_code() != kTVMNullptr) {
    Array<String> separators_arr = args[2];
    CHECK(separators_arr.size() == 2);
    separators = std::make_pair(separators_arr[0], separators_arr[1]);
  } else {
    separators = std::nullopt;
  }

  *rv = BNFGrammar::FromSchema(args[0], indent, separators, args[3]);
});

const std::string kJSONGrammarString = R"(
main ::= (
    "{" ws members_or_embrace |
    "[" ws elements_or_embrace
)
value ::= (
    "{" ws members_or_embrace |
    "[" ws elements_or_embrace |
    "\"" characters "\"" |
    [0-9] fraction exponent |
    [1-9] digits fraction exponent |
    "-" [0-9] fraction exponent |
    "-" [1-9] digits fraction exponent |
    "true" |
    "false" |
    "null"
)
members_or_embrace ::= (
    "\"" characters "\"" ws ":" ws value members_rest ws "}" |
    "}"
)
members ::= "\"" characters "\"" ws ":" ws value members_rest
members_rest ::= (
    "" |
    "," ws "\"" characters "\"" ws ":" ws value members_rest |
    " " ws "," ws "\"" characters "\"" ws ":" ws value members_rest |
    "\n" ws "," ws "\"" characters "\"" ws ":" ws value members_rest |
    "\t" ws "," ws "\"" characters "\"" ws ":" ws value members_rest
)
elements_or_embrace ::= (
    "{" ws members_or_embrace elements_rest ws "]" |
    "[" ws elements_or_embrace elements_rest ws "]" |
    "\"" characters "\"" elements_rest ws "]" |
    [0-9] fraction exponent elements_rest ws "]" |
    [1-9] digits fraction exponent elements_rest ws "]" |
    "-" [0-9] fraction exponent elements_rest ws "]" |
    "-" [1-9] digits fraction exponent elements_rest ws "]" |
    "true" elements_rest ws "]" |
    "false" elements_rest ws "]" |
    "null" elements_rest ws "]" |
    "]"
)
elements ::= (
    "{" ws members_or_embrace elements_rest |
    "[" ws elements_or_embrace elements_rest |
    "\"" characters "\"" elements_rest |
    [0-9] fraction exponent elements_rest |
    [1-9] digits fraction exponent elements_rest |
    "-" [0-9] fraction exponent elements_rest |
    "-" [1-9] digits fraction exponent elements_rest |
    "true" elements_rest |
    "false" elements_rest |
    "null" elements_rest
)
elements_rest ::= (
    "" |
    "," ws elements |
    " " ws "," ws elements |
    "\n" ws "," ws elements |
    "\t" ws "," ws elements
)
characters ::= "" | [^"\\\r\n] characters | "\\" escape characters
escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
digits ::= [0-9] | [0-9] digits
fraction ::= "" | "." digits
exponent ::= "" |  "e" sign digits | "E" sign digits
sign ::= "" | "+" | "-"
ws ::= [ \n\t]*
)";

BNFGrammar BNFGrammar::GetGrammarOfJSON() {
  static const BNFGrammar grammar =
      BNFGrammar::FromEBNFString(kJSONGrammarString, "main", true, false);
  return grammar;
}

TVM_REGISTER_GLOBAL("mlc.serve.BNFGrammarGetGrammarOfJSON").set_body_typed([]() {
  return BNFGrammar::GetGrammarOfJSON();
});

}  // namespace serve
}  // namespace llm
}  // namespace mlc
