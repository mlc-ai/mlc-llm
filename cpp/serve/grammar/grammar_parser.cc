/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar_parser.cc
 */

#include "grammar_parser.h"

#include "../../metadata/json_parser.h"
#include "../../support/encoding.h"
#include "grammar_builder.h"

namespace mlc {
namespace llm {
namespace serve {

class EBNFParserImpl {
 public:
  /*! \brief The logic of parsing the grammar string. */
  BNFGrammar DoParse(String ebnf_string);

 private:
  using Rule = BNFGrammarNode::Rule;
  using ParseError = EBNFParser::ParseError;

  // Parsing different parts of the grammar
  std::string ParseName(bool accept_empty = false);
  int32_t ParseCharacterClass();
  int32_t ParseString();
  int32_t ParseRuleRef();
  int32_t ParseElement();
  int32_t ParseQuantifier();
  int32_t ParseSequence();
  int32_t ParseChoices();
  Rule ParseRule();

  // Helper functions
  // Helper for ParseQuantifier
  int32_t HandleStarQuantifier(int32_t rule_expr_id);
  int32_t HandlePlusQuantifier(int32_t rule_expr_id);
  int32_t HandleQuestionQuantifier(int32_t rule_expr_id);

  // When parsing, we first find the names of all rules, and build the mapping from name to rule id.
  void BuildRuleNameToId();
  // Consumes several spaces (newline, space, tab, comment, etc.)
  void ConsumeSpace(bool allow_newline = true);
  // Check the validity of a name
  static bool IsNameChar(TCodepoint c, bool first_char = false);
  // Reset the parser to the beginning of the string.
  void ResetStringIterator(const char* cur);

  // Consume a specified number of characters, and maintain the line and column number.
  void Consume(int cnt = 1) {
    for (int i = 0; i < cnt; ++i) {
      // \n \r \r\n
      if (Peek() == '\n' || (Peek() == '\r' && Peek(1) != '\n')) {
        ++cur_line_;
        cur_column_ = 1;
      } else {
        ++cur_column_;
      }
      ++cur_;
    }
  }

  // Peek the next character.
  char Peek(int delta = 0) const { return *(cur_ + delta); }

  // Throw a ParseError with the given message and the line and column number.
  [[noreturn]] void ThrowParseError(const std::string& msg) {
    throw ParseError("EBNF parse error at line " + std::to_string(cur_line_) + ", column " +
                     std::to_string(cur_column_) + ": " + msg);
  }

  // The grammar builder
  BNFGrammarBuilder builder_;
  // A pointer to the current parse position in the string
  const char* cur_ = nullptr;
  // The current line and column number
  int cur_line_ = 1;
  int cur_column_ = 1;
  // The current rule name. Help to generate a name for a new rule.
  std::string cur_rule_name_;
  // Whether the current element is in parentheses.
  // A sequence expression cannot contain newline, unless it is in parentheses.
  bool in_parentheses_ = false;
};

void EBNFParserImpl::ConsumeSpace(bool allow_newline) {
  while (Peek() && (Peek() == ' ' || Peek() == '\t' || Peek() == '#' ||
                    (allow_newline && (Peek() == '\n' || Peek() == '\r')))) {
    Consume();
    if (Peek(-1) == '#') {
      while (Peek() && Peek() != '\n' && Peek() != '\r') {
        Consume();
      }
      if (!Peek()) {
        return;
      }
      Consume();
      if (Peek(-1) == '\r' && Peek() == '\n') {
        Consume();
      }
    }
  }
}

bool EBNFParserImpl::IsNameChar(TCodepoint c, bool first_char) {
  return c == '_' || c == '-' || c == '.' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (!first_char && c >= '0' && c <= '9');
}

// name should be a char string (not a utf8 string)
std::string EBNFParserImpl::ParseName(bool accept_empty) {
  auto start = cur_;
  bool first_char = true;
  while (Peek() && IsNameChar(Peek(), first_char)) {
    Consume();
    first_char = false;
  }
  if (start == cur_ && !accept_empty) {
    ThrowParseError("Expect rule name");
  }
  return std::string(start, cur_);
}

// Character class:
// 1. Examples: [a-z] [ab] [a-zA-Z0-9] [^a-z] [测] [\u0123]
// 2. The "-" character is treated as a literal character if it is the last or the first (after
// the "^"", if present) character within the brackets. E.g. [a-] and [-a] means "a" or "-"
// 3. "-" and "]" should be escaped when used as a literal character:
// [\-] means -
// [\]] means ]
// Character class should not contain newlines.
int32_t EBNFParserImpl::ParseCharacterClass() {
  static constexpr TCodepoint kUnknownUpperBound = -4;
  static const std::unordered_map<std::string, TCodepoint> kCustomEscapeMap = {{"\\-", '-'},
                                                                               {"\\]", ']'}};

  std::vector<BNFGrammarBuilder::CharacterClassElement> elements;

  bool is_negated = false;
  if (Peek() == '^') {
    is_negated = true;
    Consume();
  }

  bool past_is_hyphen = false;
  bool past_is_single_char = false;
  while (Peek() && Peek() != ']') {
    if (Peek() == '\r' || Peek() == '\n') {
      ThrowParseError("Character class should not contain newline");
    } else if (Peek() == '-' && Peek(1) != ']' && !past_is_hyphen && past_is_single_char) {
      Consume();
      past_is_hyphen = true;
      past_is_single_char = false;
      continue;
    }

    auto [codepoint, len] = Utf8OrEscapeToCodepoint(cur_, kCustomEscapeMap);
    if (codepoint == static_cast<TCodepoint>(CharHandlingError::kInvalidUtf8)) {
      ThrowParseError("Invalid utf8 sequence");
    }
    if (codepoint == static_cast<TCodepoint>(CharHandlingError::kInvalidEscape)) {
      ThrowParseError("Invalid escape sequence");
    }
    Consume(len);
    if (past_is_hyphen) {
      ICHECK(!elements.empty());
      if (elements.back().lower > codepoint) {
        ThrowParseError("Invalid character class: lower bound is larger than upper bound");
      }
      elements.back().upper = codepoint;
      past_is_hyphen = false;
      ICHECK(past_is_single_char == false);
    } else {
      elements.push_back({codepoint, kUnknownUpperBound});
      past_is_single_char = true;
    }
  }

  for (auto& element : elements) {
    if (element.upper == kUnknownUpperBound) {
      element.upper = element.lower;
    }
  }

  return builder_.AddCharacterClass(elements, is_negated);
}

// parse a c style string with utf8 support
int32_t EBNFParserImpl::ParseString() {
  std::vector<int32_t> character_classes;
  while (Peek() && Peek() != '\"') {
    if (Peek() == '\r' || Peek() == '\n') {
      ThrowParseError("String should not contain newline");
    }
    auto [codepoint, len] = Utf8OrEscapeToCodepoint(cur_);
    if (codepoint == static_cast<TCodepoint>(CharHandlingError::kInvalidUtf8)) {
      ThrowParseError("Invalid utf8 sequence");
    }
    if (codepoint == static_cast<TCodepoint>(CharHandlingError::kInvalidEscape)) {
      ThrowParseError("Invalid escape sequence");
    }
    Consume(len);
    character_classes.push_back(builder_.AddCharacterClass({{codepoint, codepoint}}));
  }
  if (character_classes.empty()) {
    return builder_.AddEmptyStr();
  }
  return builder_.AddSequence(character_classes);
}

int32_t EBNFParserImpl::ParseRuleRef() {
  std::string name = ParseName();
  auto rule_id = builder_.GetRuleId(name);
  if (rule_id == -1) {
    ThrowParseError("Rule \"" + name + "\" is not defined");
  }
  return builder_.AddRuleRef(rule_id);
}

int32_t EBNFParserImpl::ParseElement() {
  switch (Peek()) {
    case '(': {
      Consume();
      ConsumeSpace();
      auto prev_in_parentheses = in_parentheses_;
      in_parentheses_ = true;
      auto rule_expr_id = ParseChoices();
      ConsumeSpace();
      if (Peek() != ')') {
        ThrowParseError("Expect )");
      }
      Consume();
      in_parentheses_ = prev_in_parentheses;
      return rule_expr_id;
    }
    case '[': {
      Consume();
      auto rule_expr_id = ParseCharacterClass();
      if (Peek() != ']') {
        ThrowParseError("Expect ]");
      }
      Consume();
      return rule_expr_id;
    }
    case '\"': {
      Consume();
      auto rule_expr_id = ParseString();
      if (Peek() != '\"') {
        ThrowParseError("Expect \"");
      }
      Consume();
      return rule_expr_id;
    }
    default: {
      if (IsNameChar(Peek(), true)) {
        return ParseRuleRef();
      }
      ThrowParseError("Expect element");
    }
  }
}

int32_t EBNFParserImpl::HandleStarQuantifier(int32_t rule_expr_id) {
  if (builder_.GetRuleExpr(rule_expr_id).type == BNFGrammarBuilder::RuleExprType::kCharacterClass) {
    // We have special handling for character class star, e.g. [a-z]*
    return builder_.AddCharacterClassStar(rule_expr_id);
  } else {
    // For other star quantifiers, we transform it into a rule:
    // a*  -->  rule ::= a rule | ""
    auto new_rule_name = builder_.GetNewRuleName(cur_rule_name_);
    auto new_rule_id = builder_.AddEmptyRule(new_rule_name);
    auto ref_to_new_rule = builder_.AddRuleRef(new_rule_id);
    auto new_rule_expr_id = builder_.AddChoices(
        {builder_.AddSequence({rule_expr_id, ref_to_new_rule}), builder_.AddEmptyStr()});
    builder_.UpdateRuleBody(new_rule_id, new_rule_expr_id);

    // Return the reference to the new rule
    return builder_.AddRuleRef(new_rule_id);
  }
}

int32_t EBNFParserImpl::HandlePlusQuantifier(int32_t rule_expr_id) {
  // a+  -->  rule ::= a rule | a
  auto new_rule_name = builder_.GetNewRuleName(cur_rule_name_);
  auto new_rule_id = builder_.AddEmptyRule(new_rule_name);
  auto ref_to_new_rule = builder_.AddRuleRef(new_rule_id);
  auto new_rule_expr_id =
      builder_.AddChoices({builder_.AddSequence({rule_expr_id, ref_to_new_rule}), rule_expr_id});
  builder_.UpdateRuleBody(new_rule_id, new_rule_expr_id);

  // Return the reference to the new rule
  return builder_.AddRuleRef(new_rule_id);
}

int32_t EBNFParserImpl::HandleQuestionQuantifier(int32_t rule_expr_id) {
  // a?  -->  rule ::= a | empty
  auto new_rule_name = builder_.GetNewRuleName(cur_rule_name_);
  auto new_rule_expr_id = builder_.AddChoices({rule_expr_id, builder_.AddEmptyStr()});
  auto new_rule_id = builder_.AddRule({new_rule_name, new_rule_expr_id});
  return builder_.AddRuleRef(new_rule_id);
}

int32_t EBNFParserImpl::ParseQuantifier() {
  int32_t rule_expr_id = ParseElement();
  ConsumeSpace(in_parentheses_);
  if (Peek() != '*' && Peek() != '+' && Peek() != '?') {
    return rule_expr_id;
  }
  Consume();

  // We will transform a*, a+, a? into a rule, and return the reference to this rule
  switch (Peek(-1)) {
    case '*':
      // We assume that the star quantifier should be the body of some rule now
      return HandleStarQuantifier(rule_expr_id);
    case '+':
      return HandlePlusQuantifier(rule_expr_id);
    case '?':
      return HandleQuestionQuantifier(rule_expr_id);
    default:
      LOG(FATAL) << "Unreachable";
  }
}

int32_t EBNFParserImpl::ParseSequence() {
  std::vector<int32_t> elements;
  elements.push_back(ParseQuantifier());
  ConsumeSpace(in_parentheses_);
  while (Peek() && Peek() != '|' && Peek() != ')' && Peek() != '\n' && Peek() != '\r') {
    elements.push_back(ParseQuantifier());
    ConsumeSpace(in_parentheses_);
  }
  return builder_.AddSequence(elements);
}

int32_t EBNFParserImpl::ParseChoices() {
  std::vector<int32_t> choices;

  choices.push_back(ParseSequence());
  ConsumeSpace();
  while (Peek() == '|') {
    Consume();
    ConsumeSpace();
    choices.push_back(ParseSequence());
    ConsumeSpace();
  }
  return builder_.AddChoices(choices);
}

EBNFParserImpl::Rule EBNFParserImpl::ParseRule() {
  std::string name = ParseName();
  cur_rule_name_ = name;
  ConsumeSpace();
  if (Peek() != ':' || Peek(1) != ':' || Peek(2) != '=') {
    ThrowParseError("Expect ::=");
  }
  Consume(3);
  ConsumeSpace();
  return {name, ParseChoices()};
}

void EBNFParserImpl::BuildRuleNameToId() {
  ConsumeSpace();
  while (Peek()) {
    auto name = ParseName(true);
    ConsumeSpace(false);
    if (Peek() == ':' && Peek(1) == ':' && Peek(2) == '=') {
      if (name.empty()) {
        ThrowParseError("Expect rule name");
      }
      Consume(3);
      if (builder_.GetRuleId(name) != -1) {
        ThrowParseError("Rule \"" + name + "\" is defined multiple times");
      }
      builder_.AddEmptyRule(name);
    }
    while (Peek() && Peek() != '\n' && Peek() != '\r') {
      Consume();
    }
    ConsumeSpace();
  }
}

void EBNFParserImpl::ResetStringIterator(const char* cur) {
  cur_ = cur;
  cur_line_ = 1;
  cur_column_ = 1;
  cur_rule_name_ = "";
  in_parentheses_ = false;
}

BNFGrammar EBNFParserImpl::DoParse(String ebnf_string) {
  ResetStringIterator(ebnf_string.c_str());
  BuildRuleNameToId();

  ResetStringIterator(ebnf_string.c_str());
  ConsumeSpace();
  while (Peek()) {
    auto new_rule = ParseRule();
    builder_.UpdateRuleBody(new_rule.name, new_rule.body_expr_id);

    ConsumeSpace();
  }

  if (builder_.GetRuleId("main") == -1) {
    ThrowParseError("There must be a rule named \"main\"");
  }

  return builder_.Get();
}

BNFGrammar EBNFParser::Parse(String ebnf_string) {
  EBNFParserImpl parser;
  return parser.DoParse(ebnf_string);
}

BNFGrammar BNFJSONParser::Parse(String json_string) {
  auto node = make_object<BNFGrammarNode>();
  auto grammar_json = json::ParseToJsonObject(json_string);
  auto rules_json = json::Lookup<picojson::array>(grammar_json, "rules");
  for (const auto& rule_json : rules_json) {
    auto rule_json_obj = rule_json.get<picojson::object>();
    auto name = json::Lookup<std::string>(rule_json.get<picojson::object>(), "name");
    auto rule_expr = static_cast<int32_t>(
        json::Lookup<int64_t>(rule_json.get<picojson::object>(), "body_expr_id"));
    node->rules_.push_back(BNFGrammarNode::Rule({name, rule_expr}));
  }
  auto rule_expr_data_json = json::Lookup<picojson::array>(grammar_json, "rule_expr_data");
  for (const auto& data_json : rule_expr_data_json) {
    node->rule_expr_data_.push_back(static_cast<int32_t>(data_json.get<int64_t>()));
  }
  auto rule_expr_indptr_json = json::Lookup<picojson::array>(grammar_json, "rule_expr_indptr");
  for (const auto& index_ptr_json : rule_expr_indptr_json) {
    node->rule_expr_indptr_.push_back(static_cast<int32_t>(index_ptr_json.get<int64_t>()));
  }
  return BNFGrammar(std::move(node));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
