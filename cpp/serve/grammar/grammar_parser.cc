/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar_parser.cc
 */

#include "grammar_parser.h"

#include "../../metadata/json_parser.h"
#include "../encoding.h"
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
  using TRuleId = BNFGrammarNode::TRuleId;
  using TSubruleId = BNFGrammarNode::TSubruleId;
  using ParseError = EBNFParser::ParseError;

  // Parsing different parts of the grammar
  std::string ParseName(bool accept_empty = false);
  TSubruleId ParseCharacterRange();
  TSubruleId ParseString();
  TSubruleId ParseRuleRef();
  TSubruleId ParseElement();
  TSubruleId ParseQuantifier();
  TSubruleId ParseSequence();
  TSubruleId ParseChoices();
  Rule ParseRule();

  // Helper functions
  // Helper for ParseQuantifier
  TRuleId HandleStarQuantifier(TSubruleId subrule_id);
  TRuleId HandlePlusQuantifier(TSubruleId subrule_id);
  TRuleId HandleQuestionQuantifier(TSubruleId subrule_id);

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
    throw ParseError(msg + " at line " + std::to_string(cur_line_) + ", column " +
                     std::to_string(cur_column_));
  }

  // The grammar builder
  BNFGrammarBuilder grammar_builder_;
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

// Character range:
// 1. Examples: [a-z] [ab] [a-zA-Z0-9] [^a-z] [测] [\u0123]
// 2. "-" appearing in the start or end of the character range means itself. Only if it appears
// between two characters, it means a range. E.g. [a-] and [-a] means "a" or "-"" [a--] means a to -
// 3. "-" and "]" can be escaped:
// [\-] means -
// [\]] means ]
// Character range should not contain newlines.
EBNFParserImpl::TSubruleId EBNFParserImpl::ParseCharacterRange() {
  static const std::unordered_map<std::string, TCodepoint> kCustomEscapeMap = {{"\\-", '-'},
                                                                               {"\\]", ']'}};

  std::vector<BNFGrammarBuilder::CharacterRangeElement> elements;

  bool is_not_range = false;
  if (Peek() == '^') {
    is_not_range = true;
    Consume();
  }

  bool past_is_hyphen = false;
  bool past_is_single_char = false;
  while (Peek() && Peek() != ']') {
    if (Peek() == '\r' || Peek() == '\n') {
      ThrowParseError("Character range should not contain newline");
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
        ThrowParseError("Invalid character range: lower bound is larger than upper bound");
      }
      elements.back().upper = codepoint;
      past_is_hyphen = false;
      ICHECK(past_is_single_char == false);
    } else {
      elements.push_back({codepoint, -1});
      past_is_single_char = true;
    }
  }

  for (auto& element : elements) {
    if (element.upper == -1) {
      element.upper = element.lower;
    }
  }

  return grammar_builder_.InsertCharacterRange(elements);
}

// parse a c style string with utf8 support
EBNFParserImpl::TSubruleId EBNFParserImpl::ParseString() {
  std::vector<TSubruleId> character_ranges;
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
    character_ranges.push_back(grammar_builder_.InsertCharacterRange({{codepoint, codepoint}}));
  }
  if (character_ranges.empty()) {
    return grammar_builder_.InsertEmpty();
  }
  return grammar_builder_.InsertSequence(character_ranges);
}

EBNFParserImpl::TSubruleId EBNFParserImpl::ParseRuleRef() {
  std::string name = ParseName();
  auto rule_id = grammar_builder_.GetRuleId(name);
  if (rule_id == -1) {
    ThrowParseError("Rule " + name + " is not defined");
  }
  return grammar_builder_.InsertRuleRef(rule_id);
}

EBNFParserImpl::TSubruleId EBNFParserImpl::ParseElement() {
  switch (Peek()) {
    case '(': {
      Consume();
      ConsumeSpace();
      auto prev_in_parentheses = in_parentheses_;
      in_parentheses_ = true;
      auto subrule_id = ParseChoices();
      ConsumeSpace();
      if (Peek() != ')') {
        ThrowParseError("Expect )");
      }
      Consume();
      in_parentheses_ = prev_in_parentheses;
      return subrule_id;
    }
    case '[': {
      Consume();
      auto subrule_id = ParseCharacterRange();
      if (Peek() != ']') {
        ThrowParseError("Expect ]");
      }
      Consume();
      return subrule_id;
    }
    case '\"': {
      Consume();
      auto subrule_id = ParseString();
      if (Peek() != '\"') {
        ThrowParseError("Expect \"");
      }
      Consume();
      return subrule_id;
    }
    default: {
      if (IsNameChar(Peek(), true)) {
        return ParseRuleRef();
      }
      ThrowParseError("Expect element");
    }
  }
  return -1;
}

EBNFParserImpl::TRuleId EBNFParserImpl::HandleStarQuantifier(
    EBNFParserImpl::TSubruleId subrule_id) {
  // a*  -->  rule ::= a rule | empty
  auto new_rule_name = grammar_builder_.GetNewRuleName(cur_rule_name_);
  auto new_rule_id = grammar_builder_.InsertEmptyRule(new_rule_name);
  auto new_rule_ref = grammar_builder_.InsertRuleRef(new_rule_id);
  auto new_subrule =
      grammar_builder_.InsertChoices({grammar_builder_.InsertSequence({subrule_id, new_rule_ref}),
                                      grammar_builder_.InsertEmpty()});
  grammar_builder_.SetRuleBody(new_rule_id, new_subrule);
  return new_rule_id;
}

EBNFParserImpl::TRuleId EBNFParserImpl::HandlePlusQuantifier(
    EBNFParserImpl::TSubruleId subrule_id) {
  // a+  -->  rule ::= a rule | a
  // We will use subrule a for two times in this case
  // So first we create a rule for subrule a
  auto a_rule_name = grammar_builder_.GetNewRuleName(cur_rule_name_);
  auto a_rule_id = grammar_builder_.InsertRule({a_rule_name, subrule_id});

  // Then create the new subrule.
  auto new_rule_name = grammar_builder_.GetNewRuleName(cur_rule_name_);
  auto new_rule_id = grammar_builder_.InsertEmptyRule(new_rule_name);
  auto a_plus_ref = grammar_builder_.InsertRuleRef(new_rule_id);
  auto a_ref1 = grammar_builder_.InsertRuleRef(a_rule_id);
  auto a_ref2 = grammar_builder_.InsertRuleRef(a_rule_id);
  auto new_subrule = grammar_builder_.InsertChoices(
      {grammar_builder_.InsertSequence({a_ref1, a_plus_ref}), a_ref2});
  grammar_builder_.SetRuleBody(new_rule_id, new_subrule);
  return new_rule_id;
}

EBNFParserImpl::TRuleId EBNFParserImpl::HandleQuestionQuantifier(
    EBNFParserImpl::TSubruleId subrule_id) {
  // a?  -->  rule ::= a | empty
  auto new_rule_name = grammar_builder_.GetNewRuleName(cur_rule_name_);
  auto new_subrule = grammar_builder_.InsertChoices({subrule_id, grammar_builder_.InsertEmpty()});
  auto new_rule_id = grammar_builder_.InsertRule({new_rule_name, new_subrule});
  return new_rule_id;
}

EBNFParserImpl::TSubruleId EBNFParserImpl::ParseQuantifier() {
  EBNFParserImpl::TSubruleId subrule_id = ParseElement();
  ConsumeSpace(in_parentheses_);
  if (Peek() != '*' && Peek() != '+' && Peek() != '?') {
    return subrule_id;
  }
  Consume();

  // We will transform a*, a+, a? into a rule, and return the reference to this rule
  switch (Peek(-1)) {
    case '*':
      return grammar_builder_.InsertRuleRef(HandleStarQuantifier(subrule_id));
    case '+':
      return grammar_builder_.InsertRuleRef(HandlePlusQuantifier(subrule_id));
    case '?':
      return grammar_builder_.InsertRuleRef(HandleQuestionQuantifier(subrule_id));
    default:
      LOG(FATAL) << "Unreachable";
  }
}

EBNFParserImpl::TSubruleId EBNFParserImpl::ParseSequence() {
  std::vector<TSubruleId> elements;
  elements.push_back(ParseQuantifier());
  ConsumeSpace(in_parentheses_);
  while (Peek() && Peek() != '|' && Peek() != ')' && Peek() != '\n' && Peek() != '\r') {
    elements.push_back(ParseQuantifier());
    ConsumeSpace(in_parentheses_);
  }
  return grammar_builder_.InsertSequence(elements);
}

EBNFParserImpl::TSubruleId EBNFParserImpl::ParseChoices() {
  std::vector<TSubruleId> choices;

  choices.push_back(ParseSequence());
  ConsumeSpace();
  while (Peek() == '|') {
    Consume();
    ConsumeSpace();
    choices.push_back(ParseSequence());
    ConsumeSpace();
  }
  return grammar_builder_.InsertChoices(choices);
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
      if (grammar_builder_.GetRuleId(name) != -1) {
        ThrowParseError("Rule " + name + " is defined multiple times");
      }
      grammar_builder_.InsertEmptyRule(name);
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
    grammar_builder_.SetRuleBody(new_rule.name, new_rule.subrule);

    ConsumeSpace();
  }

  if (grammar_builder_.GetRuleId("main") == -1) {
    ThrowParseError("There must be a rule named main");
  }

  return grammar_builder_.Finalize();
}

BNFGrammar EBNFParser::Parse(String ebnf_string) {
  EBNFParserImpl parser;
  return parser.DoParse(ebnf_string);
}

TVM_REGISTER_GLOBAL("mlc.serve.BNFGrammarFromEBNFString").set_body_typed([](String ebnf_string) {
  return EBNFParser::Parse(ebnf_string);
});

BNFGrammar BNFJSONParser::Parse(String json_string) {
  auto node = make_object<BNFGrammarNode>();
  auto grammar_json = json::ParseToJsonObject(json_string);
  auto rules_json = json::Lookup<picojson::array>(grammar_json, "rules");
  for (const auto& rule_json : rules_json) {
    auto rule_json_obj = rule_json.get<picojson::object>();
    auto name = json::Lookup<std::string>(rule_json.get<picojson::object>(), "name");
    auto subrule =
        static_cast<int32_t>(json::Lookup<int64_t>(rule_json.get<picojson::object>(), "subrule"));
    node->rules.push_back(BNFGrammarNode::Rule({name, subrule}));
  }
  auto subrule_data_json = json::Lookup<picojson::array>(grammar_json, "subrule_data");
  for (const auto& data_json : subrule_data_json) {
    node->subrule_data.push_back(static_cast<int32_t>(data_json.get<int64_t>()));
  }
  auto subrule_indptr_json = json::Lookup<picojson::array>(grammar_json, "subrule_indptr");
  for (const auto& index_ptr_json : subrule_indptr_json) {
    node->subrule_indptr.push_back(static_cast<int32_t>(index_ptr_json.get<int64_t>()));
  }
  return BNFGrammar(std::move(node));
}

TVM_REGISTER_GLOBAL("mlc.serve.BNFGrammarFromJSON").set_body_typed([](String json_string) {
  return BNFJSONParser::Parse(json_string);
});

}  // namespace serve
}  // namespace llm
}  // namespace mlc
