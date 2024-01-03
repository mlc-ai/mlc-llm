/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar_parser.cc
 */

#include "grammar_parser.h"

namespace mlc {
namespace llm {
namespace serve {

void EBNFParser::ConsumeSpace(bool allow_newline) {
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

bool EBNFParser::IsNameChar(TCodepoint c, bool first_char) {
  return c == '_' || c == '-' || c == '.' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (!first_char && c >= '0' && c <= '9');
}

// name should be a char string (not a utf8 string)
std::string EBNFParser::ParseName(bool accept_empty) {
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
// [a-z] [ab] [a-zA-Z0-9] [^a-z] [测] [\u0123]
// [a-] and [-a] means a or -
// [a--] means a to -
// [\-] means -
// [\]] means ]
// Character range should not contain newlines.
BNFGrammarImpl::TSubruleId EBNFParser::ParseCharacterRange() {
  static const std::unordered_map<std::string, TCodepoint> kCustomEscapeMap = {{"\\-", '-'},
                                                                               {"\\]", ']'}};

  std::vector<BNFGrammarImpl::CharacterRangeElement> elements;

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

  return bnf_grammar_->InsertCharacterRange(elements);
}

// parse a c style string with utf8 support
BNFGrammarImpl::TSubruleId EBNFParser::ParseString() {
  std::vector<BNFGrammarImpl::TSubruleId> character_ranges;
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
    character_ranges.push_back(bnf_grammar_->InsertCharacterRange({{codepoint, codepoint}}));
  }
  if (character_ranges.empty()) {
    return bnf_grammar_->InsertEmpty();
  }
  return bnf_grammar_->InsertSequence(character_ranges);
}

BNFGrammarImpl::TSubruleId EBNFParser::ParseRuleRef() {
  std::string name = ParseName();
  if (rule_name_to_id_.count(name) == 0) {
    ThrowParseError("Rule " + name + " is not defined");
  }
  auto rule_id = rule_name_to_id_[name];
  return bnf_grammar_->InsertRuleRef(rule_id);
}

BNFGrammarImpl::TSubruleId EBNFParser::ParseElement() {
  switch (Peek()) {
    case '(': {
      Consume();
      ConsumeSpace();
      auto prev_in_parentheses = in_parentheses_;
      in_parentheses_ = true;
      auto subrule_id = ParseOrRule();
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

BNFGrammarImpl::TRuleId EBNFParser::HandleStarQuantifier(BNFGrammarImpl::TSubruleId subrule_id) {
  // a*  -->  rule ::= a rule | empty
  auto new_rule_id = NewRule(cur_rule_name_);
  auto new_rule_ref = bnf_grammar_->InsertRuleRef(new_rule_id);
  auto new_subrule = bnf_grammar_->InsertOrRule(
      {bnf_grammar_->InsertSequence({subrule_id, new_rule_ref}), bnf_grammar_->InsertEmpty()});

  (*bnf_grammar_)[new_rule_id].subrule = new_subrule;
  return new_rule_id;
}

BNFGrammarImpl::TRuleId EBNFParser::HandlePlusQuantifier(BNFGrammarImpl::TSubruleId subrule_id) {
  // a+  -->  rule ::= a rule | a
  auto new_rule_id = NewRule(cur_rule_name_);
  // We will use subrule a for two times in this case
  // So first we create a rule for subrule a
  auto a_rule_id = NewRule(cur_rule_name_);
  (*bnf_grammar_)[a_rule_id].subrule = subrule_id;

  // Then create the new subrule.
  auto a_plus_ref = bnf_grammar_->InsertRuleRef(new_rule_id);
  auto a_ref1 = bnf_grammar_->InsertRuleRef(a_rule_id);
  auto a_ref2 = bnf_grammar_->InsertRuleRef(a_rule_id);
  auto new_subrule =
      bnf_grammar_->InsertOrRule({bnf_grammar_->InsertSequence({a_ref1, a_plus_ref}), a_ref2});
  (*bnf_grammar_)[new_rule_id].subrule = new_subrule;
  return new_rule_id;
}

BNFGrammarImpl::TRuleId EBNFParser::HandleQuestionQuantifier(
    BNFGrammarImpl::TSubruleId subrule_id) {
  // a?  -->  rule ::= a | empty
  auto new_rule_id = NewRule(cur_rule_name_);
  auto new_subrule = bnf_grammar_->InsertOrRule({subrule_id, bnf_grammar_->InsertEmpty()});
  (*bnf_grammar_)[new_rule_id].subrule = new_subrule;
  return new_rule_id;
}

BNFGrammarImpl::TSubruleId EBNFParser::ParseQuantifier() {
  BNFGrammarImpl::TSubruleId subrule_id = ParseElement();
  ConsumeSpace(in_parentheses_);
  if (Peek() != '*' && Peek() != '+' && Peek() != '?') {
    return subrule_id;
  }
  Consume();

  // We will transform a*, a+, a? into a rule, and return the reference to this rule
  switch (Peek(-1)) {
    case '*':
      return bnf_grammar_->InsertRuleRef(HandleStarQuantifier(subrule_id));
    case '+':
      return bnf_grammar_->InsertRuleRef(HandlePlusQuantifier(subrule_id));
    case '?':
      return bnf_grammar_->InsertRuleRef(HandleQuestionQuantifier(subrule_id));
    default:
      LOG(FATAL) << "Unreachable";
  }
}

BNFGrammarImpl::TSubruleId EBNFParser::ParseSequence() {
  std::vector<BNFGrammarImpl::TSubruleId> elements;
  elements.push_back(ParseQuantifier());
  ConsumeSpace(in_parentheses_);
  while (Peek() && Peek() != '|' && Peek() != ')' && Peek() != '\n' && Peek() != '\r') {
    elements.push_back(ParseQuantifier());
    ConsumeSpace(in_parentheses_);
  }
  return bnf_grammar_->InsertSequence(elements);
}

BNFGrammarImpl::TSubruleId EBNFParser::ParseOrRule() {
  std::vector<BNFGrammarImpl::TSubruleId> choices;

  choices.push_back(ParseSequence());
  ConsumeSpace();
  while (Peek() == '|') {
    Consume();
    ConsumeSpace();
    choices.push_back(ParseSequence());
    ConsumeSpace();
  }
  return bnf_grammar_->InsertOrRule(choices);
}

BNFGrammarImpl::Rule EBNFParser::ParseRule() {
  std::string name = ParseName();
  cur_rule_name_ = name;
  ConsumeSpace();
  if (Peek() != ':' || Peek(1) != ':' || Peek(2) != '=') {
    ThrowParseError("Expect ::=");
  }
  Consume(3);
  ConsumeSpace();
  return {name, ParseOrRule()};
}

void EBNFParser::BuildRuleNameToId() {
  ConsumeSpace();
  while (Peek()) {
    auto name = ParseName(true);
    ConsumeSpace(false);
    if (Peek() == ':' && Peek(1) == ':' && Peek(2) == '=') {
      if (name.empty()) {
        ThrowParseError("Expect rule name");
      }
      Consume(3);
      if (rule_name_to_id_.count(name) != 0) {
        ThrowParseError("Rule " + name + " is defined multiple times");
      }
      NewRule(name);
    }
    while (Peek() && Peek() != '\n' && Peek() != '\r') {
      Consume();
    }
    ConsumeSpace();
  }
}

void EBNFParser::Parse(String ebnf_string) {
  Reset(ebnf_string.c_str());
  BuildRuleNameToId();

  Reset(ebnf_string.c_str());
  ConsumeSpace();
  while (Peek()) {
    auto new_rule = ParseRule();
    (*bnf_grammar_)[rule_name_to_id_[new_rule.name]].subrule = new_rule.subrule;

    ConsumeSpace();
  }

  if (rule_name_to_id_.count("main") == 0) {
    ThrowParseError("There must be a rule named main");
  }
}

void EBNFParser::Reset(const char* cur) {
  cur_ = cur;
  cur_line_ = 1;
  cur_column_ = 1;
  cur_rule_name_ = "";
  in_parentheses_ = false;
}

BNFGrammarImpl::TRuleId EBNFParser::NewRule(std::string name_hint) {
  std::string name;
  if (rule_name_to_id_.count(name_hint) == 0) {
    name = name_hint;
  } else {
    int cnt = 1;
    while (rule_name_to_id_.count(name_hint + "_" + std::to_string(cnt)) != 0) {
      ++cnt;
    }
    name = name_hint + "_" + std::to_string(cnt);
  }
  auto rule_id = bnf_grammar_->InsertRule({name, -1});
  rule_name_to_id_[name] = rule_id;
  return rule_id;
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
