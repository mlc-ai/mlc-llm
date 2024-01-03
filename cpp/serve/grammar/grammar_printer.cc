/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar_printer.cc
 */

#include "grammar_printer.h"

#include "../encoding.h"

namespace mlc {
namespace llm {
namespace serve {

std::string BNFGrammarPrinter::PrintSubrule(TSubruleId subrule_id) {
  std::string result;
  auto [begin, end] = grammar_.GetSubrule(subrule_id);
  switch (*begin) {
    case static_cast<TData>(DataKind::kCharacterRange):
      result += PrintCharacterRange(begin, end, false);
      break;
    case static_cast<TData>(DataKind::kNotCharacterRange):
      result += PrintCharacterRange(begin, end, true);
      break;
    case static_cast<TData>(DataKind::kEmpty):
      result += PrintEmpty();
      break;
    case static_cast<TData>(DataKind::kRuleRef):
      result += PrintRuleRef(begin);
      break;
    case static_cast<TData>(DataKind::kSequence):
      result += PrintSequence(begin, end);
      break;
    case static_cast<TData>(DataKind::kOrRule):
      result += PrintOrRule(begin, end);
      break;
  }
  return result;
}

std::string BNFGrammarPrinter::PrintCharacterRange(const TData* begin, const TData* end,
                                                   bool is_not_range) {
  static const std::unordered_map<TCodepoint, std::string> kCustomEscapeMap = {{'-', "\\-"},
                                                                               {']', "\\]"}};
  std::string result = "[";
  if (is_not_range) {
    result += "^";
  }
  for (auto it = begin + 1; it != end; it += 2) {
    result += CodepointToPrintable(*it, kCustomEscapeMap);
    if (*it == *(it + 1)) {
      continue;
    }
    result += "-";
    result += CodepointToPrintable(*(it + 1), kCustomEscapeMap);
  }
  result += "]";
  return result;
}

std::string BNFGrammarPrinter::PrintEmpty() { return "\"\""; }

std::string BNFGrammarPrinter::PrintRuleRef(const TData* begin) {
  return grammar_.rules_[*(begin + 1)].name;
}

std::string BNFGrammarPrinter::PrintSequence(const TData* begin, const TData* end) {
  std::string result;
  auto prev_require_parentheses = require_parentheses_;
  // If the sequence contains >= 2 elements, and is nested in another subrule with >= 2 elements,
  // we need to print parentheses.
  auto now_require_parentheses = require_parentheses_ && end - begin > 2;
  require_parentheses_ = require_parentheses_ || end - begin > 2;
  if (now_require_parentheses) {
    result += "(";
  }
  for (auto it = begin + 1; it != end; ++it) {
    result += PrintSubrule(*it);
    if (it + 1 != end) {
      result += " ";
    }
  }
  if (now_require_parentheses) {
    result += ")";
  }
  require_parentheses_ = prev_require_parentheses;
  return result;
}

std::string BNFGrammarPrinter::PrintOrRule(const TData* begin, const TData* end) {
  std::string result;

  auto prev_require_parentheses = require_parentheses_;
  auto now_require_parentheses = require_parentheses_ && end - begin > 2;
  require_parentheses_ = require_parentheses_ || end - begin > 2;
  if (now_require_parentheses) {
    result += "(";
  }
  for (auto it = begin + 1; it != end; ++it) {
    result += PrintSubrule(*it);
    if (it + 1 != end) {
      result += " | ";
    }
  }
  if (now_require_parentheses) {
    result += ")";
  }
  require_parentheses_ = prev_require_parentheses;
  return result;
}

std::string BNFGrammarPrinter::PrintGrammar() {
  std::string result;
  for (const auto& rule : grammar_.rules_) {
    result += rule.name + " ::= " + PrintSubrule(rule.subrule) + "\n";
  }
  return result;
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
