/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar_printer.cc
 */

#include "grammar_serializer.h"

#include <picojson.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/registry.h>

#include "../encoding.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

std::string BNFGrammarPrinter::PrintSubrule(TSubruleId subrule_id) {
  std::string result;
  auto subrule = grammar_->GetSubrule(subrule_id);
  switch (subrule[0]) {
    case static_cast<TSubruleData>(DataKind::kCharacterRange):
      result += PrintCharacterRange(subrule);
      break;
    case static_cast<TSubruleData>(DataKind::kNotCharacterRange):
      result += PrintCharacterRange(subrule);
      break;
    case static_cast<TSubruleData>(DataKind::kEmpty):
      result += PrintEmpty(subrule);
      break;
    case static_cast<TSubruleData>(DataKind::kRuleRef):
      result += PrintRuleRef(subrule);
      break;
    case static_cast<TSubruleData>(DataKind::kSequence):
      result += PrintSequence(subrule);
      break;
    case static_cast<TSubruleData>(DataKind::kChoices):
      result += PrintChoices(subrule);
      break;
  }
  return result;
}

std::string BNFGrammarPrinter::PrintCharacterRange(const Subrule& subrule) {
  static const std::unordered_map<TCodepoint, std::string> kCustomEscapeMap = {{'-', "\\-"},
                                                                               {']', "\\]"}};
  std::string result = "[";
  if (static_cast<DataKind>(subrule[0]) == DataKind::kNotCharacterRange) {
    result += "^";
  }
  for (auto i = 1; i < subrule.size; i += 2) {
    result += CodepointToPrintable(subrule[i], kCustomEscapeMap);
    if (subrule[i] == subrule[i + 1]) {
      continue;
    }
    result += "-";
    result += CodepointToPrintable(subrule[i + 1], kCustomEscapeMap);
  }
  result += "]";
  return result;
}

std::string BNFGrammarPrinter::PrintEmpty(const Subrule& subrule) { return "\"\""; }

std::string BNFGrammarPrinter::PrintRuleRef(const Subrule& subrule) {
  return grammar_->GetRule(subrule[1]).name;
}

std::string BNFGrammarPrinter::PrintSequence(const Subrule& subrule) {
  std::string result;
  auto prev_require_parentheses = require_parentheses_;
  // If the sequence contains >= 2 elements, and is nested in another subrule with >= 2 elements,
  // we need to print parentheses.
  auto now_require_parentheses = require_parentheses_ && subrule.size > 2;
  require_parentheses_ = require_parentheses_ || subrule.size > 2;
  if (now_require_parentheses) {
    result += "(";
  }
  for (int i = 1; i < subrule.size; ++i) {
    result += PrintSubrule(subrule[i]);
    if (i + 1 != subrule.size) {
      result += " ";
    }
  }
  if (now_require_parentheses) {
    result += ")";
  }
  require_parentheses_ = prev_require_parentheses;
  return result;
}

std::string BNFGrammarPrinter::PrintChoices(const Subrule& subrule) {
  std::string result;

  auto prev_require_parentheses = require_parentheses_;
  auto now_require_parentheses = require_parentheses_ && subrule.size > 2;
  require_parentheses_ = require_parentheses_ || subrule.size > 2;
  if (now_require_parentheses) {
    result += "(";
  }
  for (int i = 1; i < subrule.size; ++i) {
    result += PrintSubrule(subrule[i]);
    if (i + 1 != subrule.size) {
      result += " | ";
    }
  }
  if (now_require_parentheses) {
    result += ")";
  }
  require_parentheses_ = prev_require_parentheses;
  return result;
}

String BNFGrammarPrinter::ToString() {
  std::string result;
  for (const auto& rule : grammar_->rules) {
    result += rule.name + " ::= " + PrintSubrule(rule.subrule) + "\n";
  }
  return result;
}

TVM_REGISTER_GLOBAL("mlc.serve.BNFGrammarToString").set_body_typed([](const BNFGrammar& grammar) {
  return BNFGrammarPrinter(grammar).ToString();
});

String BNFGrammarJSONSerializer::ToString() {
  picojson::object grammar_json;

  picojson::array rules_json;
  for (const auto& rule : grammar_->rules) {
    picojson::object rule_json;
    rule_json["name"] = picojson::value(rule.name);
    rule_json["subrule"] = picojson::value(static_cast<int64_t>(rule.subrule));
    rules_json.push_back(picojson::value(rule_json));
  }
  grammar_json["rules"] = picojson::value(rules_json);

  picojson::array subrule_data_json;
  for (const auto& data : grammar_->subrule_data) {
    subrule_data_json.push_back(picojson::value(static_cast<int64_t>(data)));
  }
  grammar_json["subrule_data"] = picojson::value(subrule_data_json);
  picojson::array subrule_indptr_json;
  for (const auto& index_ptr : grammar_->subrule_indptr) {
    subrule_indptr_json.push_back(picojson::value(static_cast<int64_t>(index_ptr)));
  }
  grammar_json["subrule_indptr"] = picojson::value(subrule_indptr_json);

  return picojson::value(grammar_json).serialize(prettify_);
}

TVM_REGISTER_GLOBAL("mlc.serve.BNFGrammarToJSON")
    .set_body_typed([](const BNFGrammar& grammar, bool prettify) {
      return BNFGrammarJSONSerializer(grammar, prettify).ToString();
    });

}  // namespace serve
}  // namespace llm
}  // namespace mlc
