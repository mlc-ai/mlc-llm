/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar_serializer.cc
 */

#include "grammar_serializer.h"

#include <picojson.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/registry.h>

#include "../../support/encoding.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

std::string BNFGrammarPrinter::PrintRule(const Rule& rule) {
  return rule.name + " ::= " + PrintRuleExpr(rule.body_expr_id);
}

std::string BNFGrammarPrinter::PrintRule(int32_t rule_id) {
  return PrintRule(grammar_->GetRule(rule_id));
}

std::string BNFGrammarPrinter::PrintRuleExpr(const RuleExpr& rule_expr) {
  std::string result;
  switch (rule_expr.type) {
    case RuleExprType::kCharacterClass:
      return PrintCharacterClass(rule_expr);
    case RuleExprType::kNegCharacterClass:
      return PrintCharacterClass(rule_expr);
    case RuleExprType::kEmptyStr:
      return PrintEmptyStr(rule_expr);
    case RuleExprType::kRuleRef:
      return PrintRuleRef(rule_expr);
    case RuleExprType::kSequence:
      return PrintSequence(rule_expr);
    case RuleExprType::kChoices:
      return PrintChoices(rule_expr);
    case RuleExprType::kCharacterClassStar:
      return PrintCharacterClassStar(rule_expr);
    default:
      LOG(FATAL) << "Unexpected RuleExpr type: " << static_cast<int>(rule_expr.type);
  }
}

std::string BNFGrammarPrinter::PrintRuleExpr(int32_t rule_expr_id) {
  return PrintRuleExpr(grammar_->GetRuleExpr(rule_expr_id));
}

std::string BNFGrammarPrinter::PrintCharacterClass(const RuleExpr& rule_expr) {
  static const std::unordered_map<TCodepoint, std::string> kCustomEscapeMap = {{'-', "\\-"},
                                                                               {']', "\\]"}};
  std::string result = "[";
  if (rule_expr.type == RuleExprType::kNegCharacterClass) {
    result += "^";
  }
  for (auto i = 0; i < rule_expr.data_len; i += 2) {
    result += CodepointToPrintable(rule_expr[i], kCustomEscapeMap);
    if (rule_expr[i] == rule_expr[i + 1]) {
      continue;
    }
    result += "-";
    result += CodepointToPrintable(rule_expr[i + 1], kCustomEscapeMap);
  }
  result += "]";
  return result;
}

std::string BNFGrammarPrinter::PrintEmptyStr(const RuleExpr& rule_expr) { return "\"\""; }

std::string BNFGrammarPrinter::PrintRuleRef(const RuleExpr& rule_expr) {
  return grammar_->GetRule(rule_expr[0]).name;
}

std::string BNFGrammarPrinter::PrintSequence(const RuleExpr& rule_expr) {
  std::string result;
  result += "(";
  for (int i = 0; i < rule_expr.data_len; ++i) {
    result += PrintRuleExpr(rule_expr[i]);
    if (i + 1 != rule_expr.data_len) {
      result += " ";
    }
  }
  result += ")";
  return result;
}

std::string BNFGrammarPrinter::PrintChoices(const RuleExpr& rule_expr) {
  std::string result;

  result += "(";
  for (int i = 0; i < rule_expr.data_len; ++i) {
    result += PrintRuleExpr(rule_expr[i]);
    if (i + 1 != rule_expr.data_len) {
      result += " | ";
    }
  }
  result += ")";
  return result;
}

std::string BNFGrammarPrinter::PrintCharacterClassStar(const RuleExpr& rule_expr) {
  return PrintRuleExpr(rule_expr[0]) + "*";
}

String BNFGrammarPrinter::ToString() {
  std::string result;
  auto num_rules = grammar_->NumRules();
  for (auto i = 0; i < num_rules; ++i) {
    result += PrintRule(grammar_->GetRule(i)) + "\n";
  }
  return result;
}

TVM_REGISTER_GLOBAL("mlc.serve.BNFGrammarToString").set_body_typed([](const BNFGrammar& grammar) {
  return BNFGrammarPrinter(grammar).ToString();
});

String BNFGrammarJSONSerializer::ToString() {
  picojson::object grammar_json;

  picojson::array rules_json;
  for (const auto& rule : grammar_->rules_) {
    picojson::object rule_json;
    rule_json["name"] = picojson::value(rule.name);
    rule_json["body_expr_id"] = picojson::value(static_cast<int64_t>(rule.body_expr_id));
    rules_json.push_back(picojson::value(rule_json));
  }
  grammar_json["rules"] = picojson::value(rules_json);

  picojson::array rule_expr_data_json;
  for (const auto& data : grammar_->rule_expr_data_) {
    rule_expr_data_json.push_back(picojson::value(static_cast<int64_t>(data)));
  }
  grammar_json["rule_expr_data"] = picojson::value(rule_expr_data_json);
  picojson::array rule_expr_indptr_json;
  for (const auto& index_ptr : grammar_->rule_expr_indptr_) {
    rule_expr_indptr_json.push_back(picojson::value(static_cast<int64_t>(index_ptr)));
  }
  grammar_json["rule_expr_indptr"] = picojson::value(rule_expr_indptr_json);

  return picojson::value(grammar_json).serialize(prettify_);
}

TVM_REGISTER_GLOBAL("mlc.serve.BNFGrammarToJSON")
    .set_body_typed([](const BNFGrammar& grammar, bool prettify) {
      return BNFGrammarJSONSerializer(grammar, prettify).ToString();
    });

}  // namespace serve
}  // namespace llm
}  // namespace mlc
