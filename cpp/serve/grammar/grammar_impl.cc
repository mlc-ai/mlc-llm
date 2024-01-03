/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar_impl.cc
 */
#define PICOJSON_USE_INT64

#include "grammar_impl.h"

#include <picojson.h>

#include "grammar_printer.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

String BNFGrammarImpl::AsString() const { return BNFGrammarPrinter(*this).PrintGrammar(); }

String BNFGrammarImpl::AsJSON(bool prettify) const {
  picojson::object grammar_json;

  picojson::array rules_json;
  for (const auto& rule : rules_) {
    picojson::object rule_json;
    rule_json["name"] = picojson::value(rule.name);
    rule_json["subrule"] = picojson::value(static_cast<int64_t>(rule.subrule));
    rules_json.push_back(picojson::value(rule_json));
  }
  grammar_json["rules"] = picojson::value(rules_json);

  picojson::object subrule_storage_json;
  picojson::array subrule_storage_data_json;
  for (const auto& data : subrule_storage_.data_) {
    subrule_storage_data_json.push_back(picojson::value(static_cast<int64_t>(data)));
  }
  subrule_storage_json["data"] = picojson::value(subrule_storage_data_json);
  picojson::array subrule_storage_start_index_json;
  for (const auto& start_index : subrule_storage_.start_index_) {
    subrule_storage_start_index_json.push_back(picojson::value(static_cast<int64_t>(start_index)));
  }
  subrule_storage_json["start_index"] = picojson::value(subrule_storage_start_index_json);
  grammar_json["subrule_storage_json"] = picojson::value(subrule_storage_json);

  return picojson::value(grammar_json).serialize(prettify);
}

BNFGrammarImpl::TSubruleId BNFGrammarImpl::InsertCharacterRange(
    const std::vector<CharacterRangeElement>& elements) {
  std::vector<TData> data;
  data.push_back(static_cast<TData>(DataKind::kCharacterRange));
  for (const auto& range : elements) {
    data.push_back(range.lower);
    data.push_back(range.upper);
  }
  return subrule_storage_.InsertSubrule(data);
}

BNFGrammarImpl::TSubruleId BNFGrammarImpl::InsertNotCharacterRange(
    const std::vector<CharacterRangeElement>& elements) {
  std::vector<TData> data;
  data.push_back(static_cast<TData>(DataKind::kNotCharacterRange));
  for (const auto& range : elements) {
    data.push_back(range.lower);
    data.push_back(range.upper);
  }
  return subrule_storage_.InsertSubrule(data);
}

BNFGrammarImpl::TSubruleId BNFGrammarImpl::InsertEmpty() {
  std::vector<TData> data;
  data.push_back(static_cast<TData>(DataKind::kEmpty));
  return subrule_storage_.InsertSubrule(data);
}

BNFGrammarImpl::TSubruleId BNFGrammarImpl::InsertRuleRef(TRuleId rule_id) {
  std::vector<TData> data;
  data.push_back(static_cast<TData>(DataKind::kRuleRef));
  data.push_back(rule_id);
  return subrule_storage_.InsertSubrule(data);
}

BNFGrammarImpl::TSubruleId BNFGrammarImpl::InsertSequence(const std::vector<TSubruleId>& elements) {
  std::vector<TData> data;
  data.push_back(static_cast<TData>(DataKind::kSequence));
  data.insert(data.end(), elements.begin(), elements.end());
  return subrule_storage_.InsertSubrule(data);
}

BNFGrammarImpl::TSubruleId BNFGrammarImpl::InsertOrRule(const std::vector<TSubruleId>& choices) {
  std::vector<TData> data;
  data.push_back(static_cast<TData>(DataKind::kOrRule));
  data.insert(data.end(), choices.begin(), choices.end());
  return subrule_storage_.InsertSubrule(data);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
