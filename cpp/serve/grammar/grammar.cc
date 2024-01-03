/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar.cc
 */

#include "grammar.h"

#include <tvm/runtime/registry.h>

#include "../../metadata/json_parser.h"
#include "grammar_impl.h"
#include "grammar_parser.h"

namespace mlc {
namespace llm {
namespace serve {

TVM_REGISTER_OBJECT_TYPE(BNFGrammarNode);

BNFGrammar BNFGrammar::FromEBNFString(String ebnf_string) {
  auto node = make_object<BNFGrammarImpl>();
  auto parser = EBNFParser(node.get());
  parser.Parse(ebnf_string);
  return BNFGrammar(std::move(node));
}

BNFGrammar BNFGrammar::FromJSON(String json) {
  auto node = make_object<BNFGrammarImpl>();
  auto grammar_json = json::ParseToJsonObject(json);
  auto rules_json = json::Lookup<picojson::array>(grammar_json, "rules");
  for (const auto& rule_json : rules_json) {
    auto rule_json_obj = rule_json.get<picojson::object>();
    auto name = json::Lookup<std::string>(rule_json.get<picojson::object>(), "name");
    auto subrule =
        static_cast<int32_t>(json::Lookup<int64_t>(rule_json.get<picojson::object>(), "subrule"));
    node->rules_.push_back(BNFGrammarImpl::Rule({name, subrule}));
  }
  auto subrule_storage_json = json::Lookup<picojson::object>(grammar_json, "subrule_storage_json");
  auto subrule_storage_data_json = json::Lookup<picojson::array>(subrule_storage_json, "data");
  for (const auto& data_json : subrule_storage_data_json) {
    node->subrule_storage_.data_.push_back(static_cast<int32_t>(data_json.get<int64_t>()));
  }
  auto subrule_storage_start_index_json =
      json::Lookup<picojson::array>(subrule_storage_json, "start_index");
  for (const auto& start_index_json : subrule_storage_start_index_json) {
    node->subrule_storage_.start_index_.push_back(
        static_cast<int32_t>(start_index_json.get<int64_t>()));
  }
  return BNFGrammar(std::move(node));
}

TVM_REGISTER_GLOBAL("mlc.serve.BNFGrammarFromEBNFString").set_body_typed([](String ebnf_string) {
  return BNFGrammar::FromEBNFString(ebnf_string);
});

TVM_REGISTER_GLOBAL("mlc.serve.BNFGrammarFromJSON").set_body_typed([](String json) {
  return BNFGrammar::FromJSON(json);
});

TVM_REGISTER_GLOBAL("mlc.serve.BNFGrammarAsString").set_body_typed([](const BNFGrammar& grammar) {
  return grammar->AsString();
});

TVM_REGISTER_GLOBAL("mlc.serve.BNFGrammarAsJSON")
    .set_body_typed([](const BNFGrammar& grammar, bool prettify = true) {
      return grammar->AsJSON(prettify);
    });

}  // namespace serve
}  // namespace llm
}  // namespace mlc
