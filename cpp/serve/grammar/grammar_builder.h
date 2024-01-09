/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar_builder.h
 * \brief The header for the building the BNF AST.
 */

#ifndef MLC_LLM_SERVE_GRAMMAR_GRAMMAR_BUILDER_H_
#define MLC_LLM_SERVE_GRAMMAR_GRAMMAR_BUILDER_H_

#include <tvm/runtime/object.h>

#include "grammar.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm;
using namespace tvm::runtime;

/*!
 * \brief Helper class to build a BNF grammar.
 */
class BNFGrammarBuilder {
 public:
  using TRuleId = BNFGrammarNode::TRuleId;
  using TSubruleId = BNFGrammarNode::TSubruleId;
  using TSubruleData = BNFGrammarNode::TSubruleData;
  using DataKind = BNFGrammarNode::DataKind;
  using Rule = BNFGrammarNode::Rule;
  using Subrule = BNFGrammarNode::Subrule;

  /*! \brief Default constructor. Creates a new grammar object. */
  BNFGrammarBuilder() : grammar_(make_object<BNFGrammarNode>()) {}

  /*!
   * \brief Create grammar containing the rules and subrules of an existing grammar. The old grammar
   * remains unchanged.
   * \param grammar The existing grammar.
   */
  explicit BNFGrammarBuilder(const BNFGrammar& grammar)
      : grammar_(make_object<BNFGrammarNode>(*grammar.get())) {}

  /*! \brief Finalize the grammar building and return the built grammar. */
  BNFGrammar Finalize() { return BNFGrammar(grammar_); }

  /****************** Subrule handling ******************/

  /*! \brief Insert a subrule and return the subrule id. */
  TSubruleId InsertSubrule(const Subrule& subrule) {
    grammar_->subrule_indptr.push_back(grammar_->subrule_data.size());
    grammar_->subrule_data.insert(grammar_->subrule_data.end(), subrule.data,
                                  subrule.data + subrule.size);
    return grammar_->subrule_indptr.size() - 1;
  }

  /*!
   * \brief One element of a character range, containing a lower and a upper bound. Both bounds are
   * inclusive.
   */
  struct CharacterRangeElement {
    int32_t lower;
    int32_t upper;
  };

  /*! \brief Insert subrules for character range.*/
  TSubruleId InsertCharacterRange(const std::vector<CharacterRangeElement>& elements) {
    std::vector<TSubruleData> data;
    data.push_back(static_cast<TSubruleData>(DataKind::kCharacterRange));
    for (const auto& range : elements) {
      data.push_back(range.lower);
      data.push_back(range.upper);
    }
    return InsertSubrule({data.data(), data.size()});
  }

  /*! \brief Insert subrules for character range negation.*/
  TSubruleId InsertNotCharacterRange(const std::vector<CharacterRangeElement>& elements) {
    std::vector<TSubruleData> data;
    data.push_back(static_cast<TSubruleData>(DataKind::kNotCharacterRange));
    for (const auto& range : elements) {
      data.push_back(range.lower);
      data.push_back(range.upper);
    }
    return InsertSubrule({data.data(), data.size()});
  }

  /*! \brief Insert subrules for empty string.*/
  TSubruleId InsertEmpty() {
    std::vector<TSubruleData> data;
    data.push_back(static_cast<TSubruleData>(DataKind::kEmpty));
    return InsertSubrule({data.data(), data.size()});
  }

  /*! \brief Insert subrules for rule reference.*/
  TSubruleId InsertRuleRef(TRuleId rule_id) {
    std::vector<TSubruleData> data;
    data.push_back(static_cast<TSubruleData>(DataKind::kRuleRef));
    data.push_back(rule_id);
    return InsertSubrule({data.data(), data.size()});
  }

  /*! \brief Insert subrules for subrule sequence.*/
  TSubruleId InsertSequence(const std::vector<TSubruleId>& elements) {
    std::vector<TSubruleData> data;
    data.push_back(static_cast<TSubruleData>(DataKind::kSequence));
    data.insert(data.end(), elements.begin(), elements.end());
    return InsertSubrule({data.data(), data.size()});
  }

  /*! \brief Insert subrules for subrule or choices.*/
  TSubruleId InsertOrRule(const std::vector<TSubruleId>& choices) {
    std::vector<TSubruleData> data;
    data.push_back(static_cast<TSubruleData>(DataKind::kOrRule));
    data.insert(data.end(), choices.begin(), choices.end());
    return InsertSubrule({data.data(), data.size()});
  }

  /*! \brief Get the subrule with the given id. */
  Subrule GetSubrule(TSubruleId subrule_id) { return grammar_->GetSubrule(subrule_id); }

  /****************** Rule handling ******************/

  /*! \brief Insert a rule and return the rule id. */
  TRuleId InsertRule(const Rule& rule) {
    TRuleId id = grammar_->rules.size();
    auto rules = grammar_->rules;
    grammar_->rules.push_back(rule);
    rule_name_to_id_[rule.name] = id;
    return id;
  }

  /*! \brief Get the rule with the given id. */
  Rule& GetRule(TRuleId rule_id) { return grammar_->rules[rule_id]; }

  /*!
   * \brief Insert an rule without body, and return the rule id. The rule body should be set later
   * with BNFGrammarBuilder::SetRuleBody. This method is useful for cases where the rule id is
   * required to build the rule body.
   */
  TRuleId InsertEmptyRule(const std::string& name) { return InsertRule({name, -1}); }

  /*!
   * \brief Set the rule body of the given rule, specified by rule id.
   * \sa BNFGrammarBuilder::InsertEmptyRule
   */
  void SetRuleBody(TRuleId rule_id, TSubruleId subrule_id) {
    ICHECK(grammar_->rules[rule_id].subrule == -1);
    grammar_->rules[rule_id].subrule = subrule_id;
  }

  /*!
   * \brief Set the rule body of the given rule, specified by rule name.
   * \sa BNFGrammarBuilder::InsertEmptyRule
   */
  void SetRuleBody(std::string rule_name, TSubruleId subrule_id) {
    TRuleId rule_id = GetRuleId(rule_name);
    ICHECK(rule_id != -1);
    SetRuleBody(rule_id, subrule_id);
  }

  /*!
   * \brief Find a name for a new rule starting with the given name hint. Some integer suffix (_1,
   * _2, ...) will be added to avoid name conflict.
   */
  std::string GetNewRuleName(const std::string& name_hint) {
    if (rule_name_to_id_.count(name_hint) == 0) {
      return name_hint;
    } else {
      int cnt = 1;
      while (rule_name_to_id_.count(name_hint + "_" + std::to_string(cnt)) != 0) {
        ++cnt;
      }
      return name_hint + "_" + std::to_string(cnt);
    }
  }

  /*!
   * \brief Get the rule id of the rule with the given name. Return -1 if not found.
   */
  TRuleId GetRuleId(const std::string& name) const {
    auto it = rule_name_to_id_.find(name);
    if (it == rule_name_to_id_.end()) {
      return -1;
    } else {
      return it->second;
    }
  }

 private:
  // Mutable pointer to the grammar object.
  ObjectPtr<BNFGrammarNode> grammar_;
  // Map from rule name to rule id.
  std::unordered_map<std::string, TRuleId> rule_name_to_id_;
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_GRAMMAR_BUILDER_H_
