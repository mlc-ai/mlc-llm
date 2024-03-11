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
  using Rule = BNFGrammarNode::Rule;
  using RuleExprType = BNFGrammarNode::RuleExprType;
  using RuleExpr = BNFGrammarNode::RuleExpr;

  /*! \brief Default constructor. Creates a new grammar object. */
  BNFGrammarBuilder() : grammar_(make_object<BNFGrammarNode>()) {}

  /*!
   * \brief Create grammar containing the rules and rule_exprs of an existing grammar. The old
   * grammar remains unchanged.
   * \param grammar The existing grammar.
   */
  explicit BNFGrammarBuilder(const BNFGrammar& grammar)
      : grammar_(make_object<BNFGrammarNode>(*grammar.get())) {
    // for (size_t i = 0; i < grammar_->rules_.size(); ++i) {
    //   rule_name_to_id_[grammar_->rules_[i].name] = i;
    // }
  }

  /*! \brief Get the result grammar. */
  BNFGrammar Get() { return BNFGrammar(grammar_); }

  /****************** RuleExpr handling ******************/

  /*! \brief Add a rule_expr and return the rule_expr id. */
  int32_t AddRuleExpr(const RuleExpr& rule_expr) {
    grammar_->rule_expr_indptr_.push_back(grammar_->rule_expr_data_.size());
    grammar_->rule_expr_data_.push_back(static_cast<int32_t>(rule_expr.type));
    grammar_->rule_expr_data_.push_back(rule_expr.data_len);
    grammar_->rule_expr_data_.insert(grammar_->rule_expr_data_.end(), rule_expr.data,
                                     rule_expr.data + rule_expr.data_len);
    return static_cast<int32_t>(grammar_->rule_expr_indptr_.size()) - 1;
  }

  /*!
   * \brief One element of a character class, containing a lower and a upper bound. Both bounds are
   * inclusive.
   */
  struct CharacterClassElement {
    int32_t lower;
    int32_t upper;
  };

  /*!
   * \brief Add a RuleExpr for character class.
   * \param elements A vector of CharacterClassElement, each containing a lower and a upper bound.
   * \param is_neg_range Whether the character class is negated.
   */
  int32_t AddCharacterClass(const std::vector<CharacterClassElement>& elements,
                            bool is_neg_range = false) {
    std::vector<int32_t> data;
    for (const auto& range : elements) {
      data.push_back(range.lower);
      data.push_back(range.upper);
    }
    auto type = is_neg_range ? RuleExprType::kNegCharacterClass : RuleExprType::kCharacterClass;
    return AddRuleExpr({type, data.data(), static_cast<int32_t>(data.size())});
  }

  /*! \brief Add a RuleExpr for empty string.*/
  int32_t AddEmptyStr() { return AddRuleExpr({RuleExprType::kEmptyStr, nullptr, 0}); }

  /*! \brief Add a RuleExpr for rule reference.*/
  int32_t AddRuleRef(int32_t rule_id) {
    std::vector<int32_t> data;
    data.push_back(rule_id);
    return AddRuleExpr({RuleExprType::kRuleRef, data.data(), static_cast<int32_t>(data.size())});
  }

  /*! \brief Add a RuleExpr for RuleExpr sequence.*/
  int32_t AddSequence(const std::vector<int32_t>& elements) {
    std::vector<int32_t> data;
    data.insert(data.end(), elements.begin(), elements.end());
    return AddRuleExpr({RuleExprType::kSequence, data.data(), static_cast<int32_t>(data.size())});
  }

  /*! \brief Add a RuleExpr for RuleExpr choices.*/
  int32_t AddChoices(const std::vector<int32_t>& choices) {
    std::vector<int32_t> data;
    data.insert(data.end(), choices.begin(), choices.end());
    return AddRuleExpr({RuleExprType::kChoices, data.data(), static_cast<int32_t>(data.size())});
  }

  int32_t AddCharacterClassStar(int32_t element) {
    std::vector<int32_t> data;
    data.push_back(element);
    return AddRuleExpr(
        {RuleExprType::kCharacterClassStar, data.data(), static_cast<int32_t>(data.size())});
  }

  size_t NumRuleExprs() const { return grammar_->NumRuleExprs(); }
  /*! \brief Get the rule_expr with the given id. */
  RuleExpr GetRuleExpr(int32_t rule_expr_id) { return grammar_->GetRuleExpr(rule_expr_id); }

  /****************** Rule handling ******************/

  /*! \brief Add a rule and return the rule id. */
  int32_t AddRule(const Rule& rule) {
    int32_t id = grammar_->rules_.size();
    auto rules = grammar_->rules_;
    grammar_->rules_.push_back(rule);
    ICHECK_EQ(rule_name_to_id_.count(rule.name), 0);
    rule_name_to_id_[rule.name] = id;
    return id;
  }

  int32_t AddRule(const std::string& name, int32_t body_expr_id) {
    return AddRule({name, body_expr_id});
  }

  int32_t AddRuleWithHint(const std::string& name_hint, int32_t body_expr_id) {
    return AddRule({GetNewRuleName(name_hint), body_expr_id});
  }

  size_t NumRules() const { return grammar_->NumRules(); }

  /*! \brief Get the rule with the given id. */
  const Rule& GetRule(int32_t rule_id) const { return grammar_->rules_[rule_id]; }

  /*!
   * \brief Add an rule without body, and return the rule id. The rule body should be set later
   * with BNFGrammarBuilder::UpdateRuleBody. This method is useful for cases where the rule id is
   * required to build the rule body.
   * \sa BNFGrammarBuilder::UpdateRuleBody
   */
  int32_t AddEmptyRule(const std::string& name) { return AddRule({name, -1}); }

  /*!
   * \brief Update the rule body of the given rule, specified by rule id. Can be used to set the
   * rule body of a rule inserted by BNFGrammarBuilder::AddEmptyRule.
   */
  void UpdateRuleBody(int32_t rule_id, int32_t body_expr_id) {
    CHECK(rule_id < static_cast<int32_t>(grammar_->rules_.size()))
        << "Rule id " << rule_id << " is out of range.";
    grammar_->rules_[rule_id].body_expr_id = body_expr_id;
  }

  /*!
   * \brief Update the rule body of the given rule, specified by rule name. Can be used to set the
   * rule body of a rule inserted by BNFGrammarBuilder::AddEmptyRule.
   */
  void UpdateRuleBody(std::string rule_name, int32_t body_expr_id) {
    int32_t rule_id = GetRuleId(rule_name);
    CHECK(rule_id != -1) << "Rule " << rule_name << " is not found.";
    UpdateRuleBody(rule_id, body_expr_id);
  }

  /*!
   * \brief Find a name for a new rule starting with the given name hint. Some integer suffix (_1,
   * _2, ...) may be added to avoid name conflict.
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
  int32_t GetRuleId(const std::string& name) const {
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
  std::unordered_map<std::string, int32_t> rule_name_to_id_;
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_GRAMMAR_BUILDER_H_
