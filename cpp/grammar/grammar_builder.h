/*!
 *  Copyright (c) 2023 by Contributors
 * \file grammar/grammar_builder.h
 * \brief The header for the building the BNF AST.
 */

#ifndef MLC_LLM_SERVE_GRAMMAR_GRAMMAR_BUILDER_H_
#define MLC_LLM_SERVE_GRAMMAR_GRAMMAR_BUILDER_H_
#include <tvm/runtime/object.h>

#include <cstdint>

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
   * \brief Get the result grammar. This function will also set the main rule to the rule with the
   * specified name. The rule should be already added to the grammar.
   * \param main_rule The name of the main rule. Default is "main".
   */
  BNFGrammar Get(const std::string& main_rule = "main") {
    int32_t main_rule_id = GetRuleId(main_rule);
    CHECK(main_rule_id != -1) << "The in rule with name \"" << main_rule << "\" is not found.";
    grammar_->main_rule_id_ = main_rule_id;

    return BNFGrammar(grammar_);
  }

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
   * \brief Add a RuleExpr for string stored in bytes.
   * \param bytes A vector of int32_t, each representing a byte (0~255) in the string.
   * The string is stored in int32 vector to match the storage format of the grammar.
   */
  int32_t AddByteString(const std::vector<int32_t>& bytes) {
    return AddRuleExpr(
        {RuleExprType::kByteString, bytes.data(), static_cast<int32_t>(bytes.size())});
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
   * \brief Add a RuleExpr for a character class.
   * \param elements A vector of CharacterClassElement, each containing a lower and a upper bound.
   * \param is_negative Whether the character class is negated.
   */
  int32_t AddCharacterClass(const std::vector<CharacterClassElement>& elements,
                            bool is_negative = false) {
    std::vector<int32_t> data;
    data.reserve(1 + elements.size() * 2);
    data.push_back(static_cast<int32_t>(is_negative));
    for (const auto& range : elements) {
      data.push_back(range.lower);
      data.push_back(range.upper);
    }
    return AddRuleExpr(
        {RuleExprType::kCharacterClass, data.data(), static_cast<int32_t>(data.size())});
  }

  /*!
   * \brief Add a RuleExpr for a star quantifier of a character class.
   * \param elements A vector of CharacterClassElement, each containing a lower and a upper bound.
   * \param is_negative Whether the character class is negated.
   */
  int32_t AddCharacterClassStar(const std::vector<CharacterClassElement>& elements,
                                bool is_negative = false) {
    std::vector<int32_t> data;
    data.reserve(1 + elements.size() * 2);
    data.push_back(static_cast<int32_t>(is_negative));
    for (const auto& range : elements) {
      data.push_back(range.lower);
      data.push_back(range.upper);
    }
    return AddRuleExpr(
        {RuleExprType::kCharacterClassStar, data.data(), static_cast<int32_t>(data.size())});
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
    return AddRuleExpr(
        {RuleExprType::kSequence, elements.data(), static_cast<int32_t>(elements.size())});
  }

  /*! \brief Add a RuleExpr for RuleExpr choices.*/
  int32_t AddChoices(const std::vector<int32_t>& choices) {
    return AddRuleExpr(
        {RuleExprType::kChoices, choices.data(), static_cast<int32_t>(choices.size())});
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
    CHECK_EQ(rule_name_to_id_.count(rule.name), 0);
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
    CHECK(rule_id >= 0 && rule_id < static_cast<int32_t>(grammar_->rules_.size()))
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
   * \brief Add a lookahead assertion to a rule referred by the given rule_id. The lookahead
   * assertion should be a sequence RuleExpr id. An id of -1 means no lookahead assertion.
   */
  void AddLookaheadAssertion(int32_t rule_id, int32_t lookahead_assertion_id) {
    CHECK(rule_id < static_cast<int32_t>(grammar_->rules_.size()))
        << "Rule id " << rule_id << " is out of range.";
    CHECK(grammar_->rules_[rule_id].lookahead_assertion_id == -1)
        << "Rule " << rule_id << " already has a lookahead assertion.";
    grammar_->rules_[rule_id].lookahead_assertion_id = lookahead_assertion_id;
  }

  /*!
   * \brief Add a lookahead assertion to a rule referred by the given name. The lookahead
   * assertion should be a sequence RuleExpr id. An id of -1 means no lookahead assertion.
   */
  void AddLookaheadAssertion(std::string rule_name, int32_t lookahead_assertion_id) {
    int32_t rule_id = GetRuleId(rule_name);
    CHECK(rule_id != -1) << "Rule " << rule_name << " is not found.";
    AddLookaheadAssertion(rule_id, lookahead_assertion_id);
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
