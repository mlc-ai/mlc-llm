/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar_simplifier.h
 * \brief The header for the simplification of the BNF AST.
 */

#ifndef MLC_LLM_SERVE_GRAMMAR_GRAMMAR_SIMPLIFIER_H_
#define MLC_LLM_SERVE_GRAMMAR_GRAMMAR_SIMPLIFIER_H_

#include <queue>
#include <string>

#include "grammar.h"
#include "grammar_builder.h"
#include "grammar_serializer.h"

namespace mlc {
namespace llm {
namespace serve {

/*!
 * \brief Base class for visitors and mutators of the BNF grammar.
 * \tparam T The type of the return value of visitor functions. Typical values:
 * - int32_t: the id of the new rule_expr
 * - void: no return value
 * \tparam ReturnType The type of the return value of the transform function Apply(). Typical values
 * are void (for visitor) and BNFGrammar (for mutator).
 */
template <typename T = int32_t, typename ReturnType = BNFGrammar>
class BNFGrammarMutator {
 public:
  /*!
   * \brief Constructor.
   * \param grammar The grammar to visit or mutate.
   */
  explicit BNFGrammarMutator(const BNFGrammar& grammar) : grammar_(grammar) {}

  /*!
   * \brief Apply the transformation to the grammar, or visit the grammar.
   * \return The transformed grammar, or the visiting result, or void.
   * \note Should be called only once after the mutator is constructed.
   */
  virtual ReturnType Apply() {
    if constexpr (std::is_same<T, int32_t>::value && std::is_same<ReturnType, BNFGrammar>::value) {
      for (int i = 0; i < static_cast<int>(grammar_->NumRules()); ++i) {
        auto rule = grammar_->GetRule(i);
        auto rule_expr = grammar_->GetRuleExpr(rule.body_expr_id);
        auto new_body_expr_id = VisitExpr(rule_expr);
        builder_.AddRule(rule.name, new_body_expr_id);
      }
      return builder_.Get();
    } else if constexpr (!std::is_same<ReturnType, void>::value) {
      return ReturnType();
    }
  }

 protected:
  using Rule = BNFGrammarNode::Rule;
  using RuleExpr = BNFGrammarNode::RuleExpr;
  using RuleExprType = BNFGrammarNode::RuleExprType;

  /*! \brief Visit a RuleExpr. Dispatch to the corresponding Visit function. */
  virtual T VisitExpr(const RuleExpr& rule_expr) {
    switch (rule_expr.type) {
      case RuleExprType::kSequence:
        return VisitSequence(rule_expr);
      case RuleExprType::kChoices:
        return VisitChoices(rule_expr);
      case RuleExprType::kEmptyStr:
        return VisitEmptyStr(rule_expr);
      case RuleExprType::kCharacterClass:
      case RuleExprType::kNegCharacterClass:
        return VisitCharacterClass(rule_expr);
      case RuleExprType::kRuleRef:
        return VisitRuleRef(rule_expr);
      case RuleExprType::kCharacterClassStar:
        return VisitCharacterClassStar(rule_expr);
      default:
        LOG(FATAL) << "Unexpected sequence type: " << static_cast<int>(rule_expr.type);
    }
  }

  /*! \brief Visit a sequence RuleExpr. */
  virtual T VisitSequence(const RuleExpr& rule_expr) {
    if constexpr (std::is_same<T, void>::value) {
      for (auto i : rule_expr) {
        VisitExpr(grammar_->GetRuleExpr(i));
      }
    } else if constexpr (std::is_same<T, int32_t>::value) {
      std::vector<T> sequence_ids;
      for (int32_t i : rule_expr) {
        sequence_ids.push_back(VisitExpr(grammar_->GetRuleExpr(i)));
      }
      return builder_.AddSequence(sequence_ids);
    } else {
      return T();
    }
  }

  /*! \brief Visit a choices RuleExpr. */
  virtual T VisitChoices(const RuleExpr& rule_expr) {
    if constexpr (std::is_same<T, void>::value) {
      for (auto i : rule_expr) {
        VisitExpr(grammar_->GetRuleExpr(i));
      }
    } else if constexpr (std::is_same<T, int32_t>::value) {
      std::vector<int32_t> choice_ids;
      for (int32_t i : rule_expr) {
        choice_ids.push_back(VisitExpr(grammar_->GetRuleExpr(i)));
      }
      return builder_.AddChoices(choice_ids);
    } else {
      return T();
    }
  }

  /*! \brief Visit an element RuleExpr, including empty string, character class, and rule ref. */
  virtual T VisitElement(const RuleExpr& rule_expr) {
    if constexpr (std::is_same<T, void>::value) {
      return;
    } else if constexpr (std::is_same<T, int32_t>::value) {
      return builder_.AddRuleExpr(rule_expr);
    } else {
      return T();
    }
  }

  /*! \brief Visit an empty string RuleExpr. */
  virtual T VisitEmptyStr(const RuleExpr& rule_expr) { return VisitElement(rule_expr); }

  /*! \brief Visit a character class RuleExpr. */
  virtual T VisitCharacterClass(const RuleExpr& rule_expr) { return VisitElement(rule_expr); }

  /*! \brief Visit a rule reference RuleExpr. */
  virtual T VisitRuleRef(const RuleExpr& rule_expr) { return VisitElement(rule_expr); }

  /*! \brief Visit a star quantifier RuleExpr. */
  virtual T VisitCharacterClassStar(const RuleExpr& rule_expr) {
    if constexpr (std::is_same<T, void>::value) {
      VisitExpr(grammar_->GetRuleExpr(rule_expr[0]));
    } else if constexpr (std::is_same<T, int32_t>::value) {
      return builder_.AddCharacterClassStar(VisitExpr(grammar_->GetRuleExpr(rule_expr[0])));
    } else {
      return T();
    }
  }

  /*! \brief The grammar to visit or mutate. */
  BNFGrammar grammar_;
  /*!
   * \brief The builder to build the new grammar. It is empty when the mutator is constructed, and
   * can be used to build a new grammar in subclasses.
   */
  BNFGrammarBuilder builder_;
};

/*!
 * \brief Unwrap the rules containing nested expressions. After unwrapping, each rule will be in
 * the form: `rule_name ::= ("" | (element1_1 element1_2 ...) | (element2_1 element2_2 ...) | ...)`.
 *
 * I.e. a list of choices, each choice is a sequence of elements. Elements can be a character class
 * or a rule reference. And if the rule can be empty, the first choice will be an empty string.
 *
 * \example The rule `A ::= ((a) (((b)) (c)) "")` will be replaced by `A ::= ((a b c))`. One choice
 * containing a sequence of three elements. The empty string is removed.
 * \example The rule `A ::= (a | (b | (c | "")))` will be replaced by
 * `A ::= ("" | (a) | (b) | (c))`. The first choice is an empty string, and each of the other three
 * choices is a sequence containing a single element.
 * \example The rule `A ::= (a | (b (c | d)))` will be replaced by
 * `A ::= ((a) | (b B)), B ::= ((c) | (d))`. A new rule B is created to represent the nested
 * choices.
 */
class NestedRuleUnwrapper : public BNFGrammarMutator<int32_t, BNFGrammar> {
 public:
  using BNFGrammarMutator::BNFGrammarMutator;

  BNFGrammar Apply() final;
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_GRAMMAR_SIMPLIFIER_H_
