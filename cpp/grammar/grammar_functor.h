/*!
 *  Copyright (c) 2023 by Contributors
 * \file grammar/grammar_functor.h
 * \brief The header for the simplification of the BNF AST.
 */

#ifndef MLC_LLM_GRAMMAR_GRAMMAR_FUNCTOR_H_
#define MLC_LLM_GRAMMAR_GRAMMAR_FUNCTOR_H_

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
class BNFGrammarFunctor {
 public:
  /*!
   * \brief Constructor.
   * \param grammar The grammar to visit or mutate.
   */
  explicit BNFGrammarFunctor() {}

  /*!
   * \brief Apply the transformation to the grammar, or visit the grammar.
   * \return The transformed grammar, or the visiting result, or void.
   */
  virtual ReturnType Apply(const BNFGrammar& grammar) {
    Init(grammar);
    if constexpr (std::is_same<T, void>::value) {
      for (int i = 0; i < static_cast<int>(grammar_->NumRules()); ++i) {
        auto rule = grammar_->GetRule(i);
        cur_rule_name_ = rule.name;
        VisitExpr(rule.body_expr_id);
        VisitLookaheadAssertion(rule.lookahead_assertion_id);
      }
    } else if constexpr (std::is_same<T, int32_t>::value &&
                         std::is_same<ReturnType, BNFGrammar>::value) {
      // First add empty rules to ensure the new rule ids the same as the old ones, then update
      // the rule bodies
      for (int i = 0; i < static_cast<int>(grammar_->NumRules()); ++i) {
        builder_.AddEmptyRule(grammar_->GetRule(i).name);
      }
      for (int i = 0; i < static_cast<int>(grammar_->NumRules()); ++i) {
        auto rule = grammar_->GetRule(i);
        cur_rule_name_ = rule.name;
        auto new_body_expr_id = VisitExpr(rule.body_expr_id);
        builder_.UpdateRuleBody(i, new_body_expr_id);
        // Handle lookahead assertion
        builder_.AddLookaheadAssertion(i, VisitLookaheadAssertion(rule.lookahead_assertion_id));
      }
      return builder_.Get(grammar_->GetMainRule().name);
    } else {
      return ReturnType();
    }
  }

 protected:
  using Rule = BNFGrammarNode::Rule;
  using RuleExpr = BNFGrammarNode::RuleExpr;
  using RuleExprType = BNFGrammarNode::RuleExprType;

  /*! \brief Initialize the functor. Should be called at the beginning of Apply(). */
  virtual void Init(const BNFGrammar& grammar) {
    grammar_ = grammar;
    builder_ = BNFGrammarBuilder();
  }

  /*! \brief Visit a lookahead assertion expr referred by id. */
  virtual T VisitLookaheadAssertion(int32_t lookahead_assertion_id) {
    if (lookahead_assertion_id == -1) {
      return -1;
    }
    return VisitExpr(lookahead_assertion_id);
  }

  /*! \brief Visit a RuleExpr by id. */
  virtual T VisitExpr(int32_t old_rule_expr_id) {
    return VisitExpr(grammar_->GetRuleExpr(old_rule_expr_id));
  }

  /*! \brief Visit a RuleExpr. Dispatch to the corresponding Visit function. */
  virtual T VisitExpr(const RuleExpr& rule_expr) {
    switch (rule_expr.type) {
      case RuleExprType::kSequence:
        return VisitSequence(rule_expr);
      case RuleExprType::kChoices:
        return VisitChoices(rule_expr);
      case RuleExprType::kEmptyStr:
        return VisitEmptyStr(rule_expr);
      case RuleExprType::kByteString:
        return VisitByteString(rule_expr);
      case RuleExprType::kCharacterClass:
        return VisitCharacterClass(rule_expr);
      case RuleExprType::kCharacterClassStar:
        return VisitCharacterClassStar(rule_expr);
      case RuleExprType::kRuleRef:
        return VisitRuleRef(rule_expr);
      default:
        LOG(FATAL) << "Unexpected sequence type: " << static_cast<int>(rule_expr.type);
    }
  }

  /*! \brief Visit a choices RuleExpr. */
  virtual T VisitChoices(const RuleExpr& rule_expr) {
    if constexpr (std::is_same<T, void>::value) {
      for (auto i : rule_expr) {
        VisitExpr(i);
      }
    } else if constexpr (std::is_same<T, int32_t>::value) {
      std::vector<int32_t> choice_ids;
      for (int32_t i : rule_expr) {
        choice_ids.push_back(VisitExpr(i));
      }
      return builder_.AddChoices(choice_ids);
    } else {
      return T();
    }
  }

  /*! \brief Visit a sequence RuleExpr. */
  virtual T VisitSequence(const RuleExpr& rule_expr) {
    if constexpr (std::is_same<T, void>::value) {
      for (auto i : rule_expr) {
        VisitExpr(i);
      }
    } else if constexpr (std::is_same<T, int32_t>::value) {
      std::vector<T> sequence_ids;
      for (int32_t i : rule_expr) {
        sequence_ids.push_back(VisitExpr(i));
      }
      return builder_.AddSequence(sequence_ids);
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
  virtual T VisitByteString(const RuleExpr& rule_expr) { return VisitElement(rule_expr); }

  /*! \brief Visit a character class RuleExpr. */
  virtual T VisitCharacterClass(const RuleExpr& rule_expr) { return VisitElement(rule_expr); }

  /*! \brief Visit a star quantifier RuleExpr. */
  virtual T VisitCharacterClassStar(const RuleExpr& rule_expr) { return VisitElement(rule_expr); }

  /*! \brief Visit a rule reference RuleExpr. */
  virtual T VisitRuleRef(const RuleExpr& rule_expr) { return VisitElement(rule_expr); }

  /*! \brief The grammar to visit or mutate. */
  BNFGrammar grammar_;
  /*!
   * \brief The builder to build the new grammar. It is empty when the mutator is constructed, and
   * can be used to build a new grammar in subclasses.
   */
  BNFGrammarBuilder builder_;
  /*! \brief The name of the current rule being visited. */
  std::string cur_rule_name_;
};

/*!
 * \brief Visitor of BNFGrammar.
 * \tparam ReturnType The return type of the Apply() function. Denotes the collected information.
 */
template <typename ReturnType>
using BNFGrammarVisitor = BNFGrammarFunctor<void, ReturnType>;

/*!
 * \brief Mutator of BNFGrammar. The Apply() function returns the updated grammar.
 */
using BNFGrammarMutator = BNFGrammarFunctor<int32_t, BNFGrammar>;

/*!
 * \brief Normalize a BNFGrammar: expand the nested rules, combine consequent sequences and strings,
 * etc.
 */
class BNFGrammarNormalizer : public BNFGrammarMutator {
 public:
  using BNFGrammarMutator::BNFGrammarMutator;

  BNFGrammar Apply(const BNFGrammar& grammar) final;

 private:
  std::vector<std::unique_ptr<BNFGrammarMutator>> GetNormalizerList();
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_GRAMMAR_GRAMMAR_FUNCTOR_H_
