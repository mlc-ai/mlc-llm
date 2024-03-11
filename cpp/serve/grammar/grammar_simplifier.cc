/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar_simplifier.cc
 */

#include "grammar_simplifier.h"

namespace mlc {
namespace llm {
namespace serve {

/*!
 * \brief Eliminates single-element sequence or choice nodes in the grammar.
 * \example The sequence `(a)` or the choice `(a)` will be replaced by `a` in a rule.
 * \example The rule `A ::= ((b) (((d))))` will be replaced by `A ::= (b d)`.
 */
class SingleElementSequenceOrChoiceEliminator : public BNFGrammarMutator<int32_t, BNFGrammar> {
 public:
  using BNFGrammarMutator::Apply;
  using BNFGrammarMutator::BNFGrammarMutator;

 private:
  int32_t VisitSequence(const RuleExpr& rule_expr) {
    std::vector<int32_t> sequence_ids;
    for (int32_t i : rule_expr) {
      sequence_ids.push_back(VisitExpr(grammar_->GetRuleExpr(i)));
    }
    if (sequence_ids.size() == 1) {
      return sequence_ids[0];
    } else {
      return builder_.AddSequence(sequence_ids);
    }
  }

  int32_t VisitChoices(const RuleExpr& rule_expr) {
    std::vector<int32_t> choice_ids;
    for (int32_t i : rule_expr) {
      choice_ids.push_back(VisitExpr(grammar_->GetRuleExpr(i)));
    }
    if (choice_ids.size() == 1) {
      return choice_ids[0];
    } else {
      return builder_.AddChoices(choice_ids);
    }
  }
};

class NestedRuleUnwrapperImpl : public BNFGrammarMutator<int32_t, BNFGrammar> {
 public:
  using BNFGrammarMutator::BNFGrammarMutator;

  BNFGrammar Apply() final {
    grammar_ = SingleElementSequenceOrChoiceEliminator(grammar_).Apply();
    for (int i = 0; i < static_cast<int>(grammar_->NumRules()); ++i) {
      builder_.AddEmptyRule(grammar_->GetRule(i).name);
    }
    for (int i = 0; i < static_cast<int>(grammar_->NumRules()); ++i) {
      auto rule = grammar_->GetRule(i);
      auto rule_expr = grammar_->GetRuleExpr(rule.body_expr_id);
      cur_rule_name_ = rule.name;
      auto new_body_expr_id = VisitRuleBody(rule_expr);
      builder_.UpdateRuleBody(i, new_body_expr_id);
    }
    return builder_.Get();
  }

 private:
  /*! \brief Visit a RuleExpr as a rule body. */
  int32_t VisitRuleBody(const RuleExpr& rule_expr) {
    switch (rule_expr.type) {
      case RuleExprType::kSequence:
        return builder_.AddChoices({builder_.AddSequence(VisitSequence_(rule_expr))});
      case RuleExprType::kChoices:
        return builder_.AddChoices(VisitChoices_(rule_expr));
      case RuleExprType::kEmptyStr:
        return builder_.AddChoices({builder_.AddEmptyStr()});
      case RuleExprType::kCharacterClass:
      case RuleExprType::kNegCharacterClass:
      case RuleExprType::kRuleRef:
        return builder_.AddChoices({builder_.AddSequence({builder_.AddRuleExpr(rule_expr)})});
      case RuleExprType::kCharacterClassStar:
        return builder_.AddCharacterClassStar(VisitExpr(grammar_->GetRuleExpr(rule_expr[0])));
      default:
        LOG(FATAL) << "Unexpected sequence type: " << static_cast<int>(rule_expr.type);
    }
  }

  /*!
   * \brief Visit a RuleExpr containing choices.
   * \returns A list of new choice RuleExpr ids.
   */
  std::vector<int32_t> VisitChoices_(const RuleExpr& rule_expr) {
    std::vector<int32_t> new_choice_ids;
    bool found_empty = false;
    for (auto i : rule_expr) {
      auto choice_expr = grammar_->GetRuleExpr(i);
      switch (choice_expr.type) {
        case RuleExprType::kSequence:
          VisitSequenceInChoices(choice_expr, &new_choice_ids, &found_empty);
          break;
        case RuleExprType::kChoices:
          VisitChoicesInChoices(choice_expr, &new_choice_ids, &found_empty);
          break;
        case RuleExprType::kEmptyStr:
          found_empty = true;
          break;
        case RuleExprType::kCharacterClass:
        case RuleExprType::kNegCharacterClass:
        case RuleExprType::kRuleRef:
          VisitElementInChoices(choice_expr, &new_choice_ids);
          break;
        case RuleExprType::kCharacterClassStar:
          VisitCharacterClassStarInChoices(choice_expr, &new_choice_ids);
          break;
        default:
          LOG(FATAL) << "Unexpected choice type: " << static_cast<int>(choice_expr.type);
      }
    }
    if (found_empty) {
      new_choice_ids.insert(new_choice_ids.begin(), builder_.AddEmptyStr());
    }
    ICHECK_GE(new_choice_ids.size(), 1);
    return new_choice_ids;
  }

  /*! \brief Visit a sequence RuleExpr that is one of a list of choices. */
  void VisitSequenceInChoices(const RuleExpr& rule_expr, std::vector<int32_t>* new_choice_ids,
                              bool* found_empty) {
    auto sub_sequence_ids = VisitSequence_(rule_expr);
    if (sub_sequence_ids.size() == 0) {
      *found_empty = true;
    } else {
      new_choice_ids->push_back(builder_.AddSequence(sub_sequence_ids));
    }
  }

  /*! \brief Visit a choice RuleExpr that is one of a list of choices. */
  void VisitChoicesInChoices(const RuleExpr& rule_expr, std::vector<int32_t>* new_choice_ids,
                             bool* found_empty) {
    auto sub_choice_ids = VisitChoices_(rule_expr);
    bool contains_empty = builder_.GetRuleExpr(sub_choice_ids[0]).type == RuleExprType::kEmptyStr;
    if (contains_empty) {
      *found_empty = true;
      new_choice_ids->insert(new_choice_ids->end(), sub_choice_ids.begin() + 1,
                             sub_choice_ids.end());
    } else {
      new_choice_ids->insert(new_choice_ids->end(), sub_choice_ids.begin(), sub_choice_ids.end());
    }
  }

  /*! \brief Visit an atom element RuleExpr that is one of a list of choices. */
  void VisitElementInChoices(const RuleExpr& rule_expr, std::vector<int32_t>* new_choice_ids) {
    auto sub_expr_id = builder_.AddRuleExpr(rule_expr);
    new_choice_ids->push_back(builder_.AddSequence({sub_expr_id}));
  }

  /*! \brief Visit a character class star RuleExpr that is one of a list of choices. */
  void VisitCharacterClassStarInChoices(const RuleExpr& rule_expr,
                                        std::vector<int32_t>* new_choice_ids) {
    auto sub_expr_id = builder_.AddRuleExpr(grammar_->GetRuleExpr(rule_expr[0]));
    auto new_star_id = builder_.AddCharacterClassStar(sub_expr_id);
    auto new_rule_id = builder_.AddRuleWithHint(cur_rule_name_ + "_star", new_star_id);
    auto new_rule_ref_id = builder_.AddRuleRef(new_rule_id);
    new_choice_ids->push_back(builder_.AddSequence({new_rule_ref_id}));
  }

  /*!
   * \brief Visit a RuleExpr containing a sequence.
   * \returns A list of new sequence RuleExpr ids.
   */
  std::vector<int32_t> VisitSequence_(const RuleExpr& rule_expr) {
    std::vector<int32_t> new_sequence_ids;
    for (auto i : rule_expr) {
      auto seq_expr = grammar_->GetRuleExpr(i);
      switch (seq_expr.type) {
        case RuleExprType::kSequence:
          VisitSequenceInSequence(seq_expr, &new_sequence_ids);
          break;
        case RuleExprType::kChoices:
          VisitChoiceInSequence(seq_expr, &new_sequence_ids);
          break;
        case RuleExprType::kEmptyStr:
          break;
        case RuleExprType::kCharacterClass:
        case RuleExprType::kNegCharacterClass:
        case RuleExprType::kRuleRef:
          VisitElementInSequence(seq_expr, &new_sequence_ids);
          break;
        case RuleExprType::kCharacterClassStar:
          VisitCharacterClassStarInSequence(seq_expr, &new_sequence_ids);
          break;
        default:
          LOG(FATAL) << "Unexpected sequence type: " << static_cast<int>(seq_expr.type);
      }
    }
    return new_sequence_ids;
  }

  /*! \brief Visit a sequence RuleExpr that is one element in another sequence. */
  void VisitSequenceInSequence(const RuleExpr& rule_expr, std::vector<int32_t>* new_sequence_ids) {
    auto sub_sequence_ids = VisitSequence_(rule_expr);
    new_sequence_ids->insert(new_sequence_ids->end(), sub_sequence_ids.begin(),
                             sub_sequence_ids.end());
  }

  /*! \brief Visit a choice RuleExpr that is one element in a sequence. */
  void VisitChoiceInSequence(const RuleExpr& rule_expr, std::vector<int32_t>* new_sequence_ids) {
    auto sub_choice_ids = VisitChoices_(rule_expr);
    if (sub_choice_ids.size() == 1) {
      auto choice_element_expr = builder_.GetRuleExpr(sub_choice_ids[0]);
      if (choice_element_expr.type != RuleExprType::kEmptyStr) {
        new_sequence_ids->insert(new_sequence_ids->end(), choice_element_expr.begin(),
                                 choice_element_expr.end());
      }
    } else {
      auto new_choice_id = builder_.AddChoices(sub_choice_ids);
      auto new_choice_rule_id = builder_.AddRuleWithHint(cur_rule_name_ + "_choice", new_choice_id);
      new_sequence_ids->push_back(builder_.AddRuleRef(new_choice_rule_id));
    }
  }

  /*! \brief Visit an atom element RuleExpr that is in a sequence. */
  void VisitElementInSequence(const RuleExpr& rule_expr, std::vector<int32_t>* new_sequence_ids) {
    new_sequence_ids->push_back(builder_.AddRuleExpr(rule_expr));
  }

  /*! \brief Visit a character class star RuleExpr that is in a sequence. */
  void VisitCharacterClassStarInSequence(const RuleExpr& rule_expr,
                                         std::vector<int32_t>* new_sequence_ids) {
    auto sub_expr_id = builder_.AddRuleExpr(grammar_->GetRuleExpr(rule_expr[0]));
    auto new_star_id = builder_.AddCharacterClassStar(sub_expr_id);
    auto new_rule_id = builder_.AddRuleWithHint(cur_rule_name_ + "_star", new_star_id);
    auto new_rule_ref_id = builder_.AddRuleRef(new_rule_id);
    new_sequence_ids->push_back(new_rule_ref_id);
  }

  /*! \brief The name of the current rule being visited. */
  std::string cur_rule_name_;
};

BNFGrammar NestedRuleUnwrapper::Apply() { return NestedRuleUnwrapperImpl(grammar_).Apply(); }

}  // namespace serve
}  // namespace llm
}  // namespace mlc
