/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar_state_matcher_base.h
 * \brief The base class of GrammarStateMatcher. It implements a character-based matching automata.
 */
#ifndef MLC_LLM_SERVE_GRAMMAR_GRAMMAR_STATE_MATCHER_BASE_H_
#define MLC_LLM_SERVE_GRAMMAR_GRAMMAR_STATE_MATCHER_BASE_H_

#include <vector>

#include "../../tokenizers.h"
#include "grammar.h"
#include "grammar_state_matcher_state.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/*! \brief The base class of GrammarStateMatcher. It implements a character-based matching
 * automata, and supports accepting a character, rolling back by character, etc.
 */
class GrammarStateMatcherBase {
 protected:
  using RuleExpr = BNFGrammarNode::RuleExpr;
  using RuleExprType = BNFGrammarNode::RuleExprType;

 public:
  /*!
   * \brief Construct a GrammarStateMatcherBase with the given grammar and initial rule position.
   * \param grammar The grammar to match.
   * \param init_rule_position The initial rule position. If not specified, the main rule will be
   * used.
   */
  GrammarStateMatcherBase(const BNFGrammar& grammar, RulePosition init_rule_position = {})
      : grammar_(grammar), tree_(grammar), stack_tops_history_(&tree_) {
    InitStackState(init_rule_position);
  }

  /*! \brief Accept one codepoint. */
  bool AcceptCodepoint(TCodepoint codepoint, bool verbose = false);

  /*! \brief Check if the end of the main rule is reached. If so, the stop token can be accepted. */
  bool CanReachEnd() const;

  /*! \brief Rollback the matcher to a previous state. */
  void RollbackCodepoints(int rollback_codepoint_cnt);

  /*! \brief Discard the earliest history. */
  void DiscardEarliestCodepoints(int discard_codepoint_cnt);

  /*! \brief Print the stack state. */
  std::string PrintStackState(int steps_behind_latest = 0) const;

 protected:
  // Init the stack state according to the given rule position.
  // If init_rule_position is {}, init the stack with the main rule.
  void InitStackState(RulePosition init_rule_position = {});

  // Update the old stack top to the next position, and push the new stack tops to new_stack_tops.
  void UpdateNewStackTops(int32_t old_node_id, std::vector<int32_t>* new_stack_tops);

  BNFGrammar grammar_;
  RulePositionTree tree_;
  StackTopsHistory stack_tops_history_;

  // Temporary data for AcceptCodepoint.
  std::vector<int32_t> tmp_new_stack_tops_;
};

/*! \brief Check the codepoint is contained in the character class. */
inline bool CharacterClassContains(const BNFGrammarNode::RuleExpr& rule_expr,
                                   TCodepoint codepoint) {
  DCHECK(rule_expr.type == BNFGrammarNode::RuleExprType::kCharacterClass ||
         rule_expr.type == BNFGrammarNode::RuleExprType::kNegCharacterClass);
  for (int i = 0; i < rule_expr.size(); i += 2) {
    if (rule_expr.data[i] <= codepoint && codepoint <= rule_expr.data[i + 1]) {
      return rule_expr.type == BNFGrammarNode::RuleExprType::kCharacterClass;
    }
  }
  return rule_expr.type == BNFGrammarNode::RuleExprType::kNegCharacterClass;
}

inline bool GrammarStateMatcherBase::AcceptCodepoint(TCodepoint codepoint, bool verbose) {
  if (verbose) {
    std::cout << "Stack before accepting: " << PrintStackState() << std::endl;
  }
  tmp_new_stack_tops_.clear();

  const auto& prev_stack_tops = stack_tops_history_.GetLatest();
  for (auto old_top : prev_stack_tops) {
    const auto& rule_position = tree_[old_top];
    auto current_sequence = grammar_->GetRuleExpr(rule_position.sequence_id);
    if (rule_position.parent_id == RulePosition::kNoParent &&
        rule_position.element_id == current_sequence.size()) {
      // This RulePosition means previous elements has matched the complete rule.
      // But we are still need to accept a new character, so this stack will become invalid.
      continue;
    }
    auto current_char_class = grammar_->GetRuleExpr(current_sequence[rule_position.element_id]);
    // Special support for star quantifiers of character classes.
    if (current_char_class.type == RuleExprType::kRuleRef) {
      DCHECK(rule_position.char_class_id != -1);
      current_char_class = grammar_->GetRuleExpr(rule_position.char_class_id);
    }
    DCHECK(current_char_class.type == RuleExprType::kCharacterClass ||
           current_char_class.type == RuleExprType::kNegCharacterClass);
    auto ok = CharacterClassContains(current_char_class, codepoint);
    if (!ok) {
      continue;
    }
    UpdateNewStackTops(old_top, &tmp_new_stack_tops_);
  }
  if (tmp_new_stack_tops_.empty()) {
    if (verbose) {
      std::cout << "Codepoint: " << codepoint << " \"" << CodepointToPrintable(codepoint)
                << "\" Rejected" << std::endl;
    }
    return false;
  }
  stack_tops_history_.PushHistory(tmp_new_stack_tops_);
  if (verbose) {
    std::cout << "Codepoint: " << codepoint << " \"" << CodepointToPrintable(codepoint)
              << "\" Accepted" << std::endl;
    std::cout << "Stack after accepting: " << PrintStackState() << std::endl;
  }
  return true;
}

inline bool GrammarStateMatcherBase::CanReachEnd() const {
  const auto& last_stack_tops = stack_tops_history_.GetLatest();
  return std::any_of(last_stack_tops.begin(), last_stack_tops.end(),
                     [&](int32_t id) { return tree_.IsEndPosition(tree_[id]); });
}

inline void GrammarStateMatcherBase::RollbackCodepoints(int rollback_codepoint_cnt) {
  stack_tops_history_.Rollback(rollback_codepoint_cnt);
}

inline void GrammarStateMatcherBase::DiscardEarliestCodepoints(int discard_codepoint_cnt) {
  stack_tops_history_.DiscardEarliest(discard_codepoint_cnt);
}

inline std::string GrammarStateMatcherBase::PrintStackState(int steps_behind_latest) const {
  return stack_tops_history_.PrintHistory(steps_behind_latest);
}

inline void GrammarStateMatcherBase::InitStackState(RulePosition init_rule_position) {
  if (init_rule_position == kInvalidRulePosition) {
    // Initialize the stack with the main rule.
    auto main_rule = grammar_->GetRule(0);
    auto main_rule_expr = grammar_->GetRuleExpr(main_rule.body_expr_id);
    std::vector<int32_t> new_stack_tops;
    for (auto i : main_rule_expr) {
      DCHECK(grammar_->GetRuleExpr(i).type == RuleExprType::kSequence ||
             grammar_->GetRuleExpr(i).type == RuleExprType::kEmptyStr);
      new_stack_tops.push_back(tree_.NewNode(RulePosition(0, i, 0, RulePosition::kNoParent)));
    }
    stack_tops_history_.PushHistory(new_stack_tops);
  } else {
    stack_tops_history_.PushHistory({tree_.NewNode(init_rule_position)});
  }
}

inline void GrammarStateMatcherBase::UpdateNewStackTops(int32_t old_node_id,
                                                        std::vector<int32_t>* new_stack_tops) {
  const auto& old_rule_position = tree_[old_node_id];
  // For char_class*, the old rule position itself is also the next position
  if (old_rule_position.char_class_id != -1) {
    new_stack_tops->push_back(tree_.NewNode(old_rule_position));
  }

  auto cur_rule_position = tree_.GetNextPosition(tree_[old_node_id]);

  // Continuously iterate to the next position (if reachs the end of the current rule, go to the
  // next position of the parent rule). Push it into new_stack_tops. If this position can not
  // be empty, exit the loop.
  // Positions that can be empty: reference to a rule that can be empty, or a star quantifier
  // rule.
  for (; !tree_.IsEndPosition(cur_rule_position);
       cur_rule_position = tree_.GetNextPosition(cur_rule_position)) {
    auto sequence = grammar_->GetRuleExpr(cur_rule_position.sequence_id);
    auto element = grammar_->GetRuleExpr(sequence[cur_rule_position.element_id]);
    if (element.type == RuleExprType::kCharacterClass ||
        element.type == RuleExprType::kNegCharacterClass) {
      // Character class: cannot be empty. Break the loop.
      new_stack_tops->push_back(tree_.NewNode(cur_rule_position));
      break;
    } else {
      // RuleRef
      DCHECK(element.type == RuleExprType::kRuleRef);
      auto new_rule_id = element[0];
      auto new_rule = grammar_->GetRule(new_rule_id);
      auto new_rule_expr = grammar_->GetRuleExpr(new_rule.body_expr_id);
      if (new_rule_expr.type == RuleExprType::kStarQuantifier) {
        cur_rule_position.char_class_id = new_rule_expr[0];
        new_stack_tops->push_back(tree_.NewNode(cur_rule_position));
      } else {
        DCHECK(new_rule_expr.type == RuleExprType::kChoices);

        bool contain_empty = false;

        // For rule containing choices, expand the rule and push all positions into new_stack_tops
        for (auto j : new_rule_expr) {
          auto sequence = grammar_->GetRuleExpr(j);
          if (sequence.type == RuleExprType::kEmptyStr) {
            contain_empty = true;
            continue;
          }
          DCHECK(sequence.type == RuleExprType::kSequence);
          DCHECK(grammar_->GetRuleExpr(sequence[0]).type == RuleExprType::kCharacterClass ||
                 grammar_->GetRuleExpr(sequence[0]).type == RuleExprType::kNegCharacterClass);
          // Note: rule_position is not inserted to the tree yet, so it need to be inserted first
          auto parent_id = tree_.NewNode(cur_rule_position);
          new_stack_tops->push_back(tree_.NewNode(RulePosition(new_rule_id, j, 0, parent_id)));
        }

        if (!contain_empty) {
          break;
        }
      }
    }
  }

  // Reaches the end of the main rule. Insert a special node to indicate the end.
  if (tree_.IsEndPosition(cur_rule_position)) {
    new_stack_tops->push_back(tree_.NewNode(cur_rule_position));
  }
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_GRAMMAR_STATE_MATCHER_BASE_H_
