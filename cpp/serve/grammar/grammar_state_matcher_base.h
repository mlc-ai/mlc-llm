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

  // Update the char_class_star_id field of the given rule_position, if it refers to a character
  // class star rule.
  void UpdateCharClassStarId(RulePosition* rule_position) const;

  /*!
   * \brief Find the next position in the rule. If the next position is at the end of the rule,
   * the result depends on the consider_parent parameter:
   * - false: kInvalidRulePosition will be returned.
   * - true: the next position of the parent rule will be returned. If the current rule is the root
   * rule, the RulePosition will be returned as is to indicate the end of the grammar.
   * \param rule_position The current position.
   * \param consider_parent Whether to consider the parent position if the current position is at
   * the end of the rule.
   */
  RulePosition IterateToNextPosition(const RulePosition& rule_position, bool consider_parent) const;

  /*!
   * \brief Expand the given rule position (may be a RuleRef element) s.t. every new position is a
   * CharacterClass or refers to a CharacterClassStar rule. Push all new positions into
   * new_stack_tops.
   * \details This method will start from cur_rule_position and continuously iterate to the next
   * position as long as the current position can be empty (e.g. the current position is a
   * reference to an rule that can be empty, or to a character class star rule). If the current
   * position can not be empty, stop expanding. All positions collected will be pushed into
   * new_stack_tops.
   *
   * If the end of the current rule is reached:
   * - If is_outmost_level is true, we can go to the next position in the parent rule.
   * - Otherwise, stop iteration.
   * \param cur_rule_position The current rule position.
   * \param new_stack_tops The vector to store the new stack tops.
   * \param is_outmost_level Whether the current position is the outmost level of the rule.
   * \param first_id_if_inserted Being not -1 means the first node is already inserted. This is the
   * id of the first node. This is used to avoid inserting the same node twice.
   * \return Whether the end of the rule can be reached. Used as the condition of recursion.
   */
  bool ExpandRulePosition(RulePosition cur_rule_position, std::vector<int32_t>* new_stack_tops,
                          bool is_outmost_level, int32_t first_id_if_inserted = -1);

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
  const auto& prev_stack_tops = stack_tops_history_.GetLatest();

  tmp_new_stack_tops_.clear();
  for (auto prev_top : prev_stack_tops) {
    const auto& cur_rule_position = tree_[prev_top];
    auto current_sequence = grammar_->GetRuleExpr(cur_rule_position.sequence_id);
    if (cur_rule_position.parent_id == RulePosition::kNoParent &&
        cur_rule_position.element_id == current_sequence.size()) {
      // This RulePosition means previous elements has matched the complete rule.
      // But we are still need to accept a new character, so this stack will become invalid.
      continue;
    }

    auto current_char_class =
        cur_rule_position.char_class_star_id != -1
            ? grammar_->GetRuleExpr(cur_rule_position.char_class_star_id)
            : grammar_->GetRuleExpr(current_sequence[cur_rule_position.element_id]);
    DCHECK(current_char_class.type == RuleExprType::kCharacterClass ||
           current_char_class.type == RuleExprType::kNegCharacterClass);
    auto ok = CharacterClassContains(current_char_class, codepoint);
    if (!ok) {
      continue;
    }

    if (cur_rule_position.char_class_star_id == -1) {
      auto next_rule_position = IterateToNextPosition(cur_rule_position, true);
      DCHECK(next_rule_position != kInvalidRulePosition);
      ExpandRulePosition(next_rule_position, &tmp_new_stack_tops_, true);
    } else {
      ExpandRulePosition(cur_rule_position, &tmp_new_stack_tops_, true, prev_top);
    }
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
#if TVM_LOG_DEBUG
  stack_tops_history_.CheckWellFormed();
#endif
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
    auto main_rule_body = grammar_->GetRuleExpr(main_rule.body_expr_id);
    std::vector<int32_t> new_stack_tops;
    for (auto i : main_rule_body) {
      auto init_rule_position = RulePosition(0, i, 0, RulePosition::kNoParent);
      UpdateCharClassStarId(&init_rule_position);
      ExpandRulePosition(init_rule_position, &new_stack_tops, true);
    }
    stack_tops_history_.PushHistory(new_stack_tops);
  } else {
    stack_tops_history_.PushHistory({tree_.NewNode(init_rule_position)});
  }
}

inline void GrammarStateMatcherBase::UpdateCharClassStarId(RulePosition* rule_position) const {
  auto rule_expr = grammar_->GetRuleExpr(rule_position->sequence_id);
  auto element = grammar_->GetRuleExpr(rule_expr[rule_position->element_id]);
  if (element.type == RuleExprType::kRuleRef) {
    auto sub_rule_body = grammar_->GetRuleExpr(grammar_->GetRule(element[0]).body_expr_id);
    if (sub_rule_body.type == RuleExprType::kCharacterClassStar) {
      rule_position->char_class_star_id = sub_rule_body[0];
    }
  }
}

inline RulePosition GrammarStateMatcherBase::IterateToNextPosition(
    const RulePosition& rule_position, bool consider_parent) const {
  auto next_position = RulePosition(rule_position.rule_id, rule_position.sequence_id,
                                    rule_position.element_id + 1, rule_position.parent_id);
  auto rule_expr = grammar_->GetRuleExpr(rule_position.sequence_id);
  auto current_sequence_length = rule_expr.size();
  DCHECK(next_position.element_id <= current_sequence_length);

  if (next_position.element_id < current_sequence_length) {
    // Update char_class_star_id if the position refers to a character class star rule.
    UpdateCharClassStarId(&next_position);
    return next_position;
  }

  if (!consider_parent) {
    return kInvalidRulePosition;
  }

  if (next_position.parent_id == RulePosition::kNoParent) {
    return next_position;
  } else {
    auto parent_rule_position = tree_[next_position.parent_id];
    return IterateToNextPosition(parent_rule_position, true);
  }
}

inline bool GrammarStateMatcherBase::ExpandRulePosition(RulePosition cur_rule_position,
                                                        std::vector<int32_t>* new_stack_tops,
                                                        bool is_outmost_level,
                                                        int32_t first_id_if_inserted) {
  bool is_first = false;

  for (; cur_rule_position != kInvalidRulePosition;
       cur_rule_position = IterateToNextPosition(cur_rule_position, is_outmost_level)) {
    // Insert the node to the tree, if not inserted before.
    int32_t new_node_id;
    if (is_first && first_id_if_inserted != -1) {
      new_node_id = first_id_if_inserted;
    } else {
      new_node_id = tree_.NewNode(cur_rule_position);
    }
    is_first = false;

    // Case 1. The current position points to the end of the grammar.
    if (is_outmost_level) {
      if (tree_.IsEndPosition(cur_rule_position)) {
        new_stack_tops->push_back(new_node_id);
        return true;
      }
    } else {
      DCHECK(!tree_.IsEndPosition(cur_rule_position));
    }

    // Case 2. The current position refers to a character class star rule. It can be empty.
    if (cur_rule_position.char_class_star_id != -1) {
      new_stack_tops->push_back(new_node_id);
      continue;
    }

    // Case 3. Character class: cannot be empty.
    auto sequence = grammar_->GetRuleExpr(cur_rule_position.sequence_id);
    auto element = grammar_->GetRuleExpr(sequence[cur_rule_position.element_id]);
    if (element.type == RuleExprType::kCharacterClass ||
        element.type == RuleExprType::kNegCharacterClass) {
      new_stack_tops->push_back(new_node_id);
      return false;
    }

    // Case 4. The current position refers to a normal rule, i.e. a rule of choices of sequences.
    DCHECK(element.type == RuleExprType::kRuleRef);
    auto sub_rule_id = element[0];
    auto sub_rule = grammar_->GetRule(sub_rule_id);
    auto sub_rule_body = grammar_->GetRuleExpr(sub_rule.body_expr_id);
    DCHECK(sub_rule_body.type == RuleExprType::kChoices);

    bool contain_empty = false;

    for (auto sequence_id : sub_rule_body) {
      auto sequence = grammar_->GetRuleExpr(sequence_id);
      if (sequence.type == RuleExprType::kEmptyStr) {
        contain_empty = true;
        continue;
      }
      auto sub_rule_position = RulePosition(sub_rule_id, sequence_id, 0, new_node_id);
      UpdateCharClassStarId(&sub_rule_position);
      contain_empty |= ExpandRulePosition(sub_rule_position, new_stack_tops, false);
    }

    if (!contain_empty) {
      return false;
    }
  }
  return true;
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_GRAMMAR_STATE_MATCHER_BASE_H_
