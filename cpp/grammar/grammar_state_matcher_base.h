/*!
 *  Copyright (c) 2023 by Contributors
 * \file grammar/grammar_state_matcher_base.h
 * \brief The base class of GrammarStateMatcher. It implements a character-based matching automata.
 */
#ifndef MLC_LLM_SERVE_GRAMMAR_GRAMMAR_STATE_MATCHER_BASE_H_
#define MLC_LLM_SERVE_GRAMMAR_GRAMMAR_STATE_MATCHER_BASE_H_

#include <vector>

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
   * \param expand_init_rule_position Whether to expand the initial rule position to all possible
   * locations. See ExpandRulePosition.
   */
  GrammarStateMatcherBase(const BNFGrammar& grammar,
                          RulePosition init_rule_position = kInvalidRulePosition,
                          bool expand_init_rule_position = true)
      : grammar_(grammar), tree_(grammar), stack_tops_history_(&tree_) {
    PushInitialState(init_rule_position, expand_init_rule_position);
  }

  /*! \brief Accept one character. */
  bool AcceptChar(uint8_t char_value, bool verbose = false);

  /*! \brief Check if the end of the main rule is reached. If so, the stop token can be accepted. */
  bool CanReachEnd() const;

  /*! \brief Rollback the matcher to a previous state by the number of characters. */
  void RollbackChars(int rollback_cnt);

  /*! \brief Discard the earliest history by the number of characters. */
  void DiscardEarliestChars(int discard_cnt);

  /*! \brief Print the stack state. */
  std::string PrintStackState(int steps_behind_latest = 0) const;

 protected:
  // Push an initial stack state according to the given rule position.
  // If init_rule_position is kInvalidRulePosition, init the stack with the main rule.
  void PushInitialState(RulePosition init_rule_position, bool expand_init_rule_position);

  // Check if the character is accepted by the current rule position.
  bool CheckIfAccepted(const RulePosition& rule_position, uint8_t char_value) const;

  /*!
   * \brief Find the next position in the rule. If the next position is at the end of the rule,
   * and consider_parent is true, will iteratively find the next position in the parent rule.
   * \param rule_position The current position.
   * \param consider_parent Whether to consider the parent position if the current position is
   * at the end of the rule.
   * \returns (success, next_rule_position), indicating if the iteration is successful and the
   * next rule position.
   */
  std::pair<bool, RulePosition> GetNextPositionInSequence(const RulePosition& rule_position,
                                                          bool consider_parent) const;

  // Return the updated rule position after accepting the char
  RulePosition UpdatePositionWithChar(const RulePosition& rule_position, uint8_t char_value) const;

  /*!
   * \brief Expand the given rule position to all possible positions approachable in the grammar.
   * The expanded positions must refers to an element (CharacterClass or CharacterClassStar or
   * ByteString) in a rule. Push all new positions into new_stack_tops.
   * \example
   * A ::= "a" B [a-z]* "c"
   * B ::= "b" | ""
   *
   * Input position: (rule=A, position=B)
   * Approachable positions: (rule=B, position="b"), (rule=A, position=[a-z]*),
   * (rule=A, position="c"), since B and [a-z]* can be empty.
   * \param cur_rule_position The current rule position.
   * \param new_stack_tops The vector to store the new stack tops.
   * \param consider_parent Whether consider expanding the elements in the parent rule. Useful for
   * inner recursion.
   * \param first_id_if_inserted An optimization. When cur_rule_position is already inserted to
   * the state tree, pass its id to avoid inserting it again. -1 (ignore it) by default.
   * \return Whether the end of the rule can be reached. Useful for inner recursion.
   */
  bool ExpandRulePosition(RulePosition cur_rule_position, std::vector<int32_t>* new_stack_tops,
                          bool consider_parent = true, int32_t first_id_if_inserted = -1);

  // The matched grammar.
  BNFGrammar grammar_;
  // The tree storing all states
  RulePositionTree tree_;
  // The tracked history of stack tops (each stack top refers to a node in the tree).
  // We store the stack tops in different steps in the history to support rollback.
  StackTopsHistory stack_tops_history_;

  // Temporary data for AcceptChar.
  std::vector<int32_t> tmp_new_stack_tops_;
};

/*! \brief Check the codepoint is contained in the character class. */
inline bool GrammarStateMatcherBase::CheckIfAccepted(const RulePosition& rule_position,
                                                     uint8_t char_value) const {
  auto current_sequence = grammar_->GetRuleExpr(rule_position.sequence_id);
  auto current_element = grammar_->GetRuleExpr(current_sequence[rule_position.element_id]);
  if (current_element.type == RuleExprType::kCharacterClass ||
      current_element.type == RuleExprType::kCharacterClassStar) {
    if (rule_position.left_utf8_bytes > 0) {
      return (char_value & 0xC0) == 0x80;
    }
    auto [accepted, num_bytes, codepoint] = HandleUTF8FirstByte(char_value);
    if (!accepted) {
      return false;
    }
    bool is_negative = static_cast<bool>(current_element[0]);
    if (num_bytes > 1) {
      return is_negative;
    }
    for (int i = 1; i < current_element.size(); i += 2) {
      if (current_element[i] <= char_value && char_value <= current_element[i + 1]) {
        return !is_negative;
      }
    }
    return is_negative;
  } else if (current_element.type == RuleExprType::kByteString) {
    return current_element[rule_position.element_in_string] == char_value;
  } else {
    LOG(FATAL) << "Unexpected RuleExprType in CheckIfAccepted: "
               << static_cast<int>(current_element.type);
  }
}

inline RulePosition GrammarStateMatcherBase::UpdatePositionWithChar(
    const RulePosition& rule_position, uint8_t char_value) const {
  auto current_sequence = grammar_->GetRuleExpr(rule_position.sequence_id);
  auto current_element = grammar_->GetRuleExpr(current_sequence[rule_position.element_id]);
  RulePosition new_rule_position = rule_position;
  switch (current_element.type) {
    case RuleExprType::kCharacterClass: {
      if (rule_position.left_utf8_bytes > 1) {
        new_rule_position.left_utf8_bytes -= 1;
        return new_rule_position;
      } else if (rule_position.left_utf8_bytes == 1) {
        return GetNextPositionInSequence(rule_position, true).second;
      }
      // If no left utf8 bytes, check the first byte to find the left bytes needed.
      DCHECK(rule_position.left_utf8_bytes == 0);
      auto [accepted, num_bytes, codepoint] = HandleUTF8FirstByte(char_value);
      DCHECK(accepted);
      if (num_bytes > 1) {
        new_rule_position.left_utf8_bytes = num_bytes - 1;
        return new_rule_position;
      }
      return GetNextPositionInSequence(rule_position, true).second;
    }
    case RuleExprType::kCharacterClassStar: {
      if (rule_position.left_utf8_bytes >= 1) {
        new_rule_position.left_utf8_bytes -= 1;
      } else {
        DCHECK(rule_position.left_utf8_bytes == 0);
        auto [accepted, num_bytes, codepoint] = HandleUTF8FirstByte(char_value);
        DCHECK(accepted);
        new_rule_position.left_utf8_bytes = num_bytes - 1;
      }
      return new_rule_position;
    }
    case RuleExprType::kByteString: {
      if (rule_position.element_in_string + 1 < current_element.size()) {
        new_rule_position.element_in_string += 1;
        return new_rule_position;
      }
      return GetNextPositionInSequence(rule_position, true).second;
    }
    default:
      LOG(FATAL) << "Unexpected RuleExprType in UpdatePositionWithChar: "
                 << static_cast<int>(current_element.type);
  }
}

inline bool GrammarStateMatcherBase::AcceptChar(uint8_t char_value, bool verbose) {
  if (verbose) {
    LOG(INFO) << "Matching char: " << static_cast<int>(char_value) << " \""
              << PrintAsEscaped(char_value) << "\"";
    LOG(INFO) << "Previous stack: " << PrintStackState();
  }
  const auto& prev_stack_tops = stack_tops_history_.GetLatest();

  tmp_new_stack_tops_.clear();
  for (auto prev_top : prev_stack_tops) {
    auto cur_rule_position = tree_[prev_top];
    auto current_sequence = grammar_->GetRuleExpr(cur_rule_position.sequence_id);
    if (cur_rule_position.parent_id == RulePosition::kNoParent &&
        cur_rule_position.element_id == current_sequence.size()) {
      // This RulePosition means previous elements has matched the complete rule.
      // But we are still need to accept a new character, so this stack will become invalid.
      continue;
    }

    auto accepted = CheckIfAccepted(cur_rule_position, char_value);
    if (!accepted) {
      continue;
    }

    auto new_rule_position = UpdatePositionWithChar(cur_rule_position, char_value);

    if (new_rule_position == cur_rule_position) {
      ExpandRulePosition(new_rule_position, &tmp_new_stack_tops_, true, prev_top);
    } else {
      ExpandRulePosition(new_rule_position, &tmp_new_stack_tops_, true);
    }
  }
  if (tmp_new_stack_tops_.empty()) {
    if (verbose) {
      LOG(INFO) << "Character " << static_cast<int>(char_value) << " \""
                << PrintAsEscaped(char_value) << "\" Rejected";
    }
    return false;
  }
  stack_tops_history_.PushHistory(tmp_new_stack_tops_);
  if (verbose) {
    LOG(INFO) << "Character: " << static_cast<int>(char_value) << " \""
              << PrintAsEscaped(char_value) << "\" Accepted";
    LOG(INFO) << "New stack after acceptance: " << PrintStackState();
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

inline void GrammarStateMatcherBase::RollbackChars(int rollback_cnt) {
  stack_tops_history_.Rollback(rollback_cnt);
}

inline void GrammarStateMatcherBase::DiscardEarliestChars(int discard_cnt) {
  stack_tops_history_.DiscardEarliest(discard_cnt);
}

inline std::string GrammarStateMatcherBase::PrintStackState(int steps_behind_latest) const {
  return stack_tops_history_.PrintHistory(steps_behind_latest);
}

inline void GrammarStateMatcherBase::PushInitialState(RulePosition init_rule_position,
                                                      bool expand_init_rule_position) {
  if (init_rule_position == kInvalidRulePosition) {
    // Initialize the stack with the main rule.
    auto main_rule = grammar_->GetMainRule();
    auto main_rule_body = grammar_->GetRuleExpr(main_rule.body_expr_id);
    std::vector<int32_t> stack_tops;
    for (auto i : main_rule_body) {
      auto init_rule_position = RulePosition(0, i, 0, RulePosition::kNoParent);
      if (expand_init_rule_position) {
        ExpandRulePosition(init_rule_position, &stack_tops, true);
      } else {
        stack_tops.push_back(tree_.NewNode(init_rule_position));
      }
    }
    stack_tops_history_.PushHistory(stack_tops);
  } else {
    if (expand_init_rule_position) {
      std::vector<int32_t> stack_tops;
      ExpandRulePosition(init_rule_position, &stack_tops, true);
      stack_tops_history_.PushHistory(stack_tops);
    } else {
      stack_tops_history_.PushHistory({tree_.NewNode(init_rule_position)});
    }
  }
}

inline std::pair<bool, RulePosition> GrammarStateMatcherBase::GetNextPositionInSequence(
    const RulePosition& rule_position, bool consider_parent) const {
  auto sequence = grammar_->GetRuleExpr(rule_position.sequence_id);

  auto next_position = rule_position;
  next_position.element_id += 1;
  next_position.element_in_string = 0;
  next_position.left_utf8_bytes = 0;

  DCHECK(next_position.element_id <= sequence.size());

  if (next_position.element_id < sequence.size()) {
    return {true, next_position};
  }

  if (!consider_parent) {
    return {false, kInvalidRulePosition};
  }

  // Find the next position in the parent rule
  while (next_position.parent_id != RulePosition::kNoParent) {
    next_position = tree_[next_position.parent_id];
    next_position.element_id += 1;
    DCHECK(next_position.element_in_string == 0);
    DCHECK(next_position.left_utf8_bytes == 0);

    sequence = grammar_->GetRuleExpr(next_position.sequence_id);
    DCHECK(next_position.element_id <= sequence.size());

    if (next_position.element_id < sequence.size()) {
      break;
    }
  }

  return {true, next_position};
}

inline bool GrammarStateMatcherBase::ExpandRulePosition(RulePosition cur_rule_position,
                                                        std::vector<int32_t>* new_stack_tops,
                                                        bool consider_parent,
                                                        int32_t first_id_if_inserted) {
  bool is_first = false;
  bool is_iteration_successful = true;

  for (; is_iteration_successful;
       std::tie(is_iteration_successful, cur_rule_position) =
           GetNextPositionInSequence(cur_rule_position, consider_parent)) {
    // Insert the node to the tree, if not inserted before.
    int32_t new_node_id;
    if (is_first && first_id_if_inserted != -1) {
      new_node_id = first_id_if_inserted;
    } else {
      new_node_id = tree_.NewNode(cur_rule_position);
    }
    is_first = false;

    // Case 1. The current position points to the end of the grammar.
    if (consider_parent) {
      if (tree_.IsEndPosition(cur_rule_position)) {
        new_stack_tops->push_back(new_node_id);
        return true;
      }
    } else {
      DCHECK(!tree_.IsEndPosition(cur_rule_position));
    }

    auto sequence = grammar_->GetRuleExpr(cur_rule_position.sequence_id);
    auto element = grammar_->GetRuleExpr(sequence[cur_rule_position.element_id]);
    bool can_be_empty = false;

    if (element.type == RuleExprType::kRuleRef) {
      // Case 2. The current position refers to another rule.
      auto ref_rule = grammar_->GetRule(element[0]);
      auto ref_rule_body = grammar_->GetRuleExpr(ref_rule.body_expr_id);
      DCHECK(ref_rule_body.type == RuleExprType::kChoices);

      for (auto sequence_id : ref_rule_body) {
        auto ref_rule_sequence = grammar_->GetRuleExpr(sequence_id);
        if (ref_rule_sequence.type == RuleExprType::kEmptyStr) {
          can_be_empty = true;
          continue;
        }
        auto ref_rule_position = RulePosition(element[0], sequence_id, 0, new_node_id);
        // Find the positions in every choice of the referred rule
        can_be_empty |= ExpandRulePosition(ref_rule_position, new_stack_tops, false);
      }
    } else if (element.type == RuleExprType::kCharacterClass ||
               element.type == RuleExprType::kByteString) {
      // Case 3. Character class or byte string. cannot be empty.
      new_stack_tops->push_back(new_node_id);
      can_be_empty = false;
    } else {
      DCHECK(element.type == RuleExprType::kCharacterClassStar);
      // Case 4. Character class star. Might be empty.
      new_stack_tops->push_back(new_node_id);
      can_be_empty = cur_rule_position.left_utf8_bytes == 0;
    }

    if (!can_be_empty) {
      return false;
    }
  }
  return true;
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_GRAMMAR_STATE_MATCHER_BASE_H_
