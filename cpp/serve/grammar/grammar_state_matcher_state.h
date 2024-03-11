/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar_state_matcher_state.h
 * \brief The header for the definition of the state used in the grammar state matcher.
 */
#ifndef MLC_LLM_SERVE_GRAMMAR_GRAMMAR_STATE_MATCHER_STATE_H_
#define MLC_LLM_SERVE_GRAMMAR_GRAMMAR_STATE_MATCHER_STATE_H_

#include <queue>
#include <vector>

#include "grammar.h"
#include "grammar_serializer.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/*! \brief Specifies a position in a rule. */
struct RulePosition {
  /*! \brief The rule's id. */
  int32_t rule_id = -1;
  /*! \brief Which choice in this rule is selected. */
  int32_t sequence_id = -1;
  /*! \brief Which element of the choice sequence is being visited. */
  int32_t element_id = -1;
  /*!
   * \brief If the element refers to another rule, and the body of another rule is a
   * CharacterClassStar RuleExpr, this field will be set to the id of the character class.
   * This is for the special support of CharacterClassStar.
   */
  int32_t char_class_star_id = -1;
  /*! \brief The id of the parent node in the RulePositionTree. */
  int32_t parent_id = -1;
  /*! \brief The reference count of this RulePosition. If reduces to zero, the node will be
   * removed from the RulePositionBuffer. */
  int reference_count = 0;

  /*! \brief A parent_id value of kNoParent means this RulePosition is the root of the tree. */
  static constexpr int32_t kNoParent = -1;

  constexpr RulePosition() = default;
  constexpr RulePosition(int32_t rule_id, int32_t sequence_id, int32_t element_id,
                         int32_t parent_id = kNoParent, int32_t char_class_star_id = -1)
      : rule_id(rule_id),
        sequence_id(sequence_id),
        element_id(element_id),
        char_class_star_id(char_class_star_id),
        parent_id(parent_id) {}

  bool operator==(const RulePosition& other) const {
    return rule_id == other.rule_id && sequence_id == other.sequence_id &&
           element_id == other.element_id && char_class_star_id == other.char_class_star_id &&
           parent_id == other.parent_id;
  }

  bool operator!=(const RulePosition& other) const { return !(*this == other); }
};

/*! \brief A special value for invalid RulePosition. */
inline constexpr RulePosition kInvalidRulePosition(-1, -1, -1, -1, -1);

/*! \brief A buffer to manage all RulePositions. */
class RulePositionBuffer {
 public:
  /*!
   * \brief Allocate a new RulePosition. with given initial value.
   * \returns The id of the allocated node.
   */
  int32_t Allocate(RulePosition rule_position) {
    int32_t id;
    if (free_nodes_.empty()) {
      buffer_.emplace_back();
      id = buffer_.size() - 1;
    } else {
      id = free_nodes_.back();
      DCHECK(buffer_[id] == kInvalidRulePosition);
      free_nodes_.pop_back();
    }
    rule_position.reference_count = 0;
    buffer_[id] = rule_position;
    return id;
  }

  /*! \brief Free the RulePosition with the given id. */
  void Free(int32_t id) {
    DCHECK(buffer_[id] != kInvalidRulePosition);
    buffer_[id] = kInvalidRulePosition;
    free_nodes_.push_back(id);
  }

  /*! \brief Get the capacity of the buffer. */
  size_t Capacity() const { return buffer_.size(); }

  /*! \brief Get the number of allocated nodes. */
  size_t Size() const {
    DCHECK(buffer_.size() >= free_nodes_.size());
    return buffer_.size() - free_nodes_.size();
  }

  /*! \brief Get the RulePosition with the given id. */
  RulePosition& operator[](int32_t id) { return buffer_[id]; }
  const RulePosition& operator[](int32_t id) const { return buffer_[id]; }

  void Reset() {
    buffer_.clear();
    free_nodes_.clear();
  }

  friend class RulePositionTree;

 private:
  /*! \brief The buffer to store all RulePositions. */
  std::vector<RulePosition> buffer_;
  /*! \brief A stack to store all free node ids. */
  std::vector<int32_t> free_nodes_;
};

/*!
 * \brief A tree structure to store all stacks. Every stack contains several RulePositions, and
 * is represented as a path from the root to a leaf node.
 */
class RulePositionTree {
 public:
  /*! \brief Construct a RulePositionTree associated with the given grammar. */
  RulePositionTree(const BNFGrammar& grammar) : grammar_(grammar) {}

  /*!
   * \brief Create a new node with the given RulePosition. The reference count of the new node
   * is zero.
   *
   * \note Later, this node should either be pointed by some child rule, or become a stack top
   * node (so it will be pointed to by an attached pointer) to be maintained in the
   * reference-counting based memory management.
   */
  int32_t NewNode(const RulePosition& rule_position) {
    auto id = node_buffer_.Allocate(rule_position);
    if (rule_position.parent_id != RulePosition::kNoParent) {
      DCHECK(rule_position.parent_id < static_cast<int32_t>(node_buffer_.Capacity()) &&
             node_buffer_[rule_position.parent_id] != kInvalidRulePosition);
      node_buffer_[rule_position.parent_id].reference_count++;
    }
    return id;
  }

  /*!
   * \brief Check if the given RulePosition points to the end of the grammar. We use
   * (main_rule_id, sequence_id, length_of_sequence) to represent the end position. Here the
   * element_id is the length of the sequence.
   */
  bool IsEndPosition(const RulePosition& rule_position) const;

  /*! \brief Attach an additional reference to the node with the given id. */
  void AttachRefTo(int32_t id) {
    DCHECK(id != RulePosition::kNoParent);
    node_buffer_[id].reference_count++;
  }

  /*! \brief Remove a reference to the node with the given id. If the reference count becomes zero,
   * free the node and recursively all its ancestors with zero reference count. */
  void RemoveRefTo(int32_t id) {
    DCHECK(id != RulePosition::kNoParent);
    auto cur_node = id;
    while (cur_node != RulePosition::kNoParent) {
      node_buffer_[cur_node].reference_count--;
      if (node_buffer_[cur_node].reference_count != 0) {
        break;
      }
      auto next_node = node_buffer_[cur_node].parent_id;
      node_buffer_.Free(cur_node);
      cur_node = next_node;
    }
  }

  /*! \brief Get the RulePosition with the given id. */
  const RulePosition& operator[](int32_t id) const {
    DCHECK(id != RulePosition::kNoParent);
    DCHECK(node_buffer_[id] != kInvalidRulePosition);
    return node_buffer_[id];
  }

  /*! \brief Print the node with the given id to a string. */
  std::string PrintNode(int32_t id) const;

  /*! \brief Print the stack with the given top id to a string. */
  std::string PrintStackByTopId(int32_t top_id) const;

  /*!
   * \brief Check the well-formedness of the tree and the associated buffer. For debug purpose.
   * \details This function checks the following properties:
   * 1. Every node is pointed directly or indirectly by a outside pointer.
   * 2. Every node's reference count is consistent with the actual reference count.
   * 3. All ids and positions are valid.
   * 4. If a node in the buffer is free, it should be equal to kInvalidRulePosition.
   */
  void CheckWellFormed(const std::vector<int32_t>& outside_pointers) const;

  /*! \brief Reset the tree and the associated buffer. */
  void Reset() { node_buffer_.Reset(); }

 private:
  /*! \brief The grammar associated with this RulePositionTree. */
  BNFGrammar grammar_;
  /*! \brief The buffer to store all RulePositions. */
  RulePositionBuffer node_buffer_;
};

/*!
 * \brief A class to maintain the stack tops and its history to support rollback.
 * \details This class helps to maintain nodes by automatically maintaining the attached references.
 * If a node is not existing in any stack in the history record, it will be freed.
 *
 * It can store up to the previous max_rollback_steps + 1 steps of history, and thus supports
 * rolling back up to max_rollback_steps steps.
 */
class StackTopsHistory {
 public:
  /*!
   * \param tree The RulePositionTree to be associated with. Possibly modify the tree by attaching
   * and removing references to the stack top nodes.
   * \param max_rollback_steps The maximum number of rollback steps to be supported.
   */
  StackTopsHistory(RulePositionTree* tree) : tree_(tree) {}

  /*!
   * \brief Push a new history record consisting a list of stack tops. These nodes will be recorded
   * as existing in a stack (by attaching a reference to them).
   * \param stack_tops The stack tops to be pushed.
   * \param drop_old Whether to drop the oldest history record if the history size exceeds the
   * limit. If the history is dropped, node that do not exist in any stack any more will be freed.
   */
  void PushHistory(const std::vector<int32_t>& stack_tops) {
    stack_tops_history_.push_back(stack_tops);
    for (auto id : stack_tops) {
      tree_->AttachRefTo(id);
    }
  }

  /*! \brief Roll back to several previous steps. Possibly frees node that do not exist in any stack
   * any more. */
  void Rollback(int rollback_steps) {
    DCHECK(rollback_steps < stack_tops_history_.size())
        << "The number of requested rollback steps is greater than or equal to the current "
           "history "
        << "size: " << rollback_steps << " vs " << stack_tops_history_.size() << ".";
    while (rollback_steps--) {
      PopLatest();
    }
  }

  /*! \brief Discard the earliest several steps. Possibly frees node that do not exist in any stack
   * any more. */
  void DiscardEarliest(int discard_steps) {
    DCHECK(discard_steps < stack_tops_history_.size())
        << "The number of requested discard steps is greater than or equal to the current "
           "history "
        << "size: " << discard_steps << " vs " << stack_tops_history_.size() << ".";
    while (discard_steps--) {
      PopEarliest();
    }
  }

  /*! \brief Get the latest stack tops. */
  const std::vector<int32_t>& GetLatest() const { return stack_tops_history_.back(); }

  /*!
   * \brief Print one history record.
   * \param history_position_to_latest The number of steps behind the latest record. 0 means the
   * latest record.
   */
  std::string PrintHistory(int history_position_to_latest = 0) const;

  /*! \brief Get the number of history records. */
  int Size() const { return stack_tops_history_.size(); }

  /*! \brief Check the well-formedness of the tree and the associated buffer. */
  void CheckWellFormed() const;

  /*! \brief Reset the history and the associated node tree. */
  void Reset() {
    stack_tops_history_.clear();
    tree_->Reset();
  }

 private:
  /*! \brief Pop the oldest history record. Possibly frees node that do not exist in any stack any
   * more. */
  void PopEarliest() {
    const auto& old_stack_tops = stack_tops_history_.front();
    for (auto id : old_stack_tops) {
      tree_->RemoveRefTo(id);
    }
    stack_tops_history_.pop_front();
  }

  /*! \brief Pop the latest history record. Possibly frees node that do not exist in any stack any
   * more. */
  void PopLatest() {
    const auto& new_stack_tops = stack_tops_history_.back();
    for (auto id : new_stack_tops) {
      tree_->RemoveRefTo(id);
    }
    stack_tops_history_.pop_back();
  }

  /*! \brief Modifiable pointer to the RulePositionTree. */
  RulePositionTree* tree_;
  /*! \brief The history of stack tops. */
  std::deque<std::vector<int32_t>> stack_tops_history_;
};

inline bool RulePositionTree::IsEndPosition(const RulePosition& rule_position) const {
  return rule_position.parent_id == RulePosition::kNoParent &&
         grammar_->GetRuleExpr(rule_position.sequence_id).size() == rule_position.element_id;
}

inline std::string RulePositionTree::PrintNode(int32_t id) const {
  std::stringstream ss;
  const auto& rule_position = node_buffer_[id];
  ss << "id: " << id;
  ss << ", rule " << rule_position.rule_id << ": " << grammar_->GetRule(rule_position.rule_id).name;
  ss << ", sequence " << rule_position.sequence_id << ": "
     << BNFGrammarPrinter(grammar_).PrintRuleExpr(rule_position.sequence_id);
  ss << ", element id: " << rule_position.element_id;
  if (rule_position.char_class_star_id != -1) {
    ss << ", char class " << rule_position.char_class_star_id << ": "
       << BNFGrammarPrinter(grammar_).PrintRuleExpr(rule_position.char_class_star_id) << "*";
  }
  ss << ", parent id: " << rule_position.parent_id
     << ", ref count: " << rule_position.reference_count;
  return ss.str();
}

inline std::string RulePositionTree::PrintStackByTopId(int32_t top_id) const {
  std::stringstream ss;
  std::vector<int32_t> stack;
  for (auto cur_id = top_id; cur_id != RulePosition::kNoParent;
       cur_id = node_buffer_[cur_id].parent_id) {
    stack.push_back(cur_id);
  }
  ss << "{\n";
  for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
    ss << PrintNode(*it) << "\n";
  }
  ss << "}";
  return ss.str();
}

inline void RulePositionTree::CheckWellFormed(const std::vector<int32_t>& outside_pointers) const {
  const auto& buffer = node_buffer_.buffer_;
  std::unordered_set<int32_t> free_nodes_set(node_buffer_.free_nodes_.begin(),
                                             node_buffer_.free_nodes_.end());
  int buffer_size = static_cast<int>(buffer.size());
  std::vector<int> new_reference_counter(buffer_size, 0);
  std::vector<bool> visited(buffer_size, false);
  std::queue<int> visit_queue;
  for (auto id : outside_pointers) {
    CHECK(id >= 0 && id < buffer_size);
    CHECK(buffer[id] != kInvalidRulePosition);
    new_reference_counter[id]++;
    if (visited[id] == false) {
      visited[id] = true;
      visit_queue.push(id);
    }
  }
  while (!visit_queue.empty()) {
    auto cur_id = visit_queue.front();
    visit_queue.pop();
    const auto& rule_position = buffer[cur_id];
    if (rule_position.parent_id != RulePosition::kNoParent) {
      CHECK(rule_position.parent_id >= 0 && rule_position.parent_id < buffer_size);
      CHECK(buffer[rule_position.parent_id] != kInvalidRulePosition);
      new_reference_counter[rule_position.parent_id]++;
      if (visited[rule_position.parent_id] == false) {
        visited[rule_position.parent_id] = true;
        visit_queue.push(rule_position.parent_id);
      }
    }
  }

  for (int i = 0; i < static_cast<int32_t>(buffer.size()); ++i) {
    if (free_nodes_set.count(i)) {
      CHECK(buffer[i] == kInvalidRulePosition);
      CHECK(visited[i] == false);
    } else {
      CHECK(visited[i] == true);
      CHECK(buffer[i] != kInvalidRulePosition);
      CHECK(new_reference_counter[i] == buffer[i].reference_count)
          << "Reference counters unmatch for node #" << i << ": Updated "
          << new_reference_counter[i] << ", Original " << buffer[i].reference_count;
    }
  }
}

inline std::string StackTopsHistory::PrintHistory(int history_position_to_latest) const {
  const auto& latest_tops =
      stack_tops_history_[stack_tops_history_.size() - 1 - history_position_to_latest];
  std::stringstream ss;
  ss << "Stacks tops size: " << latest_tops.size() << std::endl;
  int cnt = 0;
  for (auto id : latest_tops) {
    ss << "Stack #" << cnt << ": " << tree_->PrintStackByTopId(id) << "\n";
    ++cnt;
  }
  return ss.str();
}

inline void StackTopsHistory::CheckWellFormed() const {
  std::vector<int32_t> outside_pointers;
  for (const auto& stack_tops : stack_tops_history_) {
    outside_pointers.insert(outside_pointers.end(), stack_tops.begin(), stack_tops.end());
  }
  tree_->CheckWellFormed(outside_pointers);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_GRAMMAR_STATE_MATCHER_STATE_H_
