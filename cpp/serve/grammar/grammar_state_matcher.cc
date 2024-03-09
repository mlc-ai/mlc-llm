/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar_state_matcher.cc
 */
#include "grammar_state_matcher.h"

#include <chrono>
#include <queue>

#include "../../tokenizers.h"
#include "grammar.h"
#include "grammar_serializer.h"
#include "grammar_state_matcher_base.h"
#include "grammar_state_matcher_preproc.h"
#include "grammar_state_matcher_state.h"
#include "support.h"

namespace mlc {
namespace llm {
namespace serve {

/*
 * Note on the matching algorithm
 *
 * Given a context-free grammar, we match the characters in a string one by one.
 *
 * We adopt a non-deterministic pushdown automata (NPDA) in matching. To be specific, we maintain
 * several stacks, each of which represents a possible path in the NPDA, and update the stacks
 * during matching.
 *
 * ## Stack Structure (see grammar_state_matcher_state.h)
 * The element of every stack is a RulePosition object, referring a position in the grammar. If a
 * RulePosition is a RuleRef element (referring to another rule), the next element of the stack will
 * be a position in this rule. If a RulePosition is a CharacterClass element, it will be the last
 * in the stack, meaning *the next* character to match.
 *
 * ## Matching Process (see grammar_state_matcher_base.h)
 * When accepting a new character and it is accepted by a stack, the last element of the stack will
 * be advanced to the next position in the grammar. If it gets to the end of the rule, several
 * elements at the end may be popped out, and the last element of the stack will be advanced.
 *
 * One stack may split since there may be multiple possible next positions. In this case, similar
 * stacks with different top elements will be added. When ome stack cannot accept the new character,
 * it will be removed from the stacks.
 *
 * ## Storage of Stacks (see grammar_state_matcher_state.h)
 * Note these stacks form a tree structure as when splitting, the new stacks share the same prefix.
 * We store all RulePositions as a tree, where every path from tree root to a node represents a
 * stack. To represent stack tops, we attach additional pointers pointing the stack top nodes.
 * Also, We maintain a history of the stack top pointers, so we can rollback to the previous state.
 *
 * All tree nodes are maintained by a buffer, and utilize reference counting to recycle. If a node
 * is neither pointed by a stack top pointer, not pointed by some child nodes, it will be freed.
 *
 * ## Example
 * ### Grammar
 * main ::= [a] R
 * R ::= [b] S [c] | [b] [c] T
 * S ::= "" | [c] [d]
 * T ::= [e]
 *
 * ### Previous step
 * Previous accepted string: ab
 * Previous stack tree:
 * A------
 * |  \   \
 * B   D<  E<
 * |
 * C<
 *
 * A: (rule main, choice 0, element 1)
 * B: (rule R, choice 0, element 1)
 * C: (rule S, choice 1, element 0)
 * D: (rule R, choice 0, element 2)
 * E: (rule R, choice 1, element 1)
 * < means the stack top pointers in the previous step.
 * The stacks in the previous step is: (A, B, C), (A, D), (A, E)
 *
 * ### Current step
 * Current accepted string: abc
 * Current stack tree:
 * A-----------------      G<<
 * |     \     \     \
 * B---   D<    E<    H
 * |   \              |
 * C<   F<<           I<<
 *
 * F: (rule S, choice 1, element 1)
 * G: (rule main, choice 0, element 2) (means the matching process has finished, and will be deleted
 * when next char comes)
 * H: (rule R, choice 1, element 2)
 * I: (rule T, choice 0, element 0)
 * << means the stack top pointers in the current step.
 * The stacks in the current step is: (A, B, F), (A, H, I), (G,)
 *
 * ## Preprocess (see grammar_state_matcher_preproc.h)
 * We will store all information about tokens that needed in matching in a GrammarStateInitContext
 * object. Tokens are sorted by codepoint, allowing us to reuse the repeated prefixes between
 * different tokens.
 *
 * For a given position in a rule, if we only consider this rule and its sub-rules during matching,
 * without considering its parent rules (in actual matching, we also need to consider its parent
 * rules), we can already determine that some tokens are acceptable while others are definitely
 * rejected. Therefore, for a position in a rule, we can divide the token set into three categories:
 * - accepted_indices: If a token is accepted by this rule
 * - rejected_indices: If a token is rejected by this rule
 * - uncertain_indices: Whether it can be accepted depends on the information from the parent
 * level during actual matching. To be specific, If this token has a prefix that has not been
 * rejected and has reached the end of this rule, then it is possible for it to be further accepted
 * by the parent rule.
 *
 * During actual matching, we will directly accept or reject the tokens in accepted_indices and
 * rejected_indices, and only consider the tokens in uncertain_indices. That speeds up the matching
 * process.
 */

using namespace tvm::runtime;

TVM_REGISTER_OBJECT_TYPE(GrammarStateMatcherNode);

/* \brief The concrete implementation of GrammarStateMatcherNode. */
class GrammarStateMatcherNodeImpl : public GrammarStateMatcherNode, public GrammarStateMatcherBase {
 private:
  using RuleExpr = BNFGrammarNode::RuleExpr;
  using RuleExprType = BNFGrammarNode::RuleExprType;

 public:
  GrammarStateMatcherNodeImpl(std::shared_ptr<GrammarStateInitContext> init_ctx,
                              int max_rollback_steps = 0)
      : GrammarStateMatcherBase(init_ctx->grammar),
        init_ctx_(init_ctx),
        max_rollback_steps_(max_rollback_steps) {}

  bool AcceptToken(int32_t token_id) final;

  void FindNextTokenBitmask(DLTensor* next_token_bitmask) final;

  void Rollback(int num_tokens) final;

  int MaxRollbackSteps() const final { return max_rollback_steps_; }

  bool IsTerminated() const { return stack_tops_history_.GetLatest().empty(); }

  void ResetState() final {
    stack_tops_history_.Reset();
    token_size_history_.clear();
    InitStackState();
  }

 private:
  /*!
   * \brief If is_uncertain_saved is true, find the next token in uncertain_indices. Otherwise,
   * find the next token that is set to true in uncertain_tokens_bitset.
   * \param iterator_uncertain The helper iterator to iterate over uncertain_indices or
   * uncertain_tokens_bitset.
   * \returns The index of the next token, or -1 if no more token.
   */
  int GetNextUncertainToken(bool is_uncertain_saved, int* iterator_uncertain,
                            const std::vector<int>& uncertain_indices,
                            const std::vector<bool>& uncertain_tokens_bitset);

  /*! \brief Set the acceptable next token in next_token_bitmask. */
  void SetTokenBitmask(DLTensor* next_token_bitmask, std::vector<int32_t>& accepted_indices,
                       std::vector<int32_t>& rejected_indices, bool can_reach_end);

  /*! \brief Check if a token is a stop token. */
  bool IsStopToken(int32_t token_id) const {
    return std::find(init_ctx_->stop_token_ids.begin(), init_ctx_->stop_token_ids.end(),
                     token_id) != init_ctx_->stop_token_ids.end();
  }

  /*!
   * \brief Accept the stop token and terminates the matcher.
   * \returns Whether the stop token can be accepted.
   */
  bool AcceptStopToken();

  friend IntTuple FindNextRejectedTokens(GrammarStateMatcher matcher);

  std::shared_ptr<GrammarStateInitContext> init_ctx_;
  int max_rollback_steps_;
  std::deque<int> token_size_history_;

  // Temporary data for FindNextTokenBitmask. They are stored here to avoid repeated allocation.
  std::vector<int32_t> tmp_accepted_indices_;
  std::vector<int32_t> tmp_rejected_indices_;
  std::vector<int32_t> tmp_accepted_indices_delta_;
  std::vector<int32_t> tmp_rejected_indices_delta_;
  std::vector<bool> tmp_uncertain_tokens_bitset_;
};

bool GrammarStateMatcherNodeImpl::AcceptStopToken() {
  if (!CanReachEnd()) {
    return false;
  }
  stack_tops_history_.PushHistory({});  // Terminate the matcher by setting the stack to empty
  return true;
}

bool GrammarStateMatcherNodeImpl::AcceptToken(int32_t token_id) {
  CHECK(!IsTerminated())
      << "GrammarStateMatcher has terminated after accepting the stop token, but is trying to "
         "accept another token id "
      << token_id;

  // Handle the stop token
  if (IsStopToken(token_id)) {
    return AcceptStopToken();
  }

  CHECK(init_ctx_->id_to_token_codepoints.count(token_id) > 0)
      << "Token id " << token_id << " is not supported in generation";
  const auto& token = init_ctx_->id_to_token_codepoints[token_id].token;
  for (auto codepoint : token) {
    if (!AcceptCodepoint(codepoint, false)) {
      return false;
    }
  }
  token_size_history_.push_back(token.size());
  if (token_size_history_.size() > max_rollback_steps_) {
    DiscardEarliestCodepoints(token_size_history_.front());
    token_size_history_.pop_front();
  }
  return true;
}

void GrammarStateMatcherNodeImpl::FindNextTokenBitmask(DLTensor* next_token_bitmask) {
  CHECK(!IsTerminated())
      << "GrammarStateMatcher has terminated after accepting the stop token, but is trying to "
         "find the next token mask";
  const auto& sorted_token_codepoints = init_ctx_->sorted_token_codepoints;
  const auto& catagorized_tokens_for_grammar = init_ctx_->catagorized_tokens_for_grammar;
  const auto& latest_stack_tops = stack_tops_history_.GetLatest();

  // We check all the stacks one by one, and find the accepted token set or the rejected token set
  // for each stack. We will try to find the small one of the two sets.
  // The final accepted token set is the union of the accepted token sets of all stacks.
  // The final rejected token set is the intersection of the rejected token sets of all stacks.

  // Note these indices store the indices in sorted_token_codepoints, instead of the token ids.
  tmp_accepted_indices_.clear();
  // {-1} means the universal set, i.e. all tokens initially
  tmp_rejected_indices_.assign({-1});

  for (auto top : latest_stack_tops) {
    // Step 1. Find the current catagorized_tokens
    auto cur_rule_position = tree_[top];
    auto current_sequence = grammar_->GetRuleExpr(cur_rule_position.sequence_id);
    if (cur_rule_position.parent_id == RulePosition::kNoParent &&
        cur_rule_position.element_id == current_sequence.size()) {
      continue;
    }

    const auto& catagorized_tokens = catagorized_tokens_for_grammar.at(
        {cur_rule_position.sequence_id, cur_rule_position.element_id});

    // For each stack, we will check every uncertain token and put them into the accepted or
    // rejected list.
    // If the accepted tokens are saved, it means it is likely to be smaller than the rejected
    // tokens, so we will just find the accepted tokens, and vice versa.
    bool is_find_accept_mode =
        catagorized_tokens.not_saved_index != CatagorizedTokens::NotSavedIndex::kAccepted;

    // If uncertain tokens are saved, we will iterate over the uncertain tokens.
    // Otherwise, we will iterate over all_tokens - accepted_tokens - rejected_tokens.
    bool is_uncertain_saved =
        catagorized_tokens.not_saved_index != CatagorizedTokens::NotSavedIndex::kUncertain;

    // Step 2. Update the accepted tokens in accepted_indices_delta, or the rejected tokens in
    // rejected_indices_delta.

    // Examine only the current one stack
    stack_tops_history_.PushHistory({tree_.NewNode(cur_rule_position)});

    const std::vector<TCodepoint>* prev_token = nullptr;
    int prev_matched_size = 0;

    tmp_accepted_indices_delta_.clear();
    tmp_rejected_indices_delta_.clear();

    if (!is_uncertain_saved) {
      // unc_tokens = all_tokens - accepted_tokens - rejected_tokens
      tmp_uncertain_tokens_bitset_.assign(sorted_token_codepoints.size(), true);
      for (auto idx : catagorized_tokens.accepted_indices) {
        tmp_uncertain_tokens_bitset_[idx] = false;
      }
      for (auto idx : catagorized_tokens.rejected_indices) {
        tmp_uncertain_tokens_bitset_[idx] = false;
      }
    }

    int iterator_uncertain = -1;

    while (true) {
      // Step 2.1. Find the current token.
      auto idx =
          GetNextUncertainToken(is_uncertain_saved, &iterator_uncertain,
                                catagorized_tokens.uncertain_indices, tmp_uncertain_tokens_bitset_);
      if (idx == -1) {
        break;
      }
      const auto& cur_token = sorted_token_codepoints[idx].token;

      // Step 2.2. Find the longest common prefix with the accepted part of the previous token.
      // We can reuse the previous matched size to avoid unnecessary matching.
      int prev_useful_size = 0;
      if (prev_token) {
        prev_useful_size = std::min(prev_matched_size, static_cast<int>(cur_token.size()));
        for (int j = 0; j < prev_useful_size; ++j) {
          if (cur_token[j] != (*prev_token)[j]) {
            prev_useful_size = j;
            break;
          }
        }
        RollbackCodepoints(prev_matched_size - prev_useful_size);
      }

      // Step 2.3. Find if the current token is accepted or rejected.
      bool accepted = true;
      prev_matched_size = prev_useful_size;

      for (int j = prev_useful_size; j < cur_token.size(); ++j) {
        if (!AcceptCodepoint(cur_token[j], false)) {
          accepted = false;
          break;
        }
        prev_matched_size = j + 1;
      }

      // Step 2.4. Push the result to the delta list.
      if (accepted && is_find_accept_mode) {
        tmp_accepted_indices_delta_.push_back(idx);
      } else if (!accepted && !is_find_accept_mode) {
        tmp_rejected_indices_delta_.push_back(idx);
      }

      prev_token = &cur_token;
    }

    RollbackCodepoints(prev_matched_size + 1);

    // Step 3. Update the accepted_indices and rejected_indices
    if (is_find_accept_mode) {
      // accepted_indices += catagorized_tokens.accepted_indices + accepted_indices_delta
      IntsetUnion(&tmp_accepted_indices_delta_, catagorized_tokens.accepted_indices);
      IntsetUnion(&tmp_accepted_indices_, tmp_accepted_indices_delta_);
    } else {
      // rejected_indices = Intersect(
      //     rejected_indices,
      //     catagorized_tokens.rejected_indices + rejected_indices_delta)
      IntsetUnion(&tmp_rejected_indices_delta_, catagorized_tokens.rejected_indices);
      IntsetIntersection(&tmp_rejected_indices_, tmp_rejected_indices_delta_);
    }
  }

  // Finally update the rejected_ids bitset
  bool can_reach_end = CanReachEnd();
  SetTokenBitmask(next_token_bitmask, tmp_accepted_indices_, tmp_rejected_indices_, can_reach_end);
}

void GrammarStateMatcherNodeImpl::Rollback(int num_tokens) {
  CHECK(num_tokens <= token_size_history_.size())
      << "Intended to rollback " << num_tokens << " tokens, but only the last "
      << token_size_history_.size() << " steps of history are saved";
  while (num_tokens > 0) {
    int steps = token_size_history_.back();
    RollbackCodepoints(steps);
    token_size_history_.pop_back();
    --num_tokens;
  }
}

void GrammarStateMatcherNodeImpl::SetTokenBitmask(DLTensor* next_token_bitmask,
                                                  std::vector<int32_t>& accepted_indices,
                                                  std::vector<int32_t>& rejected_indices,
                                                  bool can_reach_end) {
  // accepted_ids = Union(accepted_indices, all_tokens - rejected_indices)
  // rejected_ids = Intersect(all_tokens - accepted_indices, rejected_indices)
  CHECK(next_token_bitmask->dtype.code == kDLUInt && next_token_bitmask->dtype.bits == 32 &&
        next_token_bitmask->data && next_token_bitmask->ndim == 1 && next_token_bitmask->shape)
      << "The provied bitmask's shape or dtype is not valid.";

  BitsetManager next_token_bitset(reinterpret_cast<uint32_t*>(next_token_bitmask->data),
                                  next_token_bitmask->shape[0]);

  if (rejected_indices.size() == 1 && rejected_indices[0] == -1) {
    // If rejected_indices is the universal set, the final accepted token set is just
    // accepted_indices
    next_token_bitset.Reset(init_ctx_->vocab_size, false);
    for (int idx : accepted_indices) {
      next_token_bitset.Set(init_ctx_->sorted_token_codepoints[idx].id, true);
    }

    if (can_reach_end) {
      // add end tokens
      for (int idx : init_ctx_->stop_token_ids) {
        next_token_bitset.Set(idx, true);
      }
    }
  } else {
    // Otherwise, the final rejected token set is (rejected_indices \ accepted_indices)
    next_token_bitset.Reset(init_ctx_->vocab_size, true);

    auto it_acc = accepted_indices.begin();
    for (auto i : rejected_indices) {
      while (it_acc != accepted_indices.end() && *it_acc < i) {
        ++it_acc;
      }
      if (it_acc == accepted_indices.end() || *it_acc != i) {
        next_token_bitset.Set(init_ctx_->sorted_token_codepoints[i].id, false);
      }
    }

    for (int idx : init_ctx_->special_token_ids) {
      next_token_bitset.Set(idx, false);
    }
    if (!can_reach_end) {
      for (int idx : init_ctx_->stop_token_ids) {
        next_token_bitset.Set(idx, false);
      }
    }
  }
}

int GrammarStateMatcherNodeImpl::GetNextUncertainToken(
    bool is_uncertain_saved, int* iterator_uncertain, const std::vector<int>& uncertain_indices,
    const std::vector<bool>& uncertain_tokens_bitset) {
  if (is_uncertain_saved) {
    ++*iterator_uncertain;
    if (*iterator_uncertain == uncertain_indices.size()) {
      return -1;
    }
    return uncertain_indices[*iterator_uncertain];
  } else {
    ++*iterator_uncertain;
    while (*iterator_uncertain < uncertain_tokens_bitset.size() &&
           !uncertain_tokens_bitset[*iterator_uncertain]) {
      ++*iterator_uncertain;
    }
    if (*iterator_uncertain == uncertain_tokens_bitset.size()) {
      return -1;
    }
    return *iterator_uncertain;
  }
}

GrammarStateMatcher::GrammarStateMatcher(std::shared_ptr<GrammarStateInitContext> init_ctx,
                                         int max_rollback_steps)
    : ObjectRef(make_object<GrammarStateMatcherNodeImpl>(init_ctx, max_rollback_steps)) {}

TVM_REGISTER_GLOBAL("mlc.serve.GrammarStateMatcherFromTokenizer")
    .set_body_typed([](BNFGrammar grammar, Optional<Tokenizer> tokenizer, int max_rollback_steps) {
      auto preproc_start = std::chrono::high_resolution_clock::now();
      auto init_ctx = GrammarStateMatcher::CreateInitContext(
          grammar, tokenizer ? tokenizer.value()->TokenTable() : std::vector<std::string>());
      auto preproc_end = std::chrono::high_resolution_clock::now();
      std::cerr << "Preprocess takes "
                << std::chrono::duration_cast<std::chrono::microseconds>(preproc_end -
                                                                         preproc_start)
                       .count()
                << "us";
      return GrammarStateMatcher(init_ctx, max_rollback_steps);
    });

TVM_REGISTER_GLOBAL("mlc.serve.GrammarStateMatcherFromTokenTable")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      BNFGrammar grammar = args[0];
      std::vector<std::string> token_table;
      for (int i = 1; i < args.size() - 1; ++i) {
        token_table.push_back(args[i]);
      }
      int max_rollback_steps = args[args.size() - 1];
      auto init_ctx = GrammarStateMatcher::CreateInitContext(grammar, token_table);
      *rv = GrammarStateMatcher(init_ctx, max_rollback_steps);
    });

TVM_REGISTER_GLOBAL("mlc.serve.GrammarStateMatcherDebugAcceptCodepoint")
    .set_body_typed([](GrammarStateMatcher matcher, int32_t codepoint) {
      auto mutable_node =
          const_cast<GrammarStateMatcherNodeImpl*>(matcher.as<GrammarStateMatcherNodeImpl>());
      return mutable_node->AcceptCodepoint(codepoint);
    });

TVM_REGISTER_GLOBAL("mlc.serve.GrammarStateMatcherAcceptToken")
    .set_body_typed([](GrammarStateMatcher matcher, int32_t token_id) {
      return matcher->AcceptToken(token_id);
    });

TVM_REGISTER_GLOBAL("mlc.serve.GrammarStateMatcherRollback")
    .set_body_typed([](GrammarStateMatcher matcher, int num_tokens) {
      matcher->Rollback(num_tokens);
    });

TVM_REGISTER_GLOBAL("mlc.serve.GrammarStateMatcherMaxRollbackSteps")
    .set_body_typed([](GrammarStateMatcher matcher) { return matcher->MaxRollbackSteps(); });

TVM_REGISTER_GLOBAL("mlc.serve.GrammarStateMatcherIsTerminated")
    .set_body_typed([](GrammarStateMatcher matcher) { return matcher->IsTerminated(); });

TVM_REGISTER_GLOBAL("mlc.serve.GrammarStateMatcherResetState")
    .set_body_typed([](GrammarStateMatcher matcher) { matcher->ResetState(); });

/*! \brief Check if a matcher can accept the complete string, and then reach the end of the
 * grammar. For test purpose. */
bool MatchCompleteString(GrammarStateMatcher matcher, String str) {
  auto mutable_node =
      const_cast<GrammarStateMatcherNodeImpl*>(matcher.as<GrammarStateMatcherNodeImpl>());
  auto codepoints = Utf8StringToCodepoints(str.c_str());
  int accepted_cnt = 0;
  for (auto codepoint : codepoints) {
    if (!mutable_node->AcceptCodepoint(codepoint, false)) {
      mutable_node->RollbackCodepoints(accepted_cnt);
      return false;
    }
    ++accepted_cnt;
  }
  return mutable_node->CanReachEnd();
}

TVM_REGISTER_GLOBAL("mlc.serve.GrammarStateMatcherDebugMatchCompleteString")
    .set_body_typed([](GrammarStateMatcher matcher, String str) {
      return MatchCompleteString(matcher, str);
    });

/*!
 * \brief Find the ids of the rejected tokens for the next step. For test purposes.
 * \returns A tuple of rejected token ids.
 */
IntTuple FindNextRejectedTokens(GrammarStateMatcher matcher) {
  auto init_ctx = matcher.as<GrammarStateMatcherNodeImpl>()->init_ctx_;
  auto vocab_size = init_ctx->vocab_size;
  auto bitset_size = BitsetManager::GetBitsetSize(vocab_size);
  auto ndarray = NDArray::Empty(ShapeTuple{static_cast<long>(bitset_size)},
                                DLDataType{kDLUInt, 32, 1}, DLDevice{kDLCPU, 0});
  auto dltensor = const_cast<DLTensor*>(ndarray.operator->());

  auto start = std::chrono::high_resolution_clock::now();
  matcher->FindNextTokenBitmask(dltensor);
  auto end = std::chrono::high_resolution_clock::now();
  std::cerr << "FindNextTokenBitmask takes "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us";

  auto bitset = BitsetManager(reinterpret_cast<uint32_t*>(dltensor->data), bitset_size);
  std::vector<int64_t> rejected_ids;
  for (int i = 0; i < vocab_size; i++) {
    if (bitset[i] == 0) {
      rejected_ids.push_back(i);
    }
  }

  std::cerr << ", found accepted: " << vocab_size - rejected_ids.size()
            << ", rejected: " << rejected_ids.size() << std::endl;

  auto ret = IntTuple(rejected_ids);
  return ret;
}

TVM_REGISTER_GLOBAL("mlc.serve.GrammarStateMatcherFindNextRejectedTokens")
    .set_body_typed(FindNextRejectedTokens);

}  // namespace serve
}  // namespace llm
}  // namespace mlc
