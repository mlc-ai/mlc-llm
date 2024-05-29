/*!
 *  Copyright (c) 2023 by Contributors
 * \file grammar/grammar_state_matcher.cc
 */
// #define TVM_LOG_DEBUG 1
#include "grammar_state_matcher.h"

#include <chrono>
#include <queue>

#include "../tokenizers/tokenizers.h"
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
 * stacks with different top elements will be added. When one stack cannot accept the new character,
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
 * ### The previous step
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
 * ### The current step
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
 * when the next char comes)
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
  using SaveType = CatagorizedTokens::SaveType;

 public:
  GrammarStateMatcherNodeImpl(std::shared_ptr<GrammarStateInitContext> init_ctx,
                              int max_rollback_steps = 0)
      : GrammarStateMatcherBase(init_ctx->grammar),
        init_ctx_(init_ctx),
        max_rollback_steps_(max_rollback_steps),
        tmp_accepted_bitset_(init_ctx_->vocab_size) {}

  bool AcceptToken(int32_t token_id) final;

  void FindNextTokenBitmask(DLTensor* next_token_bitmask) final;

  void Rollback(int num_tokens) final;

  int MaxRollbackSteps() const final { return max_rollback_steps_; }

  bool IsTerminated() const { return stack_tops_history_.GetLatest().empty(); }

  void ResetState() final {
    stack_tops_history_.Reset();
    token_length_history.clear();
    PushInitialState(kInvalidRulePosition, true);
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
  void SetTokenBitmask(DLTensor* next_token_bitmask, const DynamicBitset& accepted_bitset,
                       const std::vector<int32_t>& rejected_indices, bool can_reach_end);

  /*!
   * \brief Accept the stop token and terminates the matcher.
   * \returns Whether the stop token can be accepted.
   */
  bool AcceptStopToken();

  friend IntTuple FindNextRejectedTokens(GrammarStateMatcher matcher, bool verbose);
  friend NDArray FindNextTokenBitmaskAsNDArray(GrammarStateMatcher matcher);

  std::shared_ptr<GrammarStateInitContext> init_ctx_;
  int max_rollback_steps_;
  std::deque<int> token_length_history;

  // Temporary data for FindNextTokenBitmask. They are stored here to avoid repeated allocation.
  DynamicBitset tmp_accepted_bitset_;
  std::vector<int32_t> tmp_rejected_indices_;
  std::vector<int32_t> tmp_rejected_indices_delta_;
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

  CHECK(token_id >= 0 && token_id < init_ctx_->vocab_size)
      << "Invalid token id " << token_id << " for GrammarStateMatcher";

  // Handle the stop token
  if (std::find(init_ctx_->stop_token_ids.begin(), init_ctx_->stop_token_ids.end(), token_id) !=
      init_ctx_->stop_token_ids.end()) {
    return AcceptStopToken();
  }

  if (init_ctx_->special_token_ids.count(token_id) > 0) {
    LOG(FATAL)
        << "Token id " << token_id << ": " << init_ctx_->token_table[token_id]
        << " is regarded as a special token, and cannot be accepted by the GrammarStateMatcher";
  }

  const auto& token = init_ctx_->token_table[token_id];
  for (auto char_value : token) {
    if (!AcceptChar(char_value, false)) {
      return false;
    }
  }
  token_length_history.push_back(token.size());
  if (token_length_history.size() > max_rollback_steps_) {
    DiscardEarliestChars(token_length_history.front());
    token_length_history.pop_front();
  }
  return true;
}

void GrammarStateMatcherNodeImpl::FindNextTokenBitmask(DLTensor* next_token_bitmask) {
  CHECK(!IsTerminated())
      << "GrammarStateMatcher has terminated after accepting the stop token, but is trying to "
         "find the next token mask";
  const auto& sorted_token_table = init_ctx_->sorted_token_table;
  const auto& catagorized_tokens_for_grammar = init_ctx_->catagorized_tokens_for_grammar;
  const auto& latest_stack_tops = stack_tops_history_.GetLatest();

  // We check all the stacks one by one, and find the accepted token set or the rejected token set
  // for each stack. We will try to find the small one of the two sets.
  // The final accepted token set is the union of the accepted token sets of all stacks.
  // The final rejected token set is the intersection of the rejected token sets of all stacks.

  // Note these indices store the indices in sorted_token_table, instead of the token ids.
  tmp_accepted_bitset_.Reset();
  // {-1} means the universal set, i.e. all tokens initially
  tmp_rejected_indices_.assign({-1});

  // std::chrono::microseconds time_unc(0);
  // std::chrono::microseconds time_idx(0);
  int check_cnt = 0;

  for (auto top : latest_stack_tops) {
    auto cur_rule_position = tree_[top];
    if (tree_.IsEndPosition(cur_rule_position)) {
      continue;
    }

    const auto& catagorized_tokens = catagorized_tokens_for_grammar.at(cur_rule_position);

    // auto start = std::chrono::high_resolution_clock::now();

    // For each stack, we will check every uncertain token and put them into the accepted or
    // rejected list.

    // Step 2. Update the accepted tokens in accepted_indices_delta, or the rejected tokens in
    // rejected_indices_delta.

    // If the accepted tokens are saved, it means it is likely to be smaller than the rejected
    // tokens, so we will just find the accepted tokens, and vice versa.

    tmp_rejected_indices_delta_.clear();

    // Examine only the current one stack
    stack_tops_history_.PushHistory({tree_.NewNode(cur_rule_position)});

    const std::string* prev_token = nullptr;
    int prev_matched_size = 0;

    // std::cout << tree_.PrintNode(top) << std::endl;

    // std::cout << "Accepted count: " << catagorized_tokens.accepted_indices.size()
    //           << ", rejected count: " << catagorized_tokens.rejected_indices.size()
    //           << ", uncertain count: " << catagorized_tokens.uncertain_indices.size()
    //           << ", save type: " << static_cast<int>(catagorized_tokens.save_type) << std::endl;

    // if (catagorized_tokens.accepted_indices.size() < 200) {
    //   std::cout << "Accpeted: ";
    //   for (int i = 0; i < catagorized_tokens.accepted_indices.size(); ++i) {
    //     std::cout << "<"
    //               << PrintAsEscaped(
    //                      sorted_token_table[catagorized_tokens.accepted_indices[i]].second)
    //               << "> ";
    //   }
    //   std::cout << "\n";
    // }

    // if (catagorized_tokens.uncertain_indices.size() > 100) {
    // std::cout << "Uncertain: ";
    // for (int i = 0; i < catagorized_tokens.uncertain_indices.size(); ++i) {
    //   std::cout << "<"
    //             << PrintAsEscaped(
    //                    sorted_token_table[catagorized_tokens.uncertain_indices[i]].second)
    //             << "> ";
    // }
    // std::cout << "\n";
    // }

    for (auto cur_token_idx : catagorized_tokens.uncertain_indices) {
      const auto& cur_token = sorted_token_table[cur_token_idx].second;
      bool accepted = true;

      // Step 2.1. Find the longest common prefix with the accepted part of the previous token.
      // We can reuse the previous matched size to avoid unnecessary matching.
      if (prev_token) {
        int lcp_len = std::mismatch(cur_token.begin(), cur_token.end(), prev_token->begin(),
                                    prev_token->end())
                          .first -
                      cur_token.begin();
        if (lcp_len > prev_matched_size) {
          accepted = false;
        } else if (lcp_len < prev_matched_size) {
          RollbackChars(prev_matched_size - lcp_len);
        }
        prev_matched_size = std::min(prev_matched_size, lcp_len);
      }

      // Step 2.2. Find if the current token is accepted or rejected.
      if (accepted) {
        for (int j = prev_matched_size; j < cur_token.size(); ++j) {
          ++check_cnt;
          if (!AcceptChar(cur_token[j], false)) {
            accepted = false;
            break;
          }
          prev_matched_size = j + 1;
        }
      }

      // Step 2.3. Push the result to the delta list.
      if (catagorized_tokens.save_type == SaveType::kAcceptedBitset ||
          catagorized_tokens.save_type == SaveType::kAccepted) {
        if (accepted) {
          tmp_accepted_bitset_.Set(sorted_token_table[cur_token_idx].first, true);
        }
      } else {
        if (!accepted) {
          tmp_rejected_indices_delta_.push_back(cur_token_idx);
        }
      }

      prev_token = &cur_token;
    }

    RollbackChars(prev_matched_size + 1);

    // auto end = std::chrono::high_resolution_clock::now();

    // time_unc += std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // start = std::chrono::high_resolution_clock::now();

    // Step 3. Update the accepted_indices and rejected_indices
    if (catagorized_tokens.save_type == SaveType::kAcceptedBitset) {
      tmp_accepted_bitset_ |= catagorized_tokens.accepted_bitset;
    } else if (catagorized_tokens.save_type == SaveType::kAccepted) {
      for (auto idx : catagorized_tokens.accepted_indices) {
        tmp_accepted_bitset_.Set(sorted_token_table[idx].first, true);
      }
    } else {
      // rejected_indices = Intersect(
      //     rejected_indices,
      //     catagorized_tokens.rejected_indices + rejected_indices_delta)
      IntsetUnion(&tmp_rejected_indices_delta_, catagorized_tokens.rejected_indices);
      IntsetIntersection(&tmp_rejected_indices_, tmp_rejected_indices_delta_);
    }
    // end = std::chrono::high_resolution_clock::now();
    // time_idx += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  }

  // Finally update the rejected_ids bitset
  // auto start = std::chrono::high_resolution_clock::now();
  bool can_reach_end = CanReachEnd();
  SetTokenBitmask(next_token_bitmask, tmp_accepted_bitset_, tmp_rejected_indices_, can_reach_end);
  // auto end = std::chrono::high_resolution_clock::now();
  // time_idx += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  // std::cout << "Time for uncertain: " << time_unc.count()
  //           << "us, time for index: " << time_idx.count() << "us" << std::endl;
  // std::cout << "Check cnt " << check_cnt << std::endl;
}

void GrammarStateMatcherNodeImpl::Rollback(int num_tokens) {
  CHECK(num_tokens <= token_length_history.size())
      << "Intended to rollback " << num_tokens << " tokens, but only the last "
      << token_length_history.size() << " steps of history are saved";
  while (num_tokens > 0) {
    int steps = token_length_history.back();
    RollbackChars(steps);
    token_length_history.pop_back();
    --num_tokens;
  }
}

void GrammarStateMatcherNodeImpl::SetTokenBitmask(DLTensor* next_token_bitmask,
                                                  const DynamicBitset& accepted_bitset,
                                                  const std::vector<int32_t>& rejected_indices,
                                                  bool can_reach_end) {
  // next_token_bitmask = set(all accepted tokens) =
  // 1. all_tokens - (rejected_ids / accepted_ids)
  //    (when rejected_ids != {-1}, i.e. rejected_ids is not the universal set)
  // 2. accepted_ids
  //    (otherwise, when rejected_ids is the universal set)
  CHECK(next_token_bitmask->dtype.code == kDLUInt && next_token_bitmask->dtype.bits == 32 &&
        next_token_bitmask->data && next_token_bitmask->ndim == 1 && next_token_bitmask->shape)
      << "The provied bitmask's shape or dtype is not valid.";
  CHECK(next_token_bitmask->shape[0] >= DynamicBitset::CalculateBufferSize(init_ctx_->vocab_size))
      << "The provided bitmask is not large enough to store the token set. The length should be "
      << DynamicBitset::CalculateBufferSize(init_ctx_->vocab_size) << " at least";

  DynamicBitset next_token_bitset(init_ctx_->vocab_size,
                                  reinterpret_cast<uint32_t*>(next_token_bitmask->data));
  const auto& sorted_token_table = init_ctx_->sorted_token_table;

  if (rejected_indices.size() == 1 && rejected_indices[0] == -1) {
    // If rejected_indices is the universal set, the final accepted token set is just
    // accepted_indices
    next_token_bitset = accepted_bitset;

    if (can_reach_end) {
      // add end tokens
      for (int id : init_ctx_->stop_token_ids) {
        next_token_bitset.Set(id, true);
      }
    }
  } else {
    // Otherwise, the final rejected token set is (rejected_indices \ accepted_indices)
    next_token_bitset.Set();

    for (auto i : rejected_indices) {
      auto id = sorted_token_table[i].first;
      if (!accepted_bitset[id]) {
        next_token_bitset.Set(id, false);
      }
    }

    for (int id : init_ctx_->special_token_ids) {
      next_token_bitset.Set(id, false);
    }
    if (!can_reach_end) {
      for (int id : init_ctx_->stop_token_ids) {
        next_token_bitset.Set(id, false);
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

#ifndef COMPILE_MLC_WASM_RUNTIME
// This creates tokenizer dependency issue in WASM building for web, hence skipped
TVM_REGISTER_GLOBAL("mlc.grammar.GrammarStateMatcherFromTokenizer")
    .set_body_typed([](BNFGrammar grammar, Optional<Tokenizer> tokenizer, int max_rollback_steps) {
      auto preproc_start = std::chrono::high_resolution_clock::now();
      std::shared_ptr<mlc::llm::serve::GrammarStateInitContext> init_ctx;
      if (tokenizer) {
        init_ctx = GrammarStateMatcher::CreateInitContext(
            grammar, tokenizer.value()->PostProcessedTokenTable());
      } else {
        init_ctx = GrammarStateMatcher::CreateInitContext(grammar, {});
      }

      auto preproc_end = std::chrono::high_resolution_clock::now();
      LOG(INFO) << "GrammarStateMatcher preprocess takes "
                << std::chrono::duration_cast<std::chrono::microseconds>(preproc_end -
                                                                         preproc_start)
                       .count()
                << "us";
      return GrammarStateMatcher(init_ctx, max_rollback_steps);
    });
#endif

TVM_REGISTER_GLOBAL("mlc.grammar.GrammarStateMatcherFromTokenTable")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      BNFGrammar grammar = args[0];
      Array<String> token_table_arr = args[1];
      std::vector<std::string> token_table;
      for (int i = 0; i < token_table_arr.size(); ++i) {
        token_table.push_back(token_table_arr[i]);
      }
      int max_rollback_steps = args[args.size() - 1];
      auto init_ctx = GrammarStateMatcher::CreateInitContext(grammar, token_table);
      *rv = GrammarStateMatcher(init_ctx, max_rollback_steps);
    });

TVM_REGISTER_GLOBAL("mlc.grammar.GrammarStateMatcherDebugAcceptChar")
    .set_body_typed([](GrammarStateMatcher matcher, int32_t codepoint, bool verbose) {
      auto mutable_node =
          const_cast<GrammarStateMatcherNodeImpl*>(matcher.as<GrammarStateMatcherNodeImpl>());
      return mutable_node->AcceptChar(codepoint, verbose);
    });

TVM_REGISTER_GLOBAL("mlc.grammar.GrammarStateMatcherAcceptToken")
    .set_body_typed([](GrammarStateMatcher matcher, int32_t token_id) {
      return matcher->AcceptToken(token_id);
    });

TVM_REGISTER_GLOBAL("mlc.grammar.GrammarStateMatcherRollback")
    .set_body_typed([](GrammarStateMatcher matcher, int num_tokens) {
      matcher->Rollback(num_tokens);
    });

TVM_REGISTER_GLOBAL("mlc.grammar.GrammarStateMatcherMaxRollbackSteps")
    .set_body_typed([](GrammarStateMatcher matcher) { return matcher->MaxRollbackSteps(); });

TVM_REGISTER_GLOBAL("mlc.grammar.GrammarStateMatcherIsTerminated")
    .set_body_typed([](GrammarStateMatcher matcher) { return matcher->IsTerminated(); });

TVM_REGISTER_GLOBAL("mlc.grammar.GrammarStateMatcherResetState")
    .set_body_typed([](GrammarStateMatcher matcher) { matcher->ResetState(); });

/*! \brief Check if a matcher can accept the complete string, and then reach the end of the
 * grammar. Does not change the state of the GrammarStateMatcher. For test purpose. */
bool MatchCompleteString(GrammarStateMatcher matcher, String str, bool verbose) {
  auto mutable_node =
      const_cast<GrammarStateMatcherNodeImpl*>(matcher.as<GrammarStateMatcherNodeImpl>());
  int accepted_cnt = 0;
  for (auto char_value : str.operator std::string()) {
    if (!mutable_node->AcceptChar(char_value, verbose)) {
      if (verbose) {
        LOG(INFO) << "Matching failed after accepting " << accepted_cnt << " characters";
      }
      mutable_node->RollbackChars(accepted_cnt);
      return false;
    }
    ++accepted_cnt;
  }
  auto accepted = mutable_node->CanReachEnd();
  if (verbose) {
    if (accepted) {
      LOG(INFO) << "Matching succeed after accepting " << accepted_cnt << " characters";
    } else {
      LOG(INFO) << "Matching failed due to the end state not reached after all " << accepted_cnt
                << " characters are accepted";
    }
  }
  mutable_node->RollbackChars(accepted_cnt);
  return accepted;
}

TVM_REGISTER_GLOBAL("mlc.grammar.GrammarStateMatcherDebugMatchCompleteString")
    .set_body_typed([](GrammarStateMatcher matcher, String str, bool verbose) {
      return MatchCompleteString(matcher, str, verbose);
    });

/*! \brief Print the accepted and rejected tokens stored in the bitset. For debug purposes. */
std::string PrintAcceptedRejectedTokens(
    const std::shared_ptr<mlc::llm::serve::GrammarStateInitContext>& init_ctx,
    const DynamicBitset& bitset, int threshold = 300) {
  std::stringstream ss;
  auto vocab_size = init_ctx->vocab_size;
  std::vector<int64_t> accepted_ids;
  std::vector<int64_t> rejected_ids;
  for (int i = 0; i < vocab_size; i++) {
    if (bitset[i]) {
      accepted_ids.push_back(i);
    } else {
      rejected_ids.push_back(i);
    }
  }

  ss << "Accepted: ";
  auto end_it =
      accepted_ids.size() > threshold ? accepted_ids.begin() + threshold : accepted_ids.end();
  for (auto it = accepted_ids.begin(); it != end_it; ++it) {
    ss << "<" << PrintAsEscaped(init_ctx->token_table[*it]) << "> ";
  }
  if (accepted_ids.size() > threshold) {
    ss << "...";
  }
  ss << "\n";

  ss << "Rejected: ";
  end_it = rejected_ids.size() > threshold ? rejected_ids.begin() + threshold : rejected_ids.end();
  for (auto it = rejected_ids.begin(); it != end_it; ++it) {
    ss << "<" << PrintAsEscaped(init_ctx->token_table[*it]) << "> ";
  }
  if (rejected_ids.size() > threshold) {
    ss << "...";
  }
  ss << "\n";
  return ss.str();
}

/*!
 * \brief Find the ids of the rejected tokens for the next step. For debug purposes.
 * \param matcher The matcher to test.
 * \param verbose Whether to print information about the timing and results to stderr.
 * \returns A tuple of rejected token ids.
 */
IntTuple FindNextRejectedTokens(GrammarStateMatcher matcher, bool verbose = false) {
  auto init_ctx = matcher.as<GrammarStateMatcherNodeImpl>()->init_ctx_;
  auto vocab_size = init_ctx->vocab_size;
  auto bitset_size = DynamicBitset::CalculateBufferSize(vocab_size);
  auto ndarray = NDArray::Empty(ShapeTuple{static_cast<long>(bitset_size)},
                                DLDataType{kDLUInt, 32, 1}, DLDevice{kDLCPU, 0});
  auto dltensor = const_cast<DLTensor*>(ndarray.operator->());

  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  if (verbose) {
    start = std::chrono::high_resolution_clock::now();
  }
  matcher->FindNextTokenBitmask(dltensor);
  if (verbose) {
    end = std::chrono::high_resolution_clock::now();
  }

  auto bitset = DynamicBitset(vocab_size, reinterpret_cast<uint32_t*>(dltensor->data));
  std::vector<int64_t> rejected_ids;
  for (int i = 0; i < vocab_size; i++) {
    if (bitset[i] == 0) {
      rejected_ids.push_back(i);
    }
  }

  if (verbose) {
    LOG(INFO) << "FindNextTokenBitmask takes "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us"
              << ", found accepted: " << vocab_size - rejected_ids.size()
              << ", rejected: " << rejected_ids.size();
  }

  auto ret = IntTuple(rejected_ids);
  return ret;
}

TVM_REGISTER_GLOBAL("mlc.grammar.GrammarStateMatcherFindNextRejectedTokens")
    .set_body_typed(FindNextRejectedTokens);

/*!
 * \brief Find the bitmask for the next token as an NDArray.
 * \returns An NDArray of the bitmask for the next token of shape (bitmask_size,).
 */
NDArray FindNextTokenBitmaskAsNDArray(GrammarStateMatcher matcher) {
  auto init_ctx = matcher.as<GrammarStateMatcherNodeImpl>()->init_ctx_;
  auto vocab_size = init_ctx->vocab_size;
  auto bitset_size = DynamicBitset::CalculateBufferSize(vocab_size);
  auto bitmask = NDArray::Empty(ShapeTuple{static_cast<long>(bitset_size)},
                                DLDataType{kDLUInt, 32, 1}, DLDevice{kDLCPU, 0});
  auto dltensor = const_cast<DLTensor*>(bitmask.operator->());
  matcher->FindNextTokenBitmask(dltensor);
  return bitmask;
}

TVM_REGISTER_GLOBAL("mlc.grammar.GrammarStateMatcherFindNextTokenBitmaskAsNDArray")
    .set_body_typed(FindNextTokenBitmaskAsNDArray);

}  // namespace serve
}  // namespace llm
}  // namespace mlc
