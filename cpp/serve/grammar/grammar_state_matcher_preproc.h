/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar_state_matcher_preproc.h
 * \brief The header for the preprocessing of the grammar state matcher.
 */
#ifndef MLC_LLM_SERVE_GRAMMAR_GRAMMAR_STATE_MATCHER_PREPROC_H_
#define MLC_LLM_SERVE_GRAMMAR_GRAMMAR_STATE_MATCHER_PREPROC_H_

#include <vector>

#include "../../support/encoding.h"
#include "../../support/utils.h"
#include "grammar.h"
#include "grammar_state_matcher_base.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/*!
 * \brief Preprocessed information, for a given specific RulePosition, divides the token set
 * into three categories: accepted, rejected, and uncertain.
 * Accepted: tokens that can be determined by the current RulePosition to be acceptable
 * Rejected: tokens that can be determined by the current RulePosition to be unacceptable
 * Uncertain: tokens that need the state of the parent RulePositions to determine if acceptable
 *
 * \note uncertain indices are stored directly. Accepted / rejected indices have three ways to
 * store to reduce memory and computation usage. See SaveType.
 * \note These indices are the indices of sorted_token_table in the GrammarStateInitContext
 * object, instead of the token ids. That helps the matching process.
 */
struct CatagorizedTokens {
  enum class SaveType {
    // Only store all accepted token indices. Then rejected indices = all_indices - accepted_indices
    // - uncertain_indices. This is useful when |accepted_indices| < |rejected_indices|.
    kAccepted = 0,
    // Only store all accepted token indices. Then accepted indices = all_indices - rejected_indices
    // - uncertain_indices. This is useful when |accepted_indices| > |rejected_indices|.
    kRejected = 1,
    // Store all accepted token indices in a bitset. This is useful when both |accepted_indices| and
    // |rejected_indices| are large.
    kAcceptedBitset = 2
  };
  SaveType save_type;

  static constexpr int USE_BITSET_THRESHOLD = 200;

  std::vector<int32_t> accepted_indices;
  std::vector<int32_t> rejected_indices;
  DynamicBitset accepted_bitset;

  std::vector<int32_t> uncertain_indices;

  CatagorizedTokens() = default;

  CatagorizedTokens(int vocab_size,
                    const std::vector<std::pair<int32_t, std::string>>& sorted_token_table,
                    const std::vector<int32_t>& accepted_indices,
                    const std::vector<int32_t>& rejected_indices,
                    const std::vector<int32_t>& uncertain_indices);
};

/*!
 * \brief All information that we need to match tokens in the tokenizer to the specified grammar.
 * It is the result of preprocessing.
 * \sa mlc::llm::serve::GrammarStateMatcher
 */
class GrammarStateInitContext {
 public:
  /******************* Information about the tokenizer *******************/

  /*! \brief The vocabulary size of the tokenizer. Special tokens are included. */
  size_t vocab_size;
  /*! \brief The token table. Special tokens are included. */
  std::vector<std::string> token_table;
  /*! \brief All (id, token) pairs sorted in lexicographic order. This sorting is done to
   * maximize prefix reuse during matching. Special tokens and stop tokens are not included. */
  std::vector<std::pair<int32_t, std::string>> sorted_token_table;
  /*! \brief The stop tokens. When the GrammarStateMatcher can reach the end of the= grammar,
   * stop tokens can be accepted. */
  std::vector<int32_t> stop_token_ids;
  /*! \brief The special tokens. These tokens are ignored (masked out) during the grammar-guided
   * generation. */
  std::unordered_set<int32_t> special_token_ids;

  /******************* Information about the grammar *******************/

  /*! \brief The grammar for the GrammarStateMatcher. */
  BNFGrammar grammar;

  /******************* Grammar-specific tokenizer information *******************/

  struct RulePositionEqual {
    std::size_t operator()(const RulePosition& lhs, const RulePosition& rhs) const noexcept {
      return lhs.sequence_id == rhs.sequence_id && lhs.element_id == rhs.element_id &&
             lhs.left_utf8_bytes == rhs.left_utf8_bytes &&
             lhs.element_in_string == rhs.element_in_string;
    }
  };

  struct RulePositionHash {
    std::size_t operator()(const RulePosition& rule_position) const noexcept {
      return HashCombine(rule_position.sequence_id, rule_position.element_id,
                         rule_position.left_utf8_bytes, rule_position.element_in_string);
    }
  };

  /*! \brief Mapping from RulePositions to the catagorized tokens. */
  std::unordered_map<RulePosition, CatagorizedTokens, RulePositionHash, RulePositionEqual>
      catagorized_tokens_for_grammar;
};

/*! \brief The concrete implementation of GrammarStateMatcherNode. */
class GrammarStateMatcherForInitContext : public GrammarStateMatcherBase {
 public:
  // Do not expand the initial rule position: we want to find the accepted/rejected tokens
  // that exactly start from the initial rule position.
  GrammarStateMatcherForInitContext(const BNFGrammar& grammar, RulePosition init_rule_position)
      : GrammarStateMatcherBase(grammar, init_rule_position, false),
        init_rule_id(init_rule_position.rule_id) {}

  /*!
   * \brief Get the catagorized tokens for the given RulePosition.
   * \param consider_parent_rule Whether to consider the parent rule. If false, there will be
   * no uncertain tokens. Useful for the main rule.
   */
  CatagorizedTokens GetCatagorizedTokens(
      int vocab_size, const std::vector<std::pair<int32_t, std::string>>& sorted_token_table,
      bool consider_parent_rule);

 private:
  using RuleExpr = BNFGrammarNode::RuleExpr;
  using RuleExprType = BNFGrammarNode::RuleExprType;

  /*! \brief Check if a token can pass the lookahead assertion. */
  bool IsTokenPassLookaheadAssertion(const std::string& token,
                                     const std::vector<bool>& can_reach_end_stack);

  // The id of the initial rule.
  int32_t init_rule_id;

  // Temporary data for GetCatagorizedTokens.
  std::vector<int32_t> tmp_accepted_indices_;
  std::vector<int32_t> tmp_rejected_indices_;
  std::vector<int32_t> tmp_uncertain_indices_;
  std::vector<bool> tmp_can_reach_end_stack_;
  std::vector<bool> tmp_can_reach_end_prefix_or_stack_;
};

inline CatagorizedTokens::CatagorizedTokens(
    int vocab_size, const std::vector<std::pair<int32_t, std::string>>& sorted_token_table,
    const std::vector<int32_t>& accepted_indices, const std::vector<int32_t>& rejected_indices,
    const std::vector<int32_t>& uncertain_indices) {
  auto size_acc = accepted_indices.size();
  auto size_rej = rejected_indices.size();

  save_type = size_acc >= USE_BITSET_THRESHOLD && size_rej >= USE_BITSET_THRESHOLD
                  ? SaveType::kAcceptedBitset
              : size_acc < size_rej ? SaveType::kAccepted
                                    : SaveType::kRejected;

  if (save_type == SaveType::kAcceptedBitset) {
    accepted_bitset = DynamicBitset(vocab_size);
    for (auto idx : accepted_indices) {
      accepted_bitset.Set(sorted_token_table[idx].first, true);
    }
  } else if (save_type == SaveType::kAccepted) {
    this->accepted_indices = accepted_indices;
  } else {
    this->rejected_indices = rejected_indices;
  }

  this->uncertain_indices = uncertain_indices;
}

bool GrammarStateMatcherForInitContext::IsTokenPassLookaheadAssertion(
    const std::string& token, const std::vector<bool>& can_reach_end_stack) {
  auto lookahead_assertion_id = grammar_->GetRule(init_rule_id).lookahead_assertion_id;
  if (lookahead_assertion_id == -1) {
    return true;
  }
  auto lookahead_rule_position = RulePosition(-1, lookahead_assertion_id, 0);
  PushInitialState(lookahead_rule_position, true);
  int token_len = token.size();

  // Find all positions that can come to and end. Then check if the suffix from that position
  // can be accepted by the lookahead assertion.
  for (int i = static_cast<int>(can_reach_end_stack.size()); i >= 0; --i) {
    if (!can_reach_end_stack[i]) {
      continue;
    }
    int last_accept_pos = i - 1;
    for (int pos = i; pos < token_len; ++pos) {
      if (!AcceptChar(token[pos])) {
        break;
      }
      last_accept_pos = pos;
      // Case 1. The whole rule is finished.
      if (CanReachEnd()) {
        // accepted chars: pos - i + 1
        // we need to rollback the pushed initial state as well
        RollbackChars(pos - i + 2);
        return true;
      }
    }
    // Case 2. The whole token is accepted
    if (last_accept_pos == token_len - 1) {
      RollbackChars(last_accept_pos - i + 2);
      return true;
    }
    // Case 3. The token is not accepted. Check the next position.
    RollbackChars(last_accept_pos - i + 1);
  }

  RollbackChars(1);
  return false;
}

inline CatagorizedTokens GrammarStateMatcherForInitContext::GetCatagorizedTokens(
    int vocab_size, const std::vector<std::pair<int32_t, std::string>>& sorted_token_table,
    bool consider_parent_rule) {
  tmp_accepted_indices_.clear();
  tmp_rejected_indices_.clear();
  tmp_uncertain_indices_.clear();

  // For every character in the current token, stores whether it is possible to reach the end of
  // the rule when matching until this character. Store it in a stack for later rollback.
  tmp_can_reach_end_stack_.assign({CanReachEnd()});
  tmp_can_reach_end_prefix_or_stack_.assign({tmp_can_reach_end_stack_.back()});

  int prev_matched_size = 0;
  for (int i = 0; i < static_cast<int>(sorted_token_table.size()); ++i) {
    const auto& token = sorted_token_table[i].second;

    bool accepted = true;

    // Many tokens may contain the same prefix, so we will avoid unnecessary matching
    // by finding the longest common prefix with the previous token.
    if (i > 0) {
      const auto& prev_token = sorted_token_table[i - 1].second;
      int lcp_len =
          std::mismatch(token.begin(), token.end(), prev_token.begin(), prev_token.end()).first -
          token.begin();
      if (lcp_len > prev_matched_size) {
        // Case 1. The common prefix is rejected by the matcher in the last token. Reject directly.
        accepted = false;
      } else if (lcp_len < prev_matched_size) {
        // Case 2. The common prefix is shorter than the previous matched size. Rollback
        // the non-common part.
        RollbackChars(prev_matched_size - lcp_len);
        tmp_can_reach_end_stack_.erase(
            tmp_can_reach_end_stack_.end() - (prev_matched_size - lcp_len),
            tmp_can_reach_end_stack_.end());
        tmp_can_reach_end_prefix_or_stack_.erase(
            tmp_can_reach_end_prefix_or_stack_.end() - (prev_matched_size - lcp_len),
            tmp_can_reach_end_prefix_or_stack_.end());
      }
      prev_matched_size = std::min(prev_matched_size, lcp_len);
    }

    if (accepted) {
      // Accept the rest chars one by one
      for (int j = prev_matched_size; j < token.size(); ++j) {
        if (!AcceptChar(token[j], false)) {
          accepted = false;
          break;
        }
        tmp_can_reach_end_stack_.push_back(CanReachEnd());
        tmp_can_reach_end_prefix_or_stack_.push_back(tmp_can_reach_end_stack_.back() ||
                                                     tmp_can_reach_end_prefix_or_stack_.back());
        prev_matched_size = j + 1;
      }
    }

    bool can_reach_end = tmp_can_reach_end_prefix_or_stack_.back();

    if (accepted) {
      tmp_accepted_indices_.push_back(i);
    } else if (can_reach_end && consider_parent_rule &&
               IsTokenPassLookaheadAssertion(token, tmp_can_reach_end_stack_)) {
      // 1. If the current rule is the main rule (consider_parent_rule=false), there are no
      // uncertain tokens. Not accepted tokens are just rejected.
      // 2. If a token cannot pass the lookahead assertion, it is rejected.
      tmp_uncertain_indices_.push_back(i);
    } else {
      tmp_rejected_indices_.push_back(i);
    }
  }
  // Rollback the last matched part
  RollbackChars(prev_matched_size);
  return CatagorizedTokens(vocab_size, sorted_token_table, tmp_accepted_indices_,
                           tmp_rejected_indices_, tmp_uncertain_indices_);
}

inline std::shared_ptr<GrammarStateInitContext> GrammarStateMatcher::CreateInitContext(
    const BNFGrammar& grammar, const std::vector<std::string>& token_table) {
  using RuleExprType = BNFGrammarNode::RuleExprType;
  auto ptr = std::make_shared<GrammarStateInitContext>();

  ptr->grammar = grammar;
  ptr->vocab_size = token_table.size();
  ptr->token_table = token_table;

  if (ptr->vocab_size == 0) {
    return ptr;
  }

  for (int i = 0; i < token_table.size(); ++i) {
    const auto& token = token_table[i];
    // LLaMA2: </s>
    // LLaMA3: <|end_of_text|>, <|eot_id|>
    // Phi-2: <|endoftext|>
    // Gemma: <eos>, <end_of_turn>
    if (token == "</s>" || token == "<|end_of_text|>" || token == "<|eot_id|>" ||
        token == "<|endoftext|>" || token == "<eos>" || token == "<end_of_turn>") {
      ptr->stop_token_ids.push_back(i);
    } else if ((token[0] == '<' && token.back() == '>' && token.size() >= 3) ||
               token == "[@BOS@]") {
      // gemma treats [@BOS@] as a special token
      ptr->special_token_ids.insert(i);
    } else {
      ptr->sorted_token_table.push_back({i, token});
    }
  }

  auto f_compare_token = [](const std::pair<int32_t, std::string>& a,
                            const std::pair<int32_t, std::string>& b) {
    return a.second < b.second;
  };
  std::sort(ptr->sorted_token_table.begin(), ptr->sorted_token_table.end(), f_compare_token);

  // Find the corresponding catagorized tokens for:
  // 1. All character class or character class star (with last_utf8_bytes=0, 1, 2, 3)
  // 2. All byte strings (with element_in_string=0, 1, 2, ...)
  auto main_rule_id = grammar->GetMainRuleId();
  for (int rule_id = 0; rule_id < static_cast<int>(grammar->NumRules()); ++rule_id) {
    auto rule = grammar->GetRule(rule_id);
    auto rule_body = grammar->GetRuleExpr(rule.body_expr_id);
    DCHECK(rule_body.type == RuleExprType::kChoices);
    for (auto sequence_id : rule_body) {
      auto sequence = grammar->GetRuleExpr(sequence_id);
      if (sequence.type == RuleExprType::kEmptyStr) {
        continue;
      }
      DCHECK(sequence.type == RuleExprType::kSequence);
      for (int element_id = 0; element_id < sequence.size(); ++element_id) {
        auto element = grammar->GetRuleExpr(sequence[element_id]);
        if (element.type == RuleExprType::kRuleRef) {
          continue;
        }

        auto add_catagorized_tokens = [&](const RulePosition& rule_position) {
          auto grammar_state_matcher = GrammarStateMatcherForInitContext(grammar, rule_position);
          auto cur_catagorized_tokens_for_grammar = grammar_state_matcher.GetCatagorizedTokens(
              ptr->vocab_size, ptr->sorted_token_table, rule_id != main_rule_id);
          ptr->catagorized_tokens_for_grammar[rule_position] = cur_catagorized_tokens_for_grammar;
        };

        auto cur_rule_position = RulePosition(rule_id, sequence_id, element_id);
        if (element.type == RuleExprType::kByteString) {
          for (int idx = 0; idx < element.size(); ++idx) {
            cur_rule_position.element_in_string = idx;
            add_catagorized_tokens(cur_rule_position);
          }
        } else {
          DCHECK(element.type == RuleExprType::kCharacterClassStar ||
                 element.type == RuleExprType::kCharacterClass);
          for (int left_utf8_bytes = 0; left_utf8_bytes <= 3; ++left_utf8_bytes) {
            cur_rule_position.left_utf8_bytes = left_utf8_bytes;
            add_catagorized_tokens(cur_rule_position);
          }
        }
      }
    }
  }
  return ptr;
}

class GrammarInitContextCacheImpl : public GrammarInitContextCacheNode {
 public:
  GrammarInitContextCacheImpl(const std::vector<std::string>& token_table);

  std::shared_ptr<GrammarStateInitContext> GetInitContextForJSONSchema(
      const std::string& schema) final;

  std::shared_ptr<GrammarStateInitContext> GetInitContextForJSON() final;

  void Clear() final;

 private:
  /*! \brief The token table associated with this storage class. */
  std::vector<std::string> token_table_;
  /*! \brief The cache for the init context of a JSON schema. */
  std::unordered_map<std::string, std::shared_ptr<GrammarStateInitContext>>
      init_ctx_for_schema_cache_;
  /*! \brief The init context for JSON. */
  std::shared_ptr<GrammarStateInitContext> init_ctx_for_json_;
};

inline GrammarInitContextCacheImpl::GrammarInitContextCacheImpl(
    const std::vector<std::string>& token_table)
    : token_table_(token_table) {
  init_ctx_for_json_ =
      GrammarStateMatcher::CreateInitContext(BNFGrammar::GetGrammarOfJSON(), token_table_);
}

inline std::shared_ptr<GrammarStateInitContext>
GrammarInitContextCacheImpl::GetInitContextForJSONSchema(const std::string& schema) {
  auto it = init_ctx_for_schema_cache_.find(schema);
  if (it != init_ctx_for_schema_cache_.end()) {
    return it->second;
  }
  auto init_ctx =
      GrammarStateMatcher::CreateInitContext(BNFGrammar::FromSchema(schema), token_table_);
  init_ctx_for_schema_cache_[schema] = init_ctx;
  return init_ctx;
}

inline std::shared_ptr<GrammarStateInitContext>
GrammarInitContextCacheImpl::GetInitContextForJSON() {
  return init_ctx_for_json_;
}

inline void GrammarInitContextCacheImpl::Clear() { init_ctx_for_schema_cache_.clear(); }

GrammarInitContextCache::GrammarInitContextCache(const std::vector<std::string>& token_table)
    : ObjectRef(make_object<GrammarInitContextCacheImpl>(token_table)) {}

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // TVM_LLVM_COMPILE_ENGINE_CPP_SERVE_GRAMMAR_STATE_MATCHER_PREPROC_H_
