/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar_state_matcher_preproc.h
 * \brief The header for the preprocessing of the grammar state matcher.
 */
#ifndef MLC_LLM_SERVE_GRAMMAR_GRAMMAR_STATE_MATCHER_PREPROC_H_
#define MLC_LLM_SERVE_GRAMMAR_GRAMMAR_STATE_MATCHER_PREPROC_H_

#include <vector>

#include "../../support/encoding.h"
#include "grammar.h"
#include "grammar_state_matcher_base.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/*! \brief A token and its id. */
struct TokenAndId {
  std::vector<TCodepoint> token;
  int32_t id;
  /*! \brief Compare tokens by their unicode codepoint sequence. */
  bool operator<(const TokenAndId& other) const;
};

/*!
 * \brief Preprocessed information, for a given specific rule and position, divides the token set
 * into three categories: accepted, rejected, and uncertain.
 * \note Since the union of these three sets is the whole token set, we only need to store the
 * smaller two sets. The unsaved set is specified by not_saved_index.
 * \note These indices are the indices of sorted_token_codepoints in the GrammarStateInitContext
 * object, instead of the token ids. That helps the matching process.
 */
struct CatagorizedTokens {
  std::vector<int32_t> accepted_indices;
  std::vector<int32_t> rejected_indices;
  std::vector<int32_t> uncertain_indices;
  enum class NotSavedIndex { kAccepted = 0, kRejected = 1, kUncertain = 2 };
  NotSavedIndex not_saved_index;

  CatagorizedTokens() = default;

  CatagorizedTokens(std::vector<int32_t>&& accepted_indices,
                    std::vector<int32_t>&& rejected_indices,
                    std::vector<int32_t>&& uncertain_indices);
};

/*!
 * \brief All information that we need to match tokens in the tokenizer to the specified grammar.
 * It is the result of preprocessing.
 * \sa mlc::llm::serve::GrammarStateMatcher
 */
class GrammarStateInitContext {
 public:
  /******************* Information about the tokenizer *******************/

  /*! \brief The vocabulary size of the tokenizer. */
  size_t vocab_size;
  /*! \brief All tokens represented by the id and codepoints of each. The tokens are sorted by
   * codepoint values to reuse the common prefix during matching. */
  std::vector<TokenAndId> sorted_token_codepoints;
  /*! \brief The mapping from token id to token represented by codepoints. Only contains
   * non-special and non-stop tokens. */
  std::unordered_map<int32_t, TokenAndId> id_to_token_codepoints;
  /*! \brief The stop tokens. They can be accepted iff GramamrMatcher can reach the end of the
   * grammar. */
  std::vector<int32_t> stop_token_ids;
  /*! \brief The special tokens. Currently we will ignore these tokens during grammar-guided
   * matching. */
  std::vector<int32_t> special_token_ids;

  /******************* Information about the grammar *******************/

  BNFGrammar grammar;

  /******************* Grammar-specific tokenizer information *******************/

  /*! \brief A sequence id and its position. */
  struct SequenceIdAndPosition {
    int32_t sequence_id;
    int32_t element_id;
    bool operator==(const SequenceIdAndPosition& other) const {
      return sequence_id == other.sequence_id && element_id == other.element_id;
    }
  };

  /*! \brief Hash function for SequenceIdAndPosition. */
  struct SequenceIdAndPositionHash {
    std::size_t operator()(const SequenceIdAndPosition& k) const {
      return std::hash<int32_t>()(k.sequence_id) ^ (std::hash<int32_t>()(k.element_id) << 1);
    }
  };

  /*! \brief Mapping from sequence id and its position to the catagorized tokens. */
  std::unordered_map<SequenceIdAndPosition, CatagorizedTokens, SequenceIdAndPositionHash>
      catagorized_tokens_for_grammar;
};

/* \brief The concrete implementation of GrammarStateMatcherNode. */
class GrammarStateMatcherForInitContext : public GrammarStateMatcherBase {
 public:
  GrammarStateMatcherForInitContext(const BNFGrammar& grammar, RulePosition init_rule_position)
      : GrammarStateMatcherBase(grammar, init_rule_position) {}

  CatagorizedTokens GetCatagorizedTokens(const std::vector<TokenAndId>& sorted_token_codepoints,
                                         bool is_main_rule);

 private:
  using RuleExpr = BNFGrammarNode::RuleExpr;
  using RuleExprType = BNFGrammarNode::RuleExprType;

  // Temporary data for GetCatagorizedTokens.
  std::vector<int32_t> tmp_accepted_indices_;
  std::vector<int32_t> tmp_rejected_indices_;
  std::vector<int32_t> tmp_uncertain_indices_;
  std::vector<bool> tmp_can_see_end_stack_;
};

inline bool TokenAndId::operator<(const TokenAndId& other) const {
  for (size_t i = 0; i < token.size(); ++i) {
    if (i >= other.token.size()) {
      return false;
    }
    if (token[i] < other.token[i]) {
      return true;
    } else if (token[i] > other.token[i]) {
      return false;
    }
  }
  return token.size() < other.token.size();
}

inline CatagorizedTokens::CatagorizedTokens(std::vector<int32_t>&& accepted_indices,
                                            std::vector<int32_t>&& rejected_indices,
                                            std::vector<int32_t>&& uncertain_indices) {
  auto size_acc = accepted_indices.size();
  auto size_rej = rejected_indices.size();
  auto size_unc = uncertain_indices.size();
  not_saved_index =
      (size_acc >= size_rej && size_acc >= size_unc)
          ? NotSavedIndex::kAccepted
          : (size_rej >= size_unc ? NotSavedIndex::kRejected : NotSavedIndex::kUncertain);

  if (not_saved_index != NotSavedIndex::kAccepted) {
    this->accepted_indices = std::move(accepted_indices);
  }
  if (not_saved_index != NotSavedIndex::kRejected) {
    this->rejected_indices = std::move(rejected_indices);
  }
  if (not_saved_index != NotSavedIndex::kUncertain) {
    this->uncertain_indices = std::move(uncertain_indices);
  }
}

inline CatagorizedTokens GrammarStateMatcherForInitContext::GetCatagorizedTokens(
    const std::vector<TokenAndId>& sorted_token_codepoints, bool is_main_rule) {
  // Support the current stack contains only one stack with one RulePosition.
  // Iterate over all tokens. Split them into three categories:
  // - accepted_indices: If a token is accepted by current rule
  // - rejected_indices: If a token is rejected by current rule
  // - uncertain_indices: If a prefix of a token is accepted by current rule and comes to the end
  // of the rule.

  // Note many tokens may contain the same prefix, so we will avoid unnecessary matching

  tmp_accepted_indices_.clear();
  tmp_rejected_indices_.clear();
  tmp_uncertain_indices_.clear();
  // For every character in the current token, stores whether it is possible to reach the end of
  // the rule when matching until this character. Useful for rollback.
  tmp_can_see_end_stack_.assign({CanReachEnd()});

  int prev_matched_size = 0;
  for (int i = 0; i < static_cast<int>(sorted_token_codepoints.size()); ++i) {
    const auto& token = sorted_token_codepoints[i].token;
    const auto* prev_token = i > 0 ? &sorted_token_codepoints[i - 1].token : nullptr;

    // Find the longest common prefix with the accepted part of the previous token.
    auto prev_useful_size = 0;
    if (prev_token) {
      prev_useful_size = std::min(prev_matched_size, static_cast<int>(token.size()));
      for (int j = 0; j < prev_useful_size; ++j) {
        if (token[j] != (*prev_token)[j]) {
          prev_useful_size = j;
          break;
        }
      }
      RollbackCodepoints(prev_matched_size - prev_useful_size);
      tmp_can_see_end_stack_.erase(
          tmp_can_see_end_stack_.end() - (prev_matched_size - prev_useful_size),
          tmp_can_see_end_stack_.end());
    }

    // Find if the current token is accepted or rejected or uncertain.
    bool accepted = true;
    bool can_see_end = tmp_can_see_end_stack_.back();
    prev_matched_size = prev_useful_size;
    for (int j = prev_useful_size; j < token.size(); ++j) {
      if (!AcceptCodepoint(token[j], false)) {
        accepted = false;
        break;
      }
      if (CanReachEnd()) {
        can_see_end = true;
      }
      tmp_can_see_end_stack_.push_back(can_see_end);
      prev_matched_size = j + 1;
    }
    if (accepted) {
      tmp_accepted_indices_.push_back(i);
    } else if (can_see_end && !is_main_rule) {
      // If the current rule is the main rule, there will be no uncertain indices since we will
      // never consider its parent rule. Unaccepted tokens are just rejected.
      tmp_uncertain_indices_.push_back(i);
    } else {
      tmp_rejected_indices_.push_back(i);
    }
  }
  RollbackCodepoints(prev_matched_size);
  return CatagorizedTokens(std::move(tmp_accepted_indices_), std::move(tmp_rejected_indices_),
                           std::move(tmp_uncertain_indices_));
}

inline std::string ReplaceUnderscoreWithSpace(const std::string& str,
                                              const std::string& kSpecialUnderscore) {
  std::string res;
  size_t pos = 0;
  while (pos < str.size()) {
    size_t found = str.find(kSpecialUnderscore, pos);
    if (found == std::string::npos) {
      res += str.substr(pos);
      break;
    }
    res += str.substr(pos, found - pos) + " ";
    pos = found + kSpecialUnderscore.size();
  }
  return res;
}

inline std::shared_ptr<GrammarStateInitContext> GrammarStateMatcher::CreateInitContext(
    const BNFGrammar& grammar, const std::vector<std::string>& token_table) {
  using RuleExprType = BNFGrammarNode::RuleExprType;
  auto ptr = std::make_shared<GrammarStateInitContext>();

  ptr->grammar = grammar;
  ptr->vocab_size = token_table.size();

  if (ptr->vocab_size == 0) {
    return ptr;
  }

  for (int i = 0; i < token_table.size(); ++i) {
    auto token = token_table[i];
    if (token == "<unk>" || token == "<pad>" || token == "<s>") {
      ptr->special_token_ids.push_back(i);
    } else if (token == "</s>") {
      ptr->stop_token_ids.push_back(i);
    } else if (token.size() == 1 &&
               (static_cast<unsigned char>(token[0]) >= 128 || token[0] == 0)) {
      // Currently we consider all tokens with one character that >= 128 as special tokens,
      // and will ignore generating them during grammar-guided generation.
      ptr->special_token_ids.push_back(i);
    } else {
      // First replace the special underscore with space.
      auto codepoints = Utf8StringToCodepoints(token.c_str());
      DCHECK(!codepoints.empty() &&
             codepoints[0] != static_cast<TCodepoint>(CharHandlingError::kInvalidUtf8))
          << "Invalid token: " << token;
      ptr->sorted_token_codepoints.push_back({codepoints, i});
      ptr->id_to_token_codepoints[i] = {codepoints, i};
    }
  }
  std::sort(ptr->sorted_token_codepoints.begin(), ptr->sorted_token_codepoints.end());

  // Find the corresponding catagorized tokens for:
  // 1. All character elements in the grammar
  // 2. All RuleRef elements that refers to a rule containing a CharacterClassStar RuleExpr.
  for (int i = 0; i < static_cast<int>(grammar->NumRules()); ++i) {
    auto rule = grammar->GetRule(i);
    auto rule_expr = grammar->GetRuleExpr(rule.body_expr_id);
    // Skip CharacterClassStar since we just handle it at the reference element during matching.
    if (rule_expr.type == RuleExprType::kCharacterClassStar) {
      continue;
    }
    DCHECK(rule_expr.type == RuleExprType::kChoices);
    for (auto sequence_id : rule_expr) {
      auto sequence_expr = grammar->GetRuleExpr(sequence_id);
      if (sequence_expr.type == RuleExprType::kEmptyStr) {
        continue;
      }
      DCHECK(sequence_expr.type == RuleExprType::kSequence);
      for (int element_id = 0; element_id < sequence_expr.size(); ++element_id) {
        auto element_expr = grammar->GetRuleExpr(sequence_expr[element_id]);
        auto cur_rule_position = RulePosition{i, sequence_id, element_id};
        if (element_expr.type == RuleExprType::kRuleRef) {
          auto ref_rule = grammar->GetRule(element_expr[0]);
          auto ref_rule_expr = grammar->GetRuleExpr(ref_rule.body_expr_id);
          if (ref_rule_expr.type == RuleExprType::kChoices) {
            continue;
          } else {
            // Reference to a CharacterClassStar of a character class.
            cur_rule_position.char_class_star_id = ref_rule_expr[0];
          }
        }

        auto grammar_state_matcher = GrammarStateMatcherForInitContext(grammar, cur_rule_position);
        auto cur_catagorized_tokens_for_grammar =
            grammar_state_matcher.GetCatagorizedTokens(ptr->sorted_token_codepoints, i == 0);
        ptr->catagorized_tokens_for_grammar[{sequence_id, element_id}] =
            cur_catagorized_tokens_for_grammar;
      }
    }
  }
  return ptr;
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // TVM_LLVM_COMPILE_ENGINE_CPP_SERVE_GRAMMAR_STATE_MATCHER_PREPROC_H_
