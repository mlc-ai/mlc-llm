/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/request_state.cc
 */

#include "request_state.h"

namespace mlc {
namespace llm {
namespace serve {

/****************** RequestModelState ******************/

TVM_REGISTER_OBJECT_TYPE(RequestModelStateNode);

RequestModelState::RequestModelState(
    Request request, int model_id, int64_t internal_id, Array<Data> inputs,
    const std::optional<std::shared_ptr<GrammarStateInitContext>>& grammar_state_init_ctx) {
  ObjectPtr<RequestModelStateNode> n = make_object<RequestModelStateNode>();
  n->model_id = model_id;
  n->internal_id = internal_id;
  n->inputs = std::move(inputs);

  if (grammar_state_init_ctx.has_value()) {
    // TODO(yixin): set rollback limit to a configurable value.
    n->grammar_state_matcher = GrammarStateMatcher(grammar_state_init_ctx.value(), 10);
  }

  n->request = std::move(request);
  data_ = std::move(n);
}

int RequestModelStateNode::GetInputLength() const {
  int total_length = 0;
  for (Data input : inputs) {
    total_length += input->GetLength();
  }
  return total_length;
}

bool RequestModelStateNode::RequireNextTokenBitmask() { return grammar_state_matcher.defined(); }

void RequestModelStateNode::FindNextTokenBitmask(DLTensor* bitmask) {
  ICHECK(grammar_state_matcher.defined());

  grammar_state_matcher.value()->FindNextTokenBitmask(bitmask);
}

void RequestModelStateNode::CommitToken(SampleResult sampled_token) {
  committed_tokens.push_back(std::move(sampled_token));
  appeared_token_ids[sampled_token.GetTokenId()] += 1;
  // There will be one more token that will be processed in the next decoding.
  ++num_tokens_for_next_decode;

  // Update the grammar matcher state if it exists.
  if (grammar_state_matcher) {
    bool accepted = grammar_state_matcher.value()->AcceptToken(sampled_token.GetTokenId());
    ICHECK(accepted) << "Token id " << sampled_token.GetTokenId()
                     << " is not accepted by the grammar state matcher.";
  }
}

void RequestModelStateNode::RollbackTokens(int count) {
  ICHECK(count <= static_cast<int>(committed_tokens.size()));
  for (int i = 0; i < count; ++i) {
    auto it = appeared_token_ids.find(committed_tokens.back().GetTokenId());
    CHECK(it != appeared_token_ids.end());
    if (--it->second == 0) {
      appeared_token_ids.erase(it);
    }
    committed_tokens.pop_back();
    if (grammar_state_matcher) {
      grammar_state_matcher.value()->Rollback(1);
    }
  }
}

void RequestModelStateNode::AddDraftToken(SampleResult sampled_token, int draft_token_slot,
                                          int64_t parent_idx) {
  draft_output_tokens.push_back(std::move(sampled_token));
  draft_token_slots.push_back(draft_token_slot);
  draft_token_parent_idx.push_back(parent_idx);
  appeared_token_ids[sampled_token.GetTokenId()] += 1;
}

void RequestModelStateNode::RemoveLastDraftToken() {
  ICHECK(!draft_output_tokens.empty());
  auto it = appeared_token_ids.find(draft_output_tokens.back().GetTokenId());
  draft_output_tokens.pop_back();
  draft_token_parent_idx.pop_back();
  CHECK(it != appeared_token_ids.end());
  if (--it->second == 0) {
    appeared_token_ids.erase(it);
  }
}

void RequestModelStateNode::RemoveAllDraftTokens(std::vector<int>* removed_draft_token_slots) {
  if (removed_draft_token_slots != nullptr) {
    removed_draft_token_slots->assign(draft_token_slots.begin(), draft_token_slots.end());
  }
  draft_token_slots.clear();
  while (!draft_output_tokens.empty()) {
    RemoveLastDraftToken();
  }
}

/****************** RequestStateEntry ******************/

TVM_REGISTER_OBJECT_TYPE(RequestStateEntryNode);

RequestStateEntry::RequestStateEntry(
    Request request, int num_models, int64_t internal_id, int rng_seed,
    const std::vector<std::string>& token_table,
    const std::optional<std::shared_ptr<GrammarStateInitContext>>& grammar_state_init_ctx,
    int parent_idx) {
  ObjectPtr<RequestStateEntryNode> n = make_object<RequestStateEntryNode>();
  Array<RequestModelState> mstates;
  Array<Data> inputs;
  if (parent_idx == -1) {
    inputs = request->inputs;
  }
  mstates.reserve(num_models);
  for (int i = 0; i < num_models; ++i) {
    mstates.push_back(RequestModelState(request, i, internal_id, inputs, grammar_state_init_ctx));
  }
  n->status = RequestStateStatus::kPending;
  n->rng = RandomGenerator(rng_seed);
  n->stop_str_handler = StopStrHandler(!request->generation_cfg->debug_config.ignore_eos
                                           ? request->generation_cfg->stop_strs
                                           : Array<String>(),
                                       token_table);
  n->request = std::move(request);
  n->parent_idx = parent_idx;
  n->mstates = std::move(mstates);
  n->next_callback_token_pos = 0;
  data_ = std::move(n);
}

DeltaRequestReturn RequestStateEntryNode::GetDeltaRequestReturn(
    const Tokenizer& tokenizer, int64_t max_single_sequence_length) {
  std::vector<int32_t> return_token_ids;
  std::vector<String> logprob_json_strs;
  Optional<String> finish_reason;

  String extra_prefix_string = this->extra_prefix_string;
  this->extra_prefix_string.clear();

  const std::vector<SampleResult>& committed_tokens = this->mstates[0]->committed_tokens;
  int num_committed_tokens = committed_tokens.size();
  ICHECK_LE(this->next_callback_token_pos, num_committed_tokens);

  // Case 1. There is no new token ids.
  if (this->next_callback_token_pos == num_committed_tokens && extra_prefix_string.empty()) {
    return {{}, {}, Optional<String>(), extra_prefix_string};
  }

  // Case 2. Any of the stop strings is matched.
  ICHECK(!stop_str_handler->StopTriggered());
  while (next_callback_token_pos < num_committed_tokens) {
    std::vector<int32_t> delta_token_ids =
        stop_str_handler->Put(committed_tokens[next_callback_token_pos].GetTokenId());
    logprob_json_strs.push_back(committed_tokens[next_callback_token_pos].GetLogProbJSON(
        tokenizer, request->generation_cfg->logprobs));
    ++next_callback_token_pos;
    return_token_ids.insert(return_token_ids.end(), delta_token_ids.begin(), delta_token_ids.end());
    if (stop_str_handler->StopTriggered()) {
      finish_reason = "stop";
      break;
    }
  }

  // Case 3. Any of the stop tokens appears in the committed tokens ===> Finished
  // `stop_token_ids` includes the stop tokens from conversation template and user-provided tokens.
  // This check will be ignored when `ignore_eos` is set for the benchmarking purpose.
  if (!request->generation_cfg->debug_config.ignore_eos) {
    for (int i = 0; i < static_cast<int>(return_token_ids.size()); ++i) {
      if (std::any_of(
              request->generation_cfg->stop_token_ids.begin(),
              request->generation_cfg->stop_token_ids.end(),
              [&return_token_ids, i](int32_t token) { return token == return_token_ids[i]; })) {
        // Stop token matched. Erase the stop token and all tokens after it.
        finish_reason = "stop";
        while (static_cast<int>(return_token_ids.size()) > i) {
          return_token_ids.pop_back();
        }
        break;
      }
    }
  }

  // Case 4. When stop token is not detected (e.g. ignore_eos is set), but the grammar state is
  // terminated, stop the generation and pop the last token (used to trigger the termination).
  if (finish_reason != "stop" && this->mstates[0]->grammar_state_matcher.defined() &&
      this->mstates[0]->grammar_state_matcher.value()->IsTerminated()) {
    return_token_ids.pop_back();
    finish_reason = "stop";
  }

  if (finish_reason.defined()) {
    return {return_token_ids, logprob_json_strs, finish_reason, extra_prefix_string};
  }

  // Case 5. Generation reaches the specified max generation length ==> Finished
  // `max_tokens` means the generation length is limited by model capacity.
  if (request->generation_cfg->max_tokens >= 0 &&
      num_committed_tokens >= request->generation_cfg->max_tokens) {
    std::vector<int32_t> remaining = stop_str_handler->Finish();
    return_token_ids.insert(return_token_ids.end(), remaining.begin(), remaining.end());
    return {return_token_ids, logprob_json_strs, String("length"), extra_prefix_string};
  }
  // Case 6. Total length of the request reaches the maximum single sequence length ==> Finished
  if (request->prompt_tokens + num_committed_tokens >= max_single_sequence_length) {
    std::vector<int32_t> remaining = stop_str_handler->Finish();
    return_token_ids.insert(return_token_ids.end(), remaining.begin(), remaining.end());
    return {return_token_ids, logprob_json_strs, String("length"), extra_prefix_string};
  }
  return {return_token_ids, logprob_json_strs, Optional<String>(), extra_prefix_string};
}

/****************** RequestState ******************/

TVM_REGISTER_OBJECT_TYPE(RequestStateNode);

RequestState::RequestState(std::vector<RequestStateEntry> entries,
                           std::chrono::high_resolution_clock::time_point add_time_point) {
  ICHECK(!entries.empty());
  ObjectPtr<RequestStateNode> n = make_object<RequestStateNode>();
  n->entries = std::move(entries);
  n->metrics.prompt_tokens = n->entries[0]->request->prompt_tokens;
  n->metrics.add_time_point = add_time_point;
  data_ = std::move(n);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
