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
}

void RequestModelStateNode::RemoveLastDraftToken() {
  ICHECK(!draft_output_tokens.empty());
  draft_output_tokens.pop_back();
  draft_token_parent_idx.pop_back();
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

/****************** RequestActionPostProcWorkspace ******************/

RequestStreamOutput RequestActionPostProcWorkspace::GetStreamOutput() {
  for (const RequestStreamOutput& stream_output : stream_outputs) {
    if (stream_output->unpacked) {
      return stream_output;
    }
  }

  ICHECK(!stream_outputs.empty());
  int num_response = stream_outputs[0]->group_delta_token_ids.size();
  std::vector<std::vector<int64_t>> group_delta_token_ids;
  std::vector<std::vector<String>> group_delta_logprob_json_strs;
  std::vector<Optional<String>> group_finish_reason;
  std::vector<String> group_extra_prefix_string;
  group_delta_token_ids.resize(num_response);
  group_finish_reason.resize(num_response);
  group_extra_prefix_string.resize(num_response);
  if (stream_outputs[0]->group_delta_logprob_json_strs.has_value()) {
    group_delta_logprob_json_strs.resize(num_response);
  }
  RequestStreamOutput stream_output(stream_outputs[0]->request_id, std::move(group_delta_token_ids),
                                    stream_outputs[0]->group_delta_logprob_json_strs.has_value()
                                        ? std::make_optional(group_delta_logprob_json_strs)
                                        : std::nullopt,
                                    std::move(group_finish_reason),
                                    std::move(group_extra_prefix_string));
  stream_outputs.push_back(stream_output);
  return stream_output;
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

void RequestStateEntryNode::GetDeltaRequestReturn(const Tokenizer& tokenizer,
                                                  int64_t max_single_sequence_length,
                                                  RequestStreamOutput* delta_stream_output,
                                                  int idx) {
  ICHECK_NOTNULL(delta_stream_output);
  bool needs_logprobs = (*delta_stream_output)->group_delta_logprob_json_strs.has_value();
  (*delta_stream_output)->group_delta_token_ids[idx].clear();
  if (needs_logprobs) {
    (*delta_stream_output)->group_delta_logprob_json_strs.value()[idx].clear();
  }
  (*delta_stream_output)->group_finish_reason[idx] = NullOpt;
  (*delta_stream_output)->group_extra_prefix_string[idx] = this->extra_prefix_string;
  this->extra_prefix_string.clear();

  const std::vector<SampleResult>& committed_tokens = this->mstates[0]->committed_tokens;
  int num_committed_tokens = committed_tokens.size();
  ICHECK_LE(this->next_callback_token_pos, num_committed_tokens);

  // Case 1. There is no new token ids.
  if (this->next_callback_token_pos == num_committed_tokens && extra_prefix_string.empty()) {
    return;
  }

  // Case 2. Any of the stop strings is matched.
  ICHECK(!stop_str_handler->StopTriggered());
  while (next_callback_token_pos < num_committed_tokens) {
    stop_str_handler->Put(committed_tokens[next_callback_token_pos].GetTokenId(),
                          &(*delta_stream_output)->group_delta_token_ids[idx]);
    if (needs_logprobs) {
      (*delta_stream_output)
          ->group_delta_logprob_json_strs.value()[idx]
          .push_back(committed_tokens[next_callback_token_pos].GetLogProbJSON(
              tokenizer, request->generation_cfg->logprobs));
    }
    ++next_callback_token_pos;
    if (stop_str_handler->StopTriggered()) {
      (*delta_stream_output)->group_finish_reason[idx] = "stop";
      break;
    }
  }

  // Case 3. Any of the stop tokens appears in the committed tokens ===> Finished
  // `stop_token_ids` includes the stop tokens from conversation template and user-provided tokens.
  // This check will be ignored when `ignore_eos` is set for the benchmarking purpose.
  if (!request->generation_cfg->debug_config.ignore_eos) {
    for (int i = 0; i < static_cast<int>((*delta_stream_output)->group_delta_token_ids[idx].size());
         ++i) {
      if (std::any_of(request->generation_cfg->stop_token_ids.begin(),
                      request->generation_cfg->stop_token_ids.end(),
                      [delta_stream_output, idx, i](int32_t token) {
                        return token == (*delta_stream_output)->group_delta_token_ids[idx][i];
                      })) {
        // Stop token matched. Erase the stop token and all tokens after it.
        (*delta_stream_output)->group_finish_reason[idx] = "stop";
        while (static_cast<int>((*delta_stream_output)->group_delta_token_ids[idx].size()) > i) {
          (*delta_stream_output)->group_delta_token_ids[idx].pop_back();
        }
        break;
      }
    }
  }

  // Case 4. When stop token is not detected (e.g. ignore_eos is set), but the grammar state is
  // terminated, stop the generation and pop the last token (used to trigger the termination).
  if ((*delta_stream_output)->group_finish_reason[idx] != "stop" &&
      this->mstates[0]->grammar_state_matcher.defined() &&
      this->mstates[0]->grammar_state_matcher.value()->IsTerminated()) {
    (*delta_stream_output)->group_delta_token_ids[idx].pop_back();
    (*delta_stream_output)->group_finish_reason[idx] = "stop";
  }

  if ((*delta_stream_output)->group_finish_reason[idx].defined()) {
    return;
  }

  // Case 5. Generation reaches the specified max generation length ==> Finished
  // `max_tokens` means the generation length is limited by model capacity.
  if (request->generation_cfg->max_tokens >= 0 &&
      num_committed_tokens >= request->generation_cfg->max_tokens) {
    stop_str_handler->Finish(&(*delta_stream_output)->group_delta_token_ids[idx]);
    (*delta_stream_output)->group_finish_reason[idx] = "length";
    return;
  }
  // Case 6. Total length of the request reaches the maximum single sequence length ==> Finished
  if (request->prompt_tokens + num_committed_tokens >= max_single_sequence_length) {
    stop_str_handler->Finish(&(*delta_stream_output)->group_delta_token_ids[idx]);
    (*delta_stream_output)->group_finish_reason[idx] = "length";
  }
}

/****************** RequestState ******************/

TVM_REGISTER_OBJECT_TYPE(RequestStateNode);

RequestState::RequestState(std::vector<RequestStateEntry> entries, int num_response,
                           std::chrono::high_resolution_clock::time_point add_time_point) {
  ICHECK(!entries.empty());
  ObjectPtr<RequestStateNode> n = make_object<RequestStateNode>();
  n->entries = std::move(entries);
  n->metrics.prompt_tokens = n->entries[0]->request->prompt_tokens;
  n->metrics.add_time_point = add_time_point;

  std::vector<std::vector<int64_t>> group_delta_token_ids;
  std::vector<std::vector<String>> group_delta_logprob_json_strs;
  std::vector<Optional<String>> group_finish_reason;
  std::vector<String> group_extra_prefix_string;
  group_delta_token_ids.resize(num_response);
  group_finish_reason.resize(num_response);
  group_extra_prefix_string.resize(num_response);
  if (n->entries[0]->request->generation_cfg->logprobs) {
    group_delta_logprob_json_strs.resize(num_response);
  }
  RequestStreamOutput stream_output(n->entries[0]->request->id, std::move(group_delta_token_ids),
                                    n->entries[0]->request->generation_cfg->logprobs
                                        ? std::make_optional(group_delta_logprob_json_strs)
                                        : std::nullopt,
                                    std::move(group_finish_reason),
                                    std::move(group_extra_prefix_string));
  stream_output->unpacked = true;
  n->postproc_states.stream_outputs = {std::move(stream_output)};
  data_ = std::move(n);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
