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

RequestModelState::RequestModelState(Request request, int model_id, int64_t internal_id,
                                     Array<Data> inputs) {
  ObjectPtr<RequestModelStateNode> n = make_object<RequestModelStateNode>();
  n->request = std::move(request);
  n->model_id = model_id;
  n->internal_id = internal_id;
  n->inputs = std::move(inputs);
  data_ = std::move(n);
}

int RequestModelStateNode::GetInputLength() const {
  int total_length = 0;
  for (Data input : inputs) {
    total_length += input->GetLength();
  }
  return total_length;
}

std::vector<int> RequestModelStateNode::GetTokenBitmask(int vocab_size) const {
  // TODO(mlc-team): implement this function.
  return std::vector<int>();
}

void RequestModelStateNode::CommitToken(SampleResult sampled_token) {
  committed_tokens.push_back(std::move(sampled_token));
  appeared_token_ids[sampled_token.sampled_token_id.first] += 1;
}

void RequestModelStateNode::AddDraftToken(SampleResult sampled_token, NDArray prob_dist) {
  draft_output_tokens.push_back(std::move(sampled_token));
  draft_output_prob_dist.push_back(std::move(prob_dist));
  appeared_token_ids[sampled_token.sampled_token_id.first] += 1;
}

void RequestModelStateNode::RemoveLastDraftToken() {
  ICHECK(!draft_output_tokens.empty());
  auto it = appeared_token_ids.find(draft_output_tokens.back().sampled_token_id.first);
  draft_output_tokens.pop_back();
  draft_output_prob_dist.pop_back();
  CHECK(it != appeared_token_ids.end());
  if (--it->second == 0) {
    appeared_token_ids.erase(it);
  }
}

void RequestModelStateNode::RemoveAllDraftTokens() {
  while (!draft_output_tokens.empty()) {
    RemoveLastDraftToken();
  }
}

TVM_REGISTER_OBJECT_TYPE(RequestStateNode);

RequestState::RequestState(Request request, int num_models, int64_t internal_id,
                           const std::vector<std::string>& token_table) {
  ObjectPtr<RequestStateNode> n = make_object<RequestStateNode>();
  Array<RequestModelState> mstates;
  mstates.reserve(num_models);
  for (int i = 0; i < num_models; ++i) {
    mstates.push_back(RequestModelState(request, i, internal_id, request->inputs));
  }
  n->rng = RandomGenerator(request->generation_cfg->seed);
  n->stop_str_handler = StopStrHandler(
      !request->generation_cfg->ignore_eos ? request->generation_cfg->stop_strs : Array<String>(),
      token_table);
  n->request = std::move(request);
  n->mstates = std::move(mstates);
  n->next_callback_token_pos = 0;
  n->tadd = std::chrono::high_resolution_clock::now();
  data_ = std::move(n);
}

DeltaRequestReturn RequestStateNode::GetReturnTokenIds(const Tokenizer& tokenizer,
                                                       int max_single_sequence_length) {
  // - Case 0. There is remaining draft output ==> Unfinished
  //   All draft outputs are supposed to be processed before finish.
  for (RequestModelState mstate : mstates) {
    if (!mstate->draft_output_tokens.empty()) {
      return {{}, {}, Optional<String>()};
    }
  }

  std::vector<int32_t> return_token_ids;
  std::vector<String> logprob_json_strs;
  Optional<String> finish_reason;
  const std::vector<SampleResult>& committed_tokens = mstates[0]->committed_tokens;
  int num_committed_tokens = committed_tokens.size();
  ICHECK_LE(this->next_callback_token_pos, num_committed_tokens);

  // Case 1. Any of the stop strings is matched.
  ICHECK(!stop_str_handler->StopTriggered());
  while (next_callback_token_pos < num_committed_tokens) {
    std::vector<int32_t> delta_token_ids =
        stop_str_handler->Put(committed_tokens[next_callback_token_pos].sampled_token_id.first);
    logprob_json_strs.push_back(committed_tokens[next_callback_token_pos].GetLogProbJSON(
        tokenizer, request->generation_cfg->logprobs));
    ++next_callback_token_pos;
    return_token_ids.insert(return_token_ids.end(), delta_token_ids.begin(), delta_token_ids.end());
    if (stop_str_handler->StopTriggered()) {
      finish_reason = "stop";
      break;
    }
  }

  // Case 2. Any of the stop tokens appears in the committed tokens ===> Finished
  // `stop_token_ids` includes the stop tokens from conversation template and user-provided tokens.
  // This check will be ignored when `ignore_eos` is set for the benchmarking purpose.
  if (!request->generation_cfg->ignore_eos) {
    for (int i = 0; i < static_cast<int>(return_token_ids.size()); ++i) {
      if (std::any_of(
              request->generation_cfg->stop_token_ids.begin(),
              request->generation_cfg->stop_token_ids.end(),
              [&return_token_ids, i](int32_t token) { return token == return_token_ids[i]; })) {
        // Stop token matched. Erase all tokens after the current position.
        finish_reason = "stop";
        while (static_cast<int>(return_token_ids.size()) > i) {
          return_token_ids.pop_back();
        }
        break;
      }
    }
  }

  if (finish_reason.defined()) {
    return {return_token_ids, logprob_json_strs, finish_reason};
  }

  // Case 3. Generation reaches the specified max generation length ==> Finished
  // `max_tokens` means the generation length is limited by model capacity.
  if (request->generation_cfg->max_tokens >= 0 &&
      num_committed_tokens >= request->generation_cfg->max_tokens) {
    std::vector<int32_t> remaining = stop_str_handler->Finish();
    return_token_ids.insert(return_token_ids.end(), remaining.begin(), remaining.end());
    return {return_token_ids, logprob_json_strs, String("length")};
  }
  // Case 4. Total length of the request reaches the maximum single sequence length ==> Finished
  if (request->input_total_length + num_committed_tokens >= max_single_sequence_length) {
    std::vector<int32_t> remaining = stop_str_handler->Finish();
    return_token_ids.insert(return_token_ids.end(), remaining.begin(), remaining.end());
    return {return_token_ids, logprob_json_strs, String("length")};
  }
  return {return_token_ids, logprob_json_strs, Optional<String>()};
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
