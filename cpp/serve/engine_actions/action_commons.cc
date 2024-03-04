/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_actions/action_commons.cc
 */

#include "action_commons.h"

namespace mlc {
namespace llm {
namespace serve {

void RemoveRequestFromModel(EngineState estate, int64_t req_internal_id, Array<Model> models) {
  // Remove the request from all models (usually the KV cache).
  for (Model model : models) {
    model->RemoveSequence(req_internal_id);
  }
}

void ProcessFinishedRequestStateEntries(std::vector<RequestStateEntry> finished_rsentries,
                                        EngineState estate, Array<Model> models,
                                        int max_single_sequence_length) {
  // - Remove the finished request state entries.
  for (const RequestStateEntry& rsentry : finished_rsentries) {
    // The finished entry must be a leaf.
    ICHECK(rsentry->child_indices.empty());
    // Mark the status of this entry as finished.
    rsentry->status = RequestStateStatus::kFinished;
    // Remove the request state entry from all the models.
    RemoveRequestFromModel(estate, rsentry->mstates[0]->internal_id, models);
    estate->id_manager.RecycleId(rsentry->mstates[0]->internal_id);
    estate->stats.current_total_seq_len -=
        static_cast<int>(rsentry->mstates[0]->committed_tokens.size()) - 1;

    RequestState rstate = estate->GetRequestState(rsentry->request);
    int parent_idx = rsentry->parent_idx;
    while (parent_idx != -1) {
      bool all_children_finished = true;
      for (int child_idx : rstate->entries[parent_idx]->child_indices) {
        if (rstate->entries[child_idx]->status != RequestStateStatus::kFinished) {
          all_children_finished = false;
          break;
        }
      }
      if (!all_children_finished) {
        break;
      }

      // All the children of the parent request state entry have finished.
      // So we mark the parent entry as finished.
      rstate->entries[parent_idx]->status = RequestStateStatus::kFinished;
      // Remove the request state entry from all the models.
      RemoveRequestFromModel(estate, rstate->entries[parent_idx]->mstates[0]->internal_id, models);
      estate->id_manager.RecycleId(rstate->entries[parent_idx]->mstates[0]->internal_id);
      estate->stats.current_total_seq_len -=
          static_cast<int>(rstate->entries[parent_idx]->mstates[0]->committed_tokens.size());
      // Climb up to the parent.
      parent_idx = rstate->entries[parent_idx]->parent_idx;
    }

    if (parent_idx == -1) {
      // All request state entries of the request have been removed.
      // Reduce the total input length from the engine stats.
      estate->stats.current_total_seq_len -= rsentry->request->input_total_length;
      // Remove from running queue and engine state.
      auto it =
          std::find(estate->running_queue.begin(), estate->running_queue.end(), rsentry->request);
      ICHECK(it != estate->running_queue.end());
      estate->running_queue.erase(it);
      estate->request_states.erase(rsentry->request->id);

      // Update engine statistics.
      const RequestStateEntry& root_rsentry = rstate->entries[0];
      auto trequest_finish = std::chrono::high_resolution_clock::now();
      estate->stats.request_total_prefill_time +=
          static_cast<double>((root_rsentry->tprefill_finish - root_rsentry->tadd).count()) / 1e9;
      estate->stats.total_prefill_length += rsentry->request->input_total_length;
      estate->stats.request_total_decode_time +=
          static_cast<double>((trequest_finish - root_rsentry->tprefill_finish).count()) / 1e9;
      for (const RequestStateEntry& entry : rstate->entries) {
        estate->stats.total_decode_length += entry->mstates[0]->committed_tokens.size();
      }
      estate->stats.total_decode_length -= rsentry->request->generation_cfg->n;
    }
  }
}

void ActionStepPostProcess(Array<Request> requests, EngineState estate, Array<Model> models,
                           const Tokenizer& tokenizer,
                           FRequestStreamCallback request_stream_callback,
                           int max_single_sequence_length) {
  std::vector<RequestStateEntry> finished_rsentries;
  finished_rsentries.reserve(requests.size());

  Array<RequestStreamOutput> callback_delta_outputs;
  callback_delta_outputs.reserve(requests.size());

  // - Collect new generated tokens and finish reasons for requests.
  for (Request request : requests) {
    int n = request->generation_cfg->n;
    RequestState rstate = estate->GetRequestState(request);
    Array<IntTuple> group_delta_token_ids;
    Array<Array<String>> group_delta_logprob_json_strs;
    Array<Optional<String>> group_finish_reason;
    group_delta_token_ids.reserve(n);
    group_delta_logprob_json_strs.reserve(n);
    group_finish_reason.reserve(n);

    bool invoke_callback = false;
    for (int i = 0; i < n; ++i) {
      const RequestStateEntry& rsentry = n == 1 ? rstate->entries[0] : rstate->entries[i + 1];
      const DeltaRequestReturn& delta_request_ret =
          rsentry->GetReturnTokenIds(tokenizer, max_single_sequence_length);
      group_delta_token_ids.push_back(IntTuple{delta_request_ret.delta_token_ids.begin(),
                                               delta_request_ret.delta_token_ids.end()});
      group_delta_logprob_json_strs.push_back(delta_request_ret.delta_logprob_json_strs);
      group_finish_reason.push_back(delta_request_ret.finish_reason);
      if (delta_request_ret.finish_reason.defined()) {
        invoke_callback = true;
        finished_rsentries.push_back(rsentry);
      }

      if (!delta_request_ret.delta_token_ids.empty()) {
        invoke_callback = true;
      }
    }

    if (invoke_callback) {
      callback_delta_outputs.push_back(RequestStreamOutput(
          request->id, std::move(group_delta_token_ids),
          request->generation_cfg->logprobs > 0 ? std::move(group_delta_logprob_json_strs)
                                                : Optional<Array<Array<String>>>(),
          std::move(group_finish_reason)));
    }
  }

  // - Invoke the stream callback function once for all collected requests.
  request_stream_callback(callback_delta_outputs);

  ProcessFinishedRequestStateEntries(std::move(finished_rsentries), std::move(estate),
                                     std::move(models), max_single_sequence_length);
}

RequestStateEntry PreemptLastRunningRequestStateEntry(EngineState estate,
                                                      const Array<Model>& models,
                                                      Optional<EventTraceRecorder> trace_recorder) {
  ICHECK(!estate->running_queue.empty());
  Request request = estate->running_queue.back();

  // Find the last alive request state entry, which is what we want to preempt.
  RequestState rstate = estate->GetRequestState(request);
  int preempt_rstate_idx = -1;
  for (int i = static_cast<int>(rstate->entries.size()) - 1; i >= 0; --i) {
    if (rstate->entries[i]->status == RequestStateStatus::kAlive) {
      preempt_rstate_idx = i;
      break;
    }
  }
  ICHECK_NE(preempt_rstate_idx, -1);
  RequestStateEntry rsentry = rstate->entries[preempt_rstate_idx];

  // Remove from models.
  // - Clear model speculation draft.
  // - Update `inputs` for future prefill.
  RECORD_EVENT(trace_recorder, rsentry->request->id, "preempt");
  rsentry->status = RequestStateStatus::kPending;
  estate->stats.current_total_seq_len -= rsentry->mstates[0]->committed_tokens.size();
  if (rsentry->child_indices.empty()) {
    // The length was overly decreased by 1 when the entry has no child.
    ++estate->stats.current_total_seq_len;
  }
  if (rsentry->parent_idx == -1) {
    // Subtract the input length from the total length when the
    // current entry is the root entry of the request.
    estate->stats.current_total_seq_len -= request->input_total_length;
  }
  estate->stats.current_total_seq_len -=
      request->input_total_length + rsentry->mstates[0]->committed_tokens.size() - 1;
  for (RequestModelState mstate : rsentry->mstates) {
    mstate->RemoveAllDraftTokens();
    ICHECK(mstate->inputs.empty());
    std::vector<int32_t> committed_token_ids;
    committed_token_ids.reserve(mstate->committed_tokens.size());
    for (const SampleResult& committed_token : mstate->committed_tokens) {
      committed_token_ids.push_back(committed_token.sampled_token_id.first);
    }

    Array<Data> inputs;
    if (rsentry->parent_idx == -1) {
      inputs = request->inputs;
      if (const auto* token_input = inputs.back().as<TokenDataNode>()) {
        // Merge the TokenData so that a single time TokenEmbed is needed.
        std::vector<int> token_ids{token_input->token_ids->data,
                                   token_input->token_ids->data + token_input->token_ids.size()};
        token_ids.insert(token_ids.end(), committed_token_ids.begin(), committed_token_ids.end());
        inputs.Set(inputs.size() - 1, TokenData(token_ids));
      } else if (!committed_token_ids.empty()) {
        inputs.push_back(TokenData(committed_token_ids));
      }
    } else if (!committed_token_ids.empty()) {
      inputs.push_back(TokenData(committed_token_ids));
    }
    mstate->inputs = std::move(inputs);
  }
  RemoveRequestFromModel(estate, rsentry->mstates[0]->internal_id, models);

  if (preempt_rstate_idx == 0) {
    // Remove from running queue.
    estate->running_queue.erase(estate->running_queue.end() - 1);
  }
  if (preempt_rstate_idx == static_cast<int>(rstate->entries.size()) - 1) {
    // Add to the front of waiting queue.
    estate->waiting_queue.insert(estate->waiting_queue.begin(), request);
  }
  return rsentry;
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
