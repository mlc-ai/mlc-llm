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

void ProcessFinishedRequest(Array<Request> finished_requests, EngineState estate,
                            Array<Model> models, int max_single_sequence_length) {
  // - Remove the finished request.
  for (Request request : finished_requests) {
    // Remove from running queue.
    auto it = std::find(estate->running_queue.begin(), estate->running_queue.end(), request);
    ICHECK(it != estate->running_queue.end());
    estate->running_queue.erase(it);

    // Update engine states.
    RequestState state = estate->GetRequestState(request);
    RemoveRequestFromModel(estate, state->mstates[0]->internal_id, models);
    estate->id_manager.RecycleId(state->mstates[0]->internal_id);
    estate->request_states.erase(request->id);

    // Update engine statistics.
    int num_input_tokens = request->input_total_length;
    int num_output_tokens = state->mstates[0]->committed_tokens.size() - 1;
    estate->stats.current_total_seq_len -= num_input_tokens + num_output_tokens;
    auto trequest_finish = std::chrono::high_resolution_clock::now();
    estate->stats.request_total_prefill_time +=
        static_cast<double>((state->tprefill_finish - state->tadd).count()) / 1e9;
    estate->stats.total_prefill_length += num_input_tokens;
    estate->stats.request_total_decode_time +=
        static_cast<double>((trequest_finish - state->tprefill_finish).count()) / 1e9;
    estate->stats.total_decode_length += num_output_tokens;
  }
}

void ActionStepPostProcess(Array<Request> requests, EngineState estate, Array<Model> models,
                           FRequestStreamCallback request_stream_callback,
                           int max_single_sequence_length) {
  Array<Request> finished_requests;
  finished_requests.reserve(requests.size());

  Array<RequestStreamOutput> callback_delta_outputs;
  callback_delta_outputs.reserve(requests.size());

  // - Collect new generated tokens and finish reasons for requests.
  for (Request request : requests) {
    RequestState rstate = estate->GetRequestState(request);
    Optional<String> finish_reason = rstate->GenerationFinished(max_single_sequence_length);
    int num_committed_tokens = rstate->mstates[0]->committed_tokens.size();

    // Check if there are new committed tokens.
    // If so, we will invoke the callback function for it.
    ICHECK_LE(rstate->next_callback_token_pos, num_committed_tokens);
    if (rstate->next_callback_token_pos == num_committed_tokens && !finish_reason.defined()) {
      continue;
    }

    callback_delta_outputs.push_back(RequestStreamOutput(
        request->id,
        TokenData(
            IntTuple(rstate->mstates[0]->committed_tokens.begin() + rstate->next_callback_token_pos,
                     rstate->mstates[0]->committed_tokens.end())),
        finish_reason));
    rstate->next_callback_token_pos = num_committed_tokens;

    if (finish_reason.defined()) {
      finished_requests.push_back(request);
    }
  }

  // - Invoke the stream callback function once for all collected requests.
  request_stream_callback(callback_delta_outputs);

  ProcessFinishedRequest(std::move(finished_requests), std::move(estate), std::move(models),
                         max_single_sequence_length);
}

void PreemptLastRunningRequest(EngineState estate, const Array<Model>& models,
                               Optional<EventTraceRecorder> trace_recorder) {
  Request request = estate->running_queue.back();

  // Remove from models.
  // - Clear model speculation draft.
  // - Update `inputs` for future prefill.
  RequestState rstate = estate->GetRequestState(request);
  RECORD_EVENT(trace_recorder, rstate->request->id, "preempt");
  estate->stats.current_total_seq_len -=
      request->input_total_length + rstate->mstates[0]->committed_tokens.size() - 1;
  for (RequestModelState mstate : rstate->mstates) {
    mstate->RemoveAllDraftTokens();
    mstate->draft_output_token_prob.clear();
    mstate->draft_output_prob_dist.clear();
    ICHECK(mstate->inputs.empty());
    ICHECK(!mstate->committed_tokens.empty());

    Array<Data> inputs = request->inputs;
    if (const auto* token_input = inputs.back().as<TokenDataNode>()) {
      // Merge the TokenData so that a single time TokenEmbed is needed.
      std::vector<int> token_ids{token_input->token_ids->data,
                                 token_input->token_ids->data + token_input->token_ids.size()};
      token_ids.insert(token_ids.end(), mstate->committed_tokens.begin(),
                       mstate->committed_tokens.end());
      inputs.Set(inputs.size() - 1, TokenData(token_ids));
    } else {
      inputs.push_back(TokenData(mstate->committed_tokens));
    }
    mstate->inputs = std::move(inputs);
  }
  RemoveRequestFromModel(estate, rstate->mstates[0]->internal_id, models);

  // Move from running queue to the front of waiting queue.
  estate->running_queue.erase(estate->running_queue.end() - 1);
  estate->waiting_queue.insert(estate->waiting_queue.begin(), request);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
