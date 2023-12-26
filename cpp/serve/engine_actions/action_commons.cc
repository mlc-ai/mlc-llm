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

}  // namespace serve
}  // namespace llm
}  // namespace mlc
