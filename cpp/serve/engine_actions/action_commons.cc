/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_actions/action_commons.cc
 */

#include "action_commons.h"

namespace mlc {
namespace llm {
namespace serve {

void RemoveRequestFromModel(EngineState estate, int req_id, Array<Model> models) {
  // Remove the request from all models (usually the KV cache).
  for (Model model : models) {
    model->RemoveSequence(req_id);
  }
  // Update the internal request id of other requests.
  for (auto& it : estate->request_states) {
    RequestState state = it.second;
    for (RequestModelState mstate : state->mstates) {
      ICHECK_NE(mstate->request_id, req_id);
      if (mstate->request_id > req_id) {
        --mstate->request_id;
      }
    }
  }
}

void ProcessFinishedRequest(Array<Request> finished_requests, EngineState estate,
                            Array<Model> models, const std::unique_ptr<Tokenizer>& tokenizer,
                            int max_single_sequence_length) {
  // - Remove the finished request.
  for (Request request : finished_requests) {
    // Remove from running queue.
    auto it = std::find(estate->running_queue.begin(), estate->running_queue.end(), request);
    ICHECK(it != estate->running_queue.end());
    estate->running_queue.erase(it);

    // Update engine states.
    RequestState state = estate->GetRequestState(request);
    int req_id = state->mstates[0]->request_id;
    for (RequestModelState mstate : state->mstates) {
      ICHECK_EQ(mstate->request_id, req_id);
      mstate->request_id = -1;
    }
    RemoveRequestFromModel(estate, req_id, models);
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
                           const std::unique_ptr<Tokenizer>& tokenizer,
                           int max_single_sequence_length) {
  Array<Request> finished_requests;
  finished_requests.reserve(requests.size());

  // - Invoke the function callback for requests with new generated tokens.
  for (Request request : requests) {
    RequestState rstate = estate->GetRequestState(request);
    Optional<String> finish_reason = rstate->GenerationFinishReason(max_single_sequence_length);
    int num_committed_tokens = rstate->mstates[0]->committed_tokens.size();

    // Check if there are new committed tokens.
    // If so, we invoke the callback function.
    // Otherwise, we do not invoke, and the request must have not been finished yet.
    ICHECK_LE(rstate->next_callback_token_pos, num_committed_tokens);
    if (rstate->next_callback_token_pos == num_committed_tokens) {
      ICHECK(!finish_reason.defined());
      continue;
    }

    request->fcallback(request->id,
                       TokenData(IntTuple(rstate->mstates[0]->committed_tokens.begin() +
                                              rstate->next_callback_token_pos,
                                          rstate->mstates[0]->committed_tokens.end())),
                       finish_reason);
    rstate->next_callback_token_pos = num_committed_tokens;

    if (finish_reason.defined()) {
      finished_requests.push_back(request);
    }
  }

  ProcessFinishedRequest(std::move(finished_requests), std::move(estate), std::move(models),
                         tokenizer, max_single_sequence_length);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
