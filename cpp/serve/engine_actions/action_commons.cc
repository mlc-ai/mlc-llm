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

void ProcessFinishedRequest(EngineState estate, Array<Model> models,
                            const std::unique_ptr<Tokenizer>& tokenizer,
                            int max_single_sequence_length) {
  // - Collect finished requests.
  //   We don't remove on the fly to avoid concurrent modification.
  std::vector<Request> request_to_remove;
  for (Request request : estate->running_queue) {
    if (estate->GetRequestState(request)->GenerationFinished(max_single_sequence_length)) {
      request_to_remove.push_back(request);
    }
  }

  // - Remove the finished request.
  for (Request request : request_to_remove) {
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

    // NOTE: right now we only return the generated text.
    // In the future we might optional return text or token ids.
    String output = tokenizer->Decode(state->mstates[0]->committed_tokens);
    request->fcallback(request, TextData(output));
  }
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
