/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_actions/abort_requests.cc
 */

#include "../config.h"
#include "../model.h"
#include "../sampler.h"
#include "action.h"
#include "action_commons.h"

namespace mlc {
namespace llm {
namespace serve {

/*!
 * \brief The action that aborts the requests in the `abort_queue`
 * of the engine state.
 */
class AbortRequestActionObj : public EngineActionObj {
 public:
  explicit AbortRequestActionObj(Array<Model> models) : models_(std::move(models)) {}

  bool Step(EngineState estate) final {
    // Abort all requests in the abort queue.
    while (!estate->abort_queue.empty()) {
      // - Check if the request is running or pending.
      Request request = estate->abort_queue.front();
      auto it_running =
          std::find(estate->running_queue.begin(), estate->running_queue.end(), request);
      auto it_waiting =
          std::find(estate->waiting_queue.begin(), estate->waiting_queue.end(), request);
      ICHECK(it_running != estate->running_queue.end() ||
             it_waiting != estate->waiting_queue.end());

      if (it_running != estate->running_queue.end()) {
        // The request to abort is in running queue
        int req_id = it_running - estate->running_queue.begin();
        estate->running_queue.erase(it_running);
        RequestState state = estate->GetRequestState(request);
        estate->stats.current_total_seq_len -=
            request->input_total_length + state->mstates[0]->committed_tokens.size() - 1;
        RemoveRequestFromModel(estate, req_id, models_);
      } else {
        // The request to abort is in waiting queue
        estate->waiting_queue.erase(it_waiting);
      }
      estate->request_states.erase(request->id);
    }
    return true;
  }

 private:
  /*! \brief The models where the requests to abort also need to be removed from. */
  Array<Model> models_;
};

EngineAction EngineAction::AbortRequest(Array<Model> models) {
  return EngineAction(make_object<AbortRequestActionObj>(std::move(models)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
