/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_actions/action_commons.h
 * \brief Common functions that may be used in multiple EngineActions.
 */
#ifndef MLC_LLM_SERVE_ENGINE_ACTIONS_ACTION_COMMONS_H_
#define MLC_LLM_SERVE_ENGINE_ACTIONS_ACTION_COMMONS_H_

#include "../../tokenizers.h"
#include "../engine.h"
#include "../engine_state.h"
#include "../event_trace_recorder.h"
#include "../model.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/*!
 * \brief Remove the given request from models.
 * \param estate The engine state to update after removal.
 * \param req_internal_id The internal id of the request to remove.
 * \param models The models to remove the given request from.
 */
void RemoveRequestFromModel(EngineState estate, int64_t req_internal_id, Array<Model> models);

/*!
 * \brief The request post-processing after an engine action step.
 * It includes
 * - invoke the request function callback to return new generated tokens,
 * - update the engine state for finished requests.
 * \note This function may remove requests from the `running_queue`.
 * \param requests The requests to process.
 * \param estate The engine state.
 * \param models The models to remove the finished from.
 * \param tokenizer The tokenizer for logprob process.
 * \param request_stream_callback The request stream callback function.
 * \param max_single_sequence_length The max single sequence length to help decide
 * if a request is finished.
 */
void ActionStepPostProcess(Array<Request> requests, EngineState estate, Array<Model> models,
                           const Tokenizer& tokenizer,
                           FRequestStreamCallback request_stream_callback,
                           int max_single_sequence_length);

/*!
 * \brief Preempt the last running request state entry from `running_queue`.
 * If all entries of the the selected request have been preempted,
 * remove it from running request.
 * If it is not in the waiting request queue, add it to the waiting queue.
 * \param estate The engine state to update due to preemption.
 * \param models The models to remove preempted requests from.
 * \param trace_recorder The event trace recorder for requests.
 * \return The preempted request state.
 */
RequestStateEntry PreemptLastRunningRequestStateEntry(EngineState estate,
                                                      const Array<Model>& models,
                                                      Optional<EventTraceRecorder> trace_recorder);

/*! \brief Get the running request entries from the engine state. */
inline std::vector<RequestStateEntry> GetRunningRequestStateEntries(const EngineState& estate) {
  std::vector<RequestStateEntry> rsentries;
  for (const Request& request : estate->running_queue) {
    for (const RequestStateEntry& rsentry : estate->GetRequestState(request)->entries) {
      if (rsentry->status == RequestStateStatus::kAlive && rsentry->child_indices.empty()) {
        rsentries.push_back(rsentry);
      }
    }
  }
  return rsentries;
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_ENGINE_ACTIONS_ACTION_COMMONS_H_
