/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_actions/action_commons.h
 * \brief Common functions that may be used in multiple EngineActions.
 */
#ifndef MLC_LLM_SERVE_ENGINE_ACTIONS_ACTION_COMMONS_H_
#define MLC_LLM_SERVE_ENGINE_ACTIONS_ACTION_COMMONS_H_

#include "../../tokenizers/tokenizers.h"
#include "../draft_token_workspace_manager.h"
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
 * \param trace_recorder The event trace recorder for requests.
 * if a request is finished.
 */
void ActionStepPostProcess(Array<Request> requests, EngineState estate, Array<Model> models,
                           const Tokenizer& tokenizer,
                           FRequestStreamCallback request_stream_callback,
                           int64_t max_single_sequence_length,
                           Optional<EventTraceRecorder> trace_recorder);

/*!
 * \brief Preempt the last running request state entry from `running_queue`.
 * If all entries of the selected request have been preempted,
 * remove it from running request.
 * If it is not in the waiting request queue, add it to the waiting queue.
 * \param estate The engine state to update due to preemption.
 * \param models The models to remove preempted requests from.
 * \param draft_token_workspace_manager The draft token workspace manager for requests. Must be
 * provided if speculative decoding is enabled.
 * \param trace_recorder The event trace recorder for requests.
 * \return The preempted request state.
 */
RequestStateEntry PreemptLastRunningRequestStateEntry(
    EngineState estate, const Array<Model>& models,
    Optional<DraftTokenWorkspaceManager> draft_token_workspace_manager,
    Optional<EventTraceRecorder> trace_recorder);

/*! \brief Get the running request entries from the engine state. */
inline std::vector<RequestStateEntry> GetRunningRequestStateEntries(const EngineState& estate) {
  std::vector<RequestStateEntry> rsentries;
  for (const Request& request : estate->running_queue) {
    for (const RequestStateEntry& rsentry : estate->GetRequestState(request)->entries) {
      // One request entry is considered as running for decode if it is a leaf and has
      // finished all input prefill.
      if (rsentry->status == RequestStateStatus::kAlive && rsentry->child_indices.empty() &&
          rsentry->mstates[0]->inputs.empty()) {
        rsentries.push_back(rsentry);
      }
    }
  }
  return rsentries;
}

/*!
 * \brief Apply the logit processor to the logits and sample one token for each request.
 * \param logit_processor The logit processor to apply.
 * \param sampler The sampler to sample tokens.
 * \param logits The logits to process.
 * \param generation_cfg The generation configurations of the requests.
 * \param request_ids The request ids.
 * \param mstates The model states of the requests.
 * \param rngs The random generators of the requests.
 * \param sample_indices The indices of the requests to sample.
 * \return The processed logits and the sampled results.
 */
std::pair<NDArray, std::vector<SampleResult>> ApplyLogitProcessorAndSample(
    const LogitProcessor& logit_processor, const Sampler& sampler, const NDArray& logits,
    const Array<GenerationConfig>& generation_cfg, const Array<String>& request_ids,
    const Array<RequestModelState>& mstates, const std::vector<RandomGenerator*>& rngs,
    const std::vector<int>& sample_indices);

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_ENGINE_ACTIONS_ACTION_COMMONS_H_
