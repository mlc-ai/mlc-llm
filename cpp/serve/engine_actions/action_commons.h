/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/engine_actions/action_commons.h
 * \brief Common functions that may be used in multiple EngineActions.
 */
#ifndef MLC_LLM_SERVE_ENGINE_ACTIONS_ACTION_COMMONS_H_
#define MLC_LLM_SERVE_ENGINE_ACTIONS_ACTION_COMMONS_H_

#include <tvm/ffi/container/array.h>

#include "../../tokenizers/tokenizers.h"
#include "../draft_token_workspace_manager.h"
#include "../engine.h"
#include "../engine_state.h"
#include "../event_trace_recorder.h"
#include "../model.h"
#include "action.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/*! \brief Create the engine actions based on engine config. */
Array<EngineAction> CreateEngineActions(
    Array<Model> models, EngineConfig engine_config, std::vector<picojson::object> model_configs,
    std::vector<ModelWorkspace> model_workspaces, LogitProcessor logit_processor, Sampler sampler,
    DraftTokenWorkspaceManager draft_token_workspace_manager, Tokenizer tokenizer,
    Optional<EventTraceRecorder> trace_recorder, FRequestStreamCallback request_stream_callback,
    Device device);

/*!
 * \brief Remove the given request from models.
 * \param estate The engine state to update after removal.
 * \param req_internal_id The internal id of the request to remove.
 * \param models The models to remove the given request from.
 */
void RemoveRequestFromModel(EngineState estate, int64_t req_internal_id,
                            const Array<Model>& models);

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
 * \param draft_token_workspace_manager The draft token workspace manager.
 * \param trace_recorder The event trace recorder for requests.
 * if a request is finished.
 */
void ActionStepPostProcess(Array<Request> requests, EngineState estate, const Array<Model>& models,
                           const Tokenizer& tokenizer,
                           FRequestStreamCallback request_stream_callback,
                           int64_t max_single_sequence_length,
                           Optional<DraftTokenWorkspaceManager> draft_token_workspace_manager,
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

/*!
 * \brief Apply the logit processor to the logits and sample one token for each request.
 *
 * Both the parent request configurations and the child request configurations need to be provided.
 * The parent request configurations are used to process the logits, normalize the probabilities.
 * The child request configurations are used to sample the tokens.
 *
 * When the request doesn't have children, the parent and child configurations are the same.
 *
 * \param logit_processor The logit processor to apply.
 * \param sampler The sampler to sample tokens.
 * \param logits The logits to process.
 * \param generation_cfg The generation configurations of the requests.
 * \param request_ids The request ids.
 * \param mstates The model states of the requests.
 * \param rngs The random generators of the requests.
 * \param sample_indices The indices of the requests to sample.
 * \param child_generation_cfg The generation configurations of the child requests.
 * \param child_request_ids The request ids of the child requests.
 * \param child_sample_indices The indices of the child requests to sample.
 * \return The processed logits and the sampled results.
 */
std::pair<NDArray, std::vector<SampleResult>> ApplyLogitProcessorAndSample(
    const LogitProcessor& logit_processor, const Sampler& sampler, const NDArray& logits,
    const Array<GenerationConfig>& generation_cfg, const Array<String>& request_ids,
    const Array<RequestModelState>& mstates, const std::vector<RandomGenerator*>& rngs,
    const std::vector<int>& sample_indices, const Array<GenerationConfig>& child_generation_cfg,
    const Array<String>& child_request_ids, const std::vector<int>& child_sample_indices);

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_ENGINE_ACTIONS_ACTION_COMMONS_H_
