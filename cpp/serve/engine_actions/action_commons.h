/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_actions/action_commons.h
 * \brief Common functions that may be used in multiple EngineActions.
 */
#ifndef MLC_LLM_SERVE_ENGINE_ACTIONS_ACTION_COMMONS_H_
#define MLC_LLM_SERVE_ENGINE_ACTIONS_ACTION_COMMONS_H_

#include "../../tokenizers.h"
#include "../engine_state.h"
#include "../model.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/*!
 * \brief Remove the given request from models.
 * \param estate The engine state to update after removal.
 * \param req_id The internal id of the request to remove.
 * \param models The models to remove the given request from.
 */
void RemoveRequestFromModel(EngineState estate, int req_id, Array<Model> models);

/*!
 * \brief For each request in the `running_queue` of the engine state,
 * check if the request has finished its generation. Update the state
 * and return the generation result via request callback for the finished
 * requests.
 * \note This function removes requests from the `running_queue`.
 * \param estate The engine state.
 * \param models The models to remove the finished from.
 * \param tokenizer The tokenizer used to decode the generated tokens of requests.
 * \param max_single_sequence_length The max single sequence length to help decide
 * if a request is finished.
 */
void ProcessFinishedRequest(EngineState estate, Array<Model> models,
                            const std::unique_ptr<Tokenizer>& tokenizer,
                            int max_single_sequence_length);

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_ENGINE_ACTIONS_ACTION_COMMONS_H_
