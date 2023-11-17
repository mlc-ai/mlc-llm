/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_actions/action.h
 * \brief The abstraction of actions (e.g., prefill/decode) that an
 * Engine can take at each time step.
 */
#ifndef MLC_LLM_SERVE_ENGINE_ACTIONS_ACTION_H_
#define MLC_LLM_SERVE_ENGINE_ACTIONS_ACTION_H_

#include "../config.h"
#include "../engine_state.h"
#include "../model.h"
#include "../sampler.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/*!
 * \brief The abstraction of actions that an Engine can take at each time step.
 * The only core interface of an action is the `Step` function.
 * At high level, the Step function takes the current engine state
 * as input, invokes model functions (such as batched-prefill or
 * batched-decode), run sampler to sample new tokens, and update
 * the engine state.
 */
class EngineActionObj : public Object {
 public:
  /*!
   * \brief The behavior of the engine action in a single step.
   * \param estate The engine state to be analyzed and updated.
   * \return A boolean indicating if the action is successfully taken.
   */
  virtual bool Step(EngineState estate) = 0;

  static constexpr const char* _type_key = "mlc.serve.EngineAction";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(EngineActionObj, Object);
};

/*!
 * \brief Managed reference of EngineActionObj.
 * It declares the full list of supported actions.
 * \sa EngineActionObj
 */
class EngineAction : public ObjectRef {
 public:
  /*!
   * \brief Create the action that aborts the requests in the `abort_queue`
   * of the engine state.
   * \param models The models where the requests to abort also need
   * to be removed from.
   * \return The created action object.
   */
  static EngineAction AbortRequest(Array<Model> models);
  /*!
   * \brief Create the action that prefills requests in the `waiting_queue`
   * of the engine state.
   * \param models The models to run prefill in.
   * \param sampler The sampler to sample new tokens.
   * \param kv_cache_config The KV cache config to help decide prefill is doable.
   * \param max_single_sequence_length The max single sequence length to help
   * decide if prefill is doable.
   * \return The created action object.
   */
  static EngineAction NewRequestPrefill(Array<Model> models, Sampler sampler,
                                        KVCacheConfig kv_cache_config,
                                        int max_single_sequence_length);
  /*!
   * \brief Create the action that runs one-step decode for requests in the
   * `running_queue` of engine state. Preempt low-priority requests
   * accordingly when it is impossible to decode all the running requests.
   * \note The BatchDecode action **does not** take effect for speculative
   * decoding scenarios where there are multiple models. For speculative
   * decoding in the future, we will use other specific actions.
   * \param models The model to run decode in. When there are multiple
   * models, the `Step` function of the created action will not take effect.
   * \param sampler The sampler to sample new tokens.
   * \return The created action object.
   */
  static EngineAction BatchDecode(Array<Model> models, Sampler sampler);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(EngineAction, ObjectRef, EngineActionObj);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_ENGINE_ACTIONS_ACTION_H_
