/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/engine_actions/action.h
 * \brief The abstraction of actions (e.g., prefill/decode) that an
 * Engine can take at each time step.
 */
#ifndef MLC_LLM_SERVE_ENGINE_ACTIONS_ACTION_H_
#define MLC_LLM_SERVE_ENGINE_ACTIONS_ACTION_H_

#include "../config.h"
#include "../draft_token_workspace_manager.h"
#include "../engine.h"
#include "../engine_state.h"
#include "../event_trace_recorder.h"
#include "../model.h"
#include "../sampler/sampler.h"

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
   * \return The processed requests in this step.
   */
  virtual Array<Request> Step(EngineState estate) = 0;

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
   * \brief Create the action that prefills requests in the `waiting_queue`
   * of the engine state.
   * \param models The models to run prefill in.
   * \param logit_processor The logit processor.
   * \param sampler The sampler to sample new tokens.
   * \param model_workspaces The workspace of each model.
   * \param engine_config The engine config.
   * \param model_configs The config of each model.
   * \param trace_recorder The event trace recorder for requests.
   * \return The created action object.
   */
  static EngineAction NewRequestPrefill(Array<Model> models, LogitProcessor logit_processor,
                                        Sampler sampler,
                                        std::vector<ModelWorkspace> model_workspaces,
                                        EngineConfig engine_config,
                                        std::vector<picojson::object> model_configs,
                                        Optional<EventTraceRecorder> trace_recorder);
  /*!
   * \brief Create the action that prefills requests in the `waiting_queue`
   * of the engine state.
   * \param models The models to run prefill in.
   * \param logit_processor The logit processor.
   * \param sampler The sampler to sample new tokens.
   * \param model_workspaces The workspace of each model.
   * \param draft_token_workspace_manager The draft token workspace manager.
   * \param engine_config The engine config.
   * \param model_configs The config of each model.
   * \param trace_recorder The event trace recorder for requests.
   * \return The created action object.
   */
  static EngineAction EagleNewRequestPrefill(
      Array<Model> models, LogitProcessor logit_processor, Sampler sampler,
      std::vector<ModelWorkspace> model_workspaces,
      DraftTokenWorkspaceManager draft_token_workspace_manager, EngineConfig engine_config,
      std::vector<picojson::object> model_configs, Optional<EventTraceRecorder> trace_recorder);
  /*!
   * \brief Create the action that runs one-step decode for requests in the
   * `running_queue` of engine state. Preempt low-priority requests
   * accordingly when it is impossible to decode all the running requests.
   * \note The BatchDecode action **does not** take effect for speculative
   * decoding scenarios where there are multiple models. For speculative
   * decoding in the future, we will use other specific actions.
   * \param models The model to run decode in. When there are multiple
   * models, the `Step` function of the created action will not take effect.
   * \param tokenizer The tokenizer of the engine.
   * \param sampler The sampler to sample new tokens.
   * \param engine_config The engine config.
   * \param trace_recorder The event trace recorder for requests.
   * \return The created action object.
   */
  static EngineAction BatchDecode(Array<Model> models, Tokenizer tokenizer,
                                  LogitProcessor logit_processor, Sampler sampler,
                                  EngineConfig engine_config,
                                  Optional<EventTraceRecorder> trace_recorder);

  /*!
   * \brief Create the action that runs one-step speculative draft proposal for
   * requests in the `running_queue` of engine state. Preempt low-priority requests
   * accordingly when it is impossible to decode all the running requests.
   * \param models The model to run decode in. When there are multiple
   * models, the `Step` function of the created action will not take effect.
   * \param sampler The sampler to sample new tokens.
   * \param model_workspaces The workspace of each model.
   * \param draft_token_workspace_manager The draft token workspace manager.
   * \param engine_config The engine config.
   * \param trace_recorder The event trace recorder for requests.
   * \return The created action object.
   */
  static EngineAction BatchDraft(Array<Model> models, LogitProcessor logit_processor,
                                 Sampler sampler, std::vector<ModelWorkspace> model_workspaces,
                                 DraftTokenWorkspaceManager draft_token_workspace_manager,
                                 EngineConfig engine_config,
                                 Optional<EventTraceRecorder> trace_recorder);

  /*!
   * \brief Create the action that runs one-step speculative draft proposal for
   * requests in the `running_queue` of engine state. Preempt low-priority requests
   * accordingly when it is impossible to decode all the running requests.
   * \param models The model to run decode in. When there are multiple
   * models, the `Step` function of the created action will not take effect.
   * \param sampler The sampler to sample new tokens.
   * \param model_workspaces The workspace of each model.
   * \param draft_token_workspace_manager The draft token workspace manager.
   * \param engine_config The engine config.
   * \param trace_recorder The event trace recorder for requests.
   * \return The created action object.
   */
  static EngineAction EagleBatchDraft(Array<Model> models, LogitProcessor logit_processor,
                                      Sampler sampler, std::vector<ModelWorkspace> model_workspaces,
                                      DraftTokenWorkspaceManager draft_token_workspace_manager,
                                      EngineConfig engine_config,
                                      Optional<EventTraceRecorder> trace_recorder);

  /*!
   * \brief Create the action that runs one-step speculative verification for requests in the
   * `running_queue` of engine state. Preempt low-priority requests
   * accordingly when it is impossible to decode all the running requests.
   * \param models The model to run decode in. When there are multiple
   * models, the `Step` function of the created action will not take effect.
   * \param model_workspaces The workspace of each model.
   * \param draft_token_workspace_manager The draft token workspace manager.
   * \param sampler The sampler to sample new tokens.
   * \param engine_config The engine config.
   * \param trace_recorder The event trace recorder for requests.
   * \return The created action object.
   */
  static EngineAction BatchVerify(Array<Model> models, LogitProcessor logit_processor,
                                  Sampler sampler, std::vector<ModelWorkspace> model_workspaces,
                                  DraftTokenWorkspaceManager draft_token_workspace_manager,
                                  EngineConfig engine_config,
                                  Optional<EventTraceRecorder> trace_recorder);

  /*!
   * \brief Create the action that runs one-step speculative verification for requests in the
   * `running_queue` of engine state. Preempt low-priority requests
   * accordingly when it is impossible to decode all the running requests.
   * \param models The model to run decode in. When there are multiple
   * models, the `Step` function of the created action will not take effect.
   * \param sampler The sampler to sample new tokens.
   * \param model_workspaces The workspace of each model.
   * \param draft_token_workspace_manager The draft token workspace manager.
   * \param engine_config The engine config.
   * \param trace_recorder The event trace recorder for requests.
   * \return The created action object.
   */
  static EngineAction EagleBatchVerify(Array<Model> models, LogitProcessor logit_processor,
                                       Sampler sampler,
                                       std::vector<ModelWorkspace> model_workspaces,
                                       DraftTokenWorkspaceManager draft_token_workspace_manager,
                                       EngineConfig engine_config,
                                       Optional<EventTraceRecorder> trace_recorder);
  /*!
   * \brief Create the action that executes the jump-forward decoding to predict the next tokens
   * according to the grammar constraint. Does nothing for the requests without grammar. The
   * predicted tokens will be fed to the next BatchDecode action. Retokenization may happen when
   * the predicted string breaks the tokenization boundary.
   * \param models The model to run decode in. When there are multiple
   * models, the `Step` function of the created action will not take effect.
   * \param tokenizer The tokenizer of the engine.
   * \param trace_recorder The event trace recorder for requests.
   * \return The created action object.
   */
  static EngineAction BatchJumpForward(Array<Model> models, Tokenizer tokenizer,
                                       Optional<EventTraceRecorder> trace_recorder);

  /*!
   * \brief Create the action that first makes a decision on whether to run speculative
   * decoding or normal mode batch decode, and then runs the selected actions.
   * \param spec_decode_actions The actions for speculative decoding.
   * \param batch_decode_actions The actions for normal mode batch decoding.
   * \param engine_config The engine config.
   * \return The created action object
   */
  static EngineAction AutoSpecDecode(std::vector<EngineAction> spec_decode_actions,
                                     std::vector<EngineAction> batch_decode_actions,
                                     EngineConfig engine_config);

  /*!
   * \brief Create the action that runs the disaggregation preparation for prefill.
   * \param models The underlying models whose KV cache are to be updated.
   * \param engine_config The engine config.
   * \param model_configs The config of each model.
   * \param trace_recorder The event trace recorder for requests.
   * \param request_stream_callback The stream callback function to pass the prefill
   * preparation result back, including the KV cache append metadata and the prefix
   * matched length in the prefix cache.
   * \return The created action object.
   */
  static EngineAction DisaggPrepareReceive(Array<Model> models, EngineConfig engine_config,
                                           std::vector<picojson::object> model_configs,
                                           Optional<EventTraceRecorder> trace_recorder,
                                           FRequestStreamCallback request_stream_callback);

  /*!
   * \brief Create the action that runs the prefill and sends KV data to remote instance.
   * \param models The underlying models whose KV cache are to be updated.
   * \param model_workspaces The workspace of each model.
   * \param engine_config The engine config.
   * \param model_configs The config of each model.
   * \param trace_recorder The event trace recorder for requests.
   * \param request_stream_callback The stream callback function to pass the prefill
   * preparation result back, including the KV cache append metadata and the prefix
   * matched length in the prefix cache.
   * \param device The device of the model for synchronization.
   * \return The created action object.
   */
  static EngineAction DisaggRemoteSend(
      Array<Model> models, std::vector<ModelWorkspace> model_workspaces, EngineConfig engine_config,
      std::vector<picojson::object> model_configs, Optional<EventTraceRecorder> trace_recorder,
      FRequestStreamCallback request_stream_callback, Device device);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(EngineAction, ObjectRef, EngineActionObj);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_ENGINE_ACTIONS_ACTION_H_
