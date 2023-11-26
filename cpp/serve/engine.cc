/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine.cc
 * \brief The implementation for runtime module of serving engine module in MLC LLM.
 */
#include "engine.h"

#include <dlpack/dlpack.h>
#include <tokenizers_cpp.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <tuple>

#include "../tokenizers.h"
#include "engine_actions/action.h"
#include "engine_actions/action_commons.h"
#include "engine_state.h"
#include "model.h"
#include "request.h"
#include "request_state.h"
#include "sampler.h"

namespace mlc {
namespace llm {
namespace serve {

using tvm::Device;
using namespace tvm::runtime;

class EngineModule;

/*!
 * \brief The engine for request serving in MLC LLM.
 * The engine can run one or multiple LLM models internally for
 * text generation. Usually, when there are multiple models,
 * speculative inference will be activated, where the first model
 * (index 0) is the main "large model" that has better generation
 * quality, and all other models are "small" models that used for
 * speculation.
 * The engine receives requests from the "AddRequest" method. For
 * an given request, the engine will keep generating new tokens for
 * the request until finish (under certain criterion). After finish,
 * the engine will return the generation result through the callback
 * function provided by the request.
 * \note For now only one model run in the engine is supported.
 * Multiple model support such as speculative inference will
 * be followed soon in the future.
 *
 * The public interface of Engine has the following three categories:
 * - engine management,
 * - high-level request management,
 * - engine "step" action.
 */
class Engine {
  friend class EngineModule;

 public:
  /********************** Engine Management **********************/

  explicit Engine(int max_single_sequence_length, String tokenizer_path,
                  String kv_cache_config_json_str, Optional<PackedFunc> f_request_callback,
                  std::vector<std::tuple<TVMArgValue, String, DLDevice>> model_infos) {
    CHECK_GE(model_infos.size(), 1) << "ValueError: No model is provided in the engine.";
    // Step 1. Initialize metadata and singleton states inside the engine
    this->estate_->Reset();
    this->max_single_sequence_length_ = max_single_sequence_length;
    this->kv_cache_config_ = KVCacheConfig(kv_cache_config_json_str, max_single_sequence_length);
    this->f_request_callback_ = std::move(f_request_callback);
    this->sampler_ = Sampler::Create(/*sampler_kind=*/"cpu");
    this->tokenizer_ = TokenizerFromPath(tokenizer_path);
    // Step 2. Initialize each model independently.
    this->models_.clear();
    for (const auto& model_info : model_infos) {
      TVMArgValue model_lib = std::get<0>(model_info);
      String model_path = std::get<1>(model_info);
      DLDevice device = std::get<2>(model_info);
      Model model = Model::Create(model_lib, model_path, device);
      model->CreateKVCache(this->kv_cache_config_);
      CHECK_GE(model->GetMaxWindowSize(), this->max_single_sequence_length_)
          << "The window size of the model, " << model->GetMaxWindowSize()
          << ", is smaller than the pre-defined max single sequence length, "
          << this->max_single_sequence_length_;
      this->models_.push_back(model);
    }
    // Step 3. Initialize engine actions that represent state transitions.
    this->actions_ = {EngineAction::NewRequestPrefill(this->models_,           //
                                                      this->sampler_,          //
                                                      this->kv_cache_config_,  //
                                                      this->max_single_sequence_length_),
                      EngineAction::BatchDecode(this->models_, this->sampler_)};
  }

  /*! \brief Reset the engine, clean up all running data and statistics. */
  void ResetEngine() {
    estate_->Reset();
    for (Model model : models_) {
      model->Reset();
    }
  }

  /***************** High-level Request Management *****************/

  /*!
   * \brief Add a new request to the engine.
   * \param request The request to add.
   */
  void AddRequest(Request request) {
    // Get a request copy where all text inputs are tokenized.
    request = Request::FromUntokenized(request, tokenizer_);
    ICHECK_NE(request->input_total_length, -1);
    // Append to the waiting queue and create the request state.
    estate_->waiting_queue.push_back(request);
    estate_->request_states.emplace(request->id, RequestState(request, models_.size()));
  }

  /*! \brief Abort the input request. */
  void AbortRequest(String request_id) {
    auto it_rstate = estate_->request_states.find(request_id);
    if (it_rstate == estate_->request_states.end()) {
      // The request to abort does not exist.
      return;
    }

    RequestState rstate = it_rstate->second;
    Request request = rstate->request;

    // - Check if the request is running or pending.
    auto it_running =
        std::find(estate_->running_queue.begin(), estate_->running_queue.end(), request);
    auto it_waiting =
        std::find(estate_->waiting_queue.begin(), estate_->waiting_queue.end(), request);
    ICHECK(it_running != estate_->running_queue.end() ||
           it_waiting != estate_->waiting_queue.end());

    estate_->request_states.erase(request->id);
    if (it_running != estate_->running_queue.end()) {
      // The request to abort is in running queue
      int internal_req_id = it_running - estate_->running_queue.begin();
      estate_->running_queue.erase(it_running);
      estate_->stats.current_total_seq_len -=
          request->input_total_length + rstate->mstates[0]->committed_tokens.size() - 1;
      RemoveRequestFromModel(estate_, internal_req_id, models_);
    } else {
      // The request to abort is in waiting queue
      estate_->waiting_queue.erase(it_waiting);
    }
  }

  /*********************** Engine Action ***********************/

  /*!
   * \brief The main function that the engine takes a step of action.
   * At each step, the engine may decide to
   * - run prefill for one (or more) requests,
   * - run one-step decode for the all existing requests
   * ...
   * In the end of certain actions (e.g., decode), the engine will
   * check if any request has finished, and will return the
   * generation results for those finished requests.
   */
  void Step() {
    CHECK(f_request_callback_.defined())
        << "The request callback is not set. Engine cannot execute.";
    for (EngineAction action : actions_) {
      Array<Request> processed_requests = action->Step(estate_);
      if (!processed_requests.empty()) {
        ActionStepPostProcess(processed_requests, estate_, models_, f_request_callback_.value(),
                              max_single_sequence_length_);
        return;
      }
    }
    ICHECK(estate_->running_queue.empty())
        << "Internal assumption violated: It is expected that an engine step takes at least one "
           "action (e.g. prefill, decode, etc.) but it does not.";
  }

 private:
  // Engine state, managing requests and request states.
  EngineState estate_;
  // Configurations and singletons
  KVCacheConfig kv_cache_config_;
  int max_single_sequence_length_;
  Sampler sampler_;
  std::unique_ptr<Tokenizer> tokenizer_;
  // Models
  Array<Model> models_;
  // Request callback function
  Optional<PackedFunc> f_request_callback_;
  // Engine actions.
  Array<EngineAction> actions_;
};

/*! Clear global memory manager */
void ClearGlobalMemoryManager() {
  static const char* kFunc = "vm.builtin.memory_manager.clear";
  const PackedFunc* f = tvm::runtime::Registry::Get(kFunc);
  CHECK(f != nullptr) << "ValueError: Cannot find function `" << kFunc << "` in TVM runtime";
  (*f)();
}

class EngineModule : public ModuleNode {
 public:
  TVM_MODULE_VTABLE_BEGIN("mlc.serve.engine");
  TVM_MODULE_VTABLE_ENTRY_PACKED("init", &EngineModule::InitPacked);
  TVM_MODULE_VTABLE_ENTRY("add_request", &EngineModule::AddRequest);
  TVM_MODULE_VTABLE_ENTRY("abort", &EngineModule::Abort);
  TVM_MODULE_VTABLE_ENTRY("step", &EngineModule::Step);
  TVM_MODULE_VTABLE_ENTRY("stats", &EngineModule::Stats);
  TVM_MODULE_VTABLE_ENTRY("reset", &EngineModule::Reset);
  TVM_MODULE_VTABLE_ENTRY("get_request_callback", &EngineModule::GetRequestCallback);
  TVM_MODULE_VTABLE_ENTRY("set_request_callback", &EngineModule::SetRequestCallback);
  TVM_MODULE_VTABLE_END();

  void InitPacked(TVMArgs args, TVMRetValue* rv) {
    static const char* kErrorMessage =
        "With `n` models, engine initialization "
        "takes (4 + 4 * n) arguments. The first 4 arguments should be: "
        "1) (int) maximum length of a sequence, which must be equal or smaller than the context "
        "window size of each model; "
        "2) (string) path to tokenizer configuration files, which in MLC LLM, usually in a model "
        "weights directory; "
        "3) (string) JSON configuration for the KVCache; "
        "4) (packed function, optional) global request callback function. "
        "The following (4 * n) arguments, 4 for each model, should be: "
        "1) (tvm.runtime.Module) The model library loaded into TVM's RelaxVM; "
        "2) (string) Model path which includes weights and mlc-chat-config.json; "
        "3) (int, enum DLDeviceType) Device type, e.g. CUDA, ROCm, etc; "
        "4) (int) Device id, i.e. the ordinal index of the device that exists locally.";
    int num_models = (args.size() - 4) / 4;
    int max_single_sequence_length;
    std::string tokenizer_path;
    std::string kv_cache_config_json_str;
    Optional<PackedFunc> f_request_callback;
    std::vector<std::tuple<TVMArgValue, String, DLDevice>> model_infos;
    model_infos.reserve(num_models);
    try {
      CHECK_EQ(num_models * 4 + 4, args.size()) << "Incorrect number of arguments.";
      max_single_sequence_length = args.At<int>(0);
      tokenizer_path = args.At<std::string>(1);
      kv_cache_config_json_str = args.At<std::string>(2);
      f_request_callback = args.At<Optional<PackedFunc>>(3);
      for (int i = 0; i < num_models; ++i) {
        TVMArgValue model_lib = args[i * 4 + 4];
        std::string model_path = args.At<std::string>(i * 4 + 5);
        DLDeviceType device_type = static_cast<DLDeviceType>(args.At<int>(i * 4 + 6));
        int device_id = args.At<int>(i * 4 + 7);
        model_infos.emplace_back(model_lib, model_path, DLDevice{device_type, device_id});
      }
    } catch (const dmlc::Error& e) {
      LOG(FATAL) << "ValueError: " << e.what() << kErrorMessage;
    }
    {
      // Clean up the existing engine
      this->engine_ = nullptr;
      ClearGlobalMemoryManager();
    }
    this->engine_ =
        std::make_unique<Engine>(max_single_sequence_length, tokenizer_path,
                                 kv_cache_config_json_str, f_request_callback, model_infos);
  }

  /*! \brief Construct an EngineModule. */
  static tvm::runtime::Module Create() { return Module(make_object<EngineModule>()); }
  /*! \brief Redirection to `Engine::AddRequest`. */
  void AddRequest(Request request) { return GetEngine()->AddRequest(request); }
  /*! \brief Redirection to `Engine::AbortRequest`. */
  void Abort(String request_id) { return GetEngine()->AbortRequest(request_id); }
  /*! \brief Redirection to `Engine::Step`. */
  void Step() { return GetEngine()->Step(); }
  /*! \brief Get the engine request callback function. */
  Optional<PackedFunc> GetRequestCallback() { return GetEngine()->f_request_callback_; }
  /*! \brief Set the engine request callback function. */
  void SetRequestCallback(Optional<PackedFunc> f_request_callback) {
    GetEngine()->f_request_callback_ = std::move(f_request_callback);
  }
  /*! \brief Redirection to `Engine::ResetEngine`. */
  void Reset() { return GetEngine()->ResetEngine(); }
  /*! \brief Getting stats from the engine */
  String Stats() { return GetEngine()->estate_->stats.AsJSON(); }

 private:
  Engine* GetEngine() {
    ICHECK(engine_ != nullptr) << "Engine is not initialized via reload";
    return engine_.get();
  }

  std::unique_ptr<Engine> engine_ = nullptr;
};

TVM_REGISTER_GLOBAL("mlc.serve.create_engine").set_body_typed(EngineModule::Create);

}  // namespace serve
}  // namespace llm
}  // namespace mlc
