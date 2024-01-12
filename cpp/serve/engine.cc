/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine.cc
 * \brief The implementation for runtime module of serving engine module in MLC LLM.
 */
#include "engine.h"

#include <dlpack/dlpack.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/module.h>
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

/*! \brief The implementation of Engine. */
class EngineImpl : public Engine {
  friend class EngineModule;

 public:
  /********************** Engine Management **********************/

  explicit EngineImpl(int max_single_sequence_length, const String& tokenizer_path,
                      const String& kv_cache_config_json_str,
                      Optional<PackedFunc> request_stream_callback,
                      const std::vector<std::tuple<TVMArgValue, String, DLDevice>>& model_infos) {
    CHECK_GE(model_infos.size(), 1) << "ValueError: No model is provided in the engine.";
    // Step 1. Initialize metadata and singleton states inside the engine
    this->estate_->Reset();
    this->max_single_sequence_length_ = max_single_sequence_length;
    this->kv_cache_config_ = KVCacheConfig(kv_cache_config_json_str, max_single_sequence_length);
    this->request_stream_callback_ = std::move(request_stream_callback);
    this->sampler_ = Sampler::Create(/*sampler_kind=*/"cpu");
    this->tokenizer_ = Tokenizer::FromPath(tokenizer_path);
    // Step 2. Initialize each model independently.
    this->models_.clear();
    for (const auto& model_info : model_infos) {
      TVMArgValue model_lib = std::get<0>(model_info);
      String model_path = std::get<1>(model_info);
      DLDevice device = std::get<2>(model_info);
      Model model = Model::Create(model_lib, std::move(model_path), device);
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

  void Reset() final {
    estate_->Reset();
    for (Model model : models_) {
      model->Reset();
    }
  }

  bool Empty() final { return estate_->request_states.empty(); }

  String Stats() final { return estate_->stats.AsJSON(); }

  Optional<PackedFunc> GetRequestStreamCallback() final { return request_stream_callback_; }

  void SetRequestStreamCallback(Optional<PackedFunc> request_stream_callback) final {
    request_stream_callback_ = std::move(request_stream_callback);
  }

  /***************** High-level Request Management *****************/

  void AddRequest(Request request) final {
    // Get a request copy where all text inputs are tokenized.
    request = Request::FromUntokenized(request, tokenizer_);
    ICHECK_NE(request->input_total_length, -1);
    // Append to the waiting queue and create the request state.
    estate_->waiting_queue.push_back(request);
    estate_->request_states.emplace(request->id, RequestState(request, models_.size()));
  }

  void AbortRequest(const String& request_id) final {
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

  void Step() final {
    CHECK(request_stream_callback_.defined())
        << "The request stream callback is not set. Engine cannot execute.";
    for (EngineAction action : actions_) {
      Array<Request> processed_requests = action->Step(estate_);
      if (!processed_requests.empty()) {
        ActionStepPostProcess(processed_requests, estate_, models_,
                              request_stream_callback_.value(), max_single_sequence_length_);
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
  Tokenizer tokenizer_;
  // Models
  Array<Model> models_;
  // Request stream callback function
  Optional<PackedFunc> request_stream_callback_;
  // Engine actions.
  Array<EngineAction> actions_;
};

std::unique_ptr<Engine> Engine::Create(
    int max_single_sequence_length, const String& tokenizer_path,
    const String& kv_cache_config_json_str, Optional<PackedFunc> request_stream_callback,
    const std::vector<std::tuple<TVMArgValue, String, DLDevice>>& model_infos) {
  return std::make_unique<EngineImpl>(max_single_sequence_length, tokenizer_path,
                                      kv_cache_config_json_str, request_stream_callback,
                                      model_infos);
}

/*! \brief Clear global memory manager */
void ClearGlobalMemoryManager() {
  static const char* kFunc = "vm.builtin.memory_manager.clear";
  const PackedFunc* f = tvm::runtime::Registry::Get(kFunc);
  CHECK(f != nullptr) << "ValueError: Cannot find function `" << kFunc << "` in TVM runtime";
  (*f)();
}

std::unique_ptr<Engine> CreateEnginePacked(TVMArgs args) {
  static const char* kErrorMessage =
      "With `n` models, engine initialization "
      "takes (4 + 4 * n) arguments. The first 4 arguments should be: "
      "1) (int) maximum length of a sequence, which must be equal or smaller than the context "
      "window size of each model; "
      "2) (string) path to tokenizer configuration files, which in MLC LLM, usually in a model "
      "weights directory; "
      "3) (string) JSON configuration for the KVCache; "
      "4) (packed function, optional) global request stream callback function. "
      "The following (4 * n) arguments, 4 for each model, should be: "
      "1) (tvm.runtime.Module) The model library loaded into TVM's RelaxVM; "
      "2) (string) Model path which includes weights and mlc-chat-config.json; "
      "3) (int, enum DLDeviceType) Device type, e.g. CUDA, ROCm, etc; "
      "4) (int) Device id, i.e. the ordinal index of the device that exists locally.";

  ClearGlobalMemoryManager();
  int num_models = (args.size() - 4) / 4;
  int max_single_sequence_length;
  std::string tokenizer_path;
  std::string kv_cache_config_json_str;
  Optional<PackedFunc> request_stream_callback;
  std::vector<std::tuple<TVMArgValue, String, DLDevice>> model_infos;
  model_infos.reserve(num_models);
  try {
    CHECK_EQ(num_models * 4 + 4, args.size()) << "Incorrect number of arguments.";
    max_single_sequence_length = args.At<int>(0);
    tokenizer_path = args.At<std::string>(1);
    kv_cache_config_json_str = args.At<std::string>(2);
    request_stream_callback = args.At<Optional<PackedFunc>>(3);
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
  return Engine::Create(max_single_sequence_length, tokenizer_path, kv_cache_config_json_str,
                        request_stream_callback, model_infos);
}

class EngineModule : public ModuleNode {
 public:
  TVM_MODULE_VTABLE_BEGIN("mlc.serve.engine");
  TVM_MODULE_VTABLE_ENTRY_PACKED("init", &EngineModule::InitPacked);
  TVM_MODULE_VTABLE_ENTRY("add_request", &EngineModule::AddRequest);
  TVM_MODULE_VTABLE_ENTRY("abort_request", &EngineModule::Abort);
  TVM_MODULE_VTABLE_ENTRY("step", &EngineModule::Step);
  TVM_MODULE_VTABLE_ENTRY("stats", &EngineModule::Stats);
  TVM_MODULE_VTABLE_ENTRY("reset", &EngineModule::Reset);
  TVM_MODULE_VTABLE_ENTRY("get_request_stream_callback", &EngineModule::GetRequestStreamCallback);
  TVM_MODULE_VTABLE_ENTRY("set_request_stream_callback", &EngineModule::SetRequestStreamCallback);
  TVM_MODULE_VTABLE_END();

  void InitPacked(TVMArgs args, TVMRetValue* rv) { this->engine_ = CreateEnginePacked(args); }

  /*! \brief Construct an EngineModule. */
  static tvm::runtime::Module Create() { return Module(make_object<EngineModule>()); }
  /*! \brief Redirection to `Engine::AddRequest`. */
  void AddRequest(Request request) { return GetEngine()->AddRequest(std::move(request)); }
  /*! \brief Redirection to `Engine::AbortRequest`. */
  void Abort(const String& request_id) { return GetEngine()->AbortRequest(request_id); }
  /*! \brief Redirection to `Engine::Step`. */
  void Step() { return GetEngine()->Step(); }
  /*! \brief Redirection to `Engine::GetRequestStreamCallback`. */
  Optional<PackedFunc> GetRequestStreamCallback() {
    return GetEngine()->GetRequestStreamCallback();
  }
  /*! \brief Redirection to `Engine::SetRequestStreamCallback` */
  void SetRequestStreamCallback(Optional<PackedFunc> request_stream_callback) {
    GetEngine()->SetRequestStreamCallback(std::move(request_stream_callback));
  }
  /*! \brief Redirection to `Engine::Reset`. */
  void Reset() { return GetEngine()->Reset(); }
  /*! \brief Redirection to `Engine::Stats` */
  String Stats() { return GetEngine()->Stats(); }

 private:
  Engine* GetEngine() {
    ICHECK(engine_ != nullptr) << "Engine is not initialized via init";
    return engine_.get();
  }

  std::unique_ptr<Engine> engine_ = nullptr;
};

TVM_REGISTER_GLOBAL("mlc.serve.create_engine").set_body_typed(EngineModule::Create);

}  // namespace serve
}  // namespace llm
}  // namespace mlc
