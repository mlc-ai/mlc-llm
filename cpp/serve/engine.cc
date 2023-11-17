/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine.cc
 * \brief The implementation for runtime module of serving engine module in MLC LLM.
 */
#define __STDC_FORMAT_MACROS

#include <tokenizers_cpp.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

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

  /*!
   * \brief (Re)initialize the engine with the given lists of
   * models and KV cache config.
   * \param reload_libs The model libraries of the input models.
   * \param model_paths The weight/config directories of the input models.
   * \param devices The devices where each of the input model runs.
   * \param kv_cache_config_json The page KV cache configuration.
   * \note `reload_libs`, `model_paths` and `devices` should have the same size.
   */
  void Reload(std::vector<TVMArgValue> reload_libs, std::vector<String> model_paths,
              std::vector<DLDevice> devices, String kv_cache_config_json) {
    int num_models = reload_libs.size();
    ICHECK_GE(num_models, 1);
    ICHECK_EQ(model_paths.size(), num_models);
    ICHECK_EQ(devices.size(), num_models);

    // Step 1. Create models and their PackedFuncs.
    ICHECK(models_.empty());
    models_.reserve(num_models);
    for (int i = 0; i < num_models; ++i) {
      models_.push_back(Model::Create(reload_libs[i], model_paths[i], devices[i]));
    }
    // Step 2. Fetch max single sequence length from models.
    max_single_sequence_length_ = std::numeric_limits<int>::max();
    for (Model model : models_) {
      int max_window_size = model->GetMaxWindowSize();
      max_single_sequence_length_ = std::min(max_single_sequence_length_, max_window_size);
    }
    // Step 3. Process KV cache config json string.
    kv_cache_config_ = KVCacheConfig(kv_cache_config_json, max_single_sequence_length_);
    // Step 4. Create KV cache for each model.
    for (Model model : models_) {
      model->CreateKVCache(kv_cache_config_);
    }
    // Step 5. Create sampler and tokenizer.
    //         The tokenizer is created from the first model.
    //         We assume all models have the same tokenizer, which is the basic
    //         requirement of speculative encoding.
    sampler_ = Sampler::Create(/*sampler_kind=*/"cpu");
    tokenizer_ = TokenizerFromPath(model_paths[0]);
    // Step 6. Initialize action lists.
    action_abort_request_ = EngineAction::AbortRequest(models_);
    action_new_request_prefill_ = EngineAction::NewRequestPrefill(
        models_, sampler_, kv_cache_config_, max_single_sequence_length_);
    action_batch_decode_ = EngineAction::BatchDecode(models_, sampler_);

    ResetEngine();
  }

  /*! \brief Reset the engine, clean up all running data and statistics. */
  void ResetEngine() {
    ICHECK(estate_.defined());
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
  void AbortRequest(Request request) { estate_->abort_queue.push_back(request); }

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
    // - Action 0. Abort requests.
    action_abort_request_->Step(estate_);

    // - Action 1. Prefill the front-most waiting request.
    bool prefill_processed = action_new_request_prefill_->Step(estate_);
    if (prefill_processed) {
      return;
    }

    // - Action 2. Run decode step.
    bool decode_processed = action_batch_decode_->Step(estate_);
    if (decode_processed) {
      ProcessFinishedRequest(estate_, models_, tokenizer_, max_single_sequence_length_);
      return;
    }

    ICHECK(estate_->running_queue.empty())
        << "Not taking any action in a step is not expected with running requests.";
  }

 private:
  // Engine state, managing requests and request states.
  EngineState estate_;

  // Models, sampler and tokenizer.
  Array<Model> models_;
  Sampler sampler_;
  std::unique_ptr<Tokenizer> tokenizer_;

  // Engine actions.
  EngineAction action_abort_request_;
  EngineAction action_new_request_prefill_;
  EngineAction action_batch_decode_;

  // Configurations
  KVCacheConfig kv_cache_config_;
  int max_single_sequence_length_ = -1;
};

class EngineModule : public ModuleNode {
 public:
  // clear global memory manager
  static void ClearGlobalMemoryManager() {
    // Step 0. Clear the previously allocated memory.
    const PackedFunc* fclear_memory_manager =
        tvm::runtime::Registry::Get("vm.builtin.memory_manager.clear");
    ICHECK(fclear_memory_manager) << "Cannot find env function vm.builtin.memory_manager.clear";
    (*fclear_memory_manager)();
  }

  // overrides
  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "reload") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        // The args of `reload` is expected to be in the following pattern.
        // Assume we want to load `n` models in the engine, then there
        // is supposed to have (4n + 1) arguments.
        // For each i (i from 0 to n),
        // - args[4 * i    ] denotes the model lib,
        // - args[4 * i + 1] denotes the model path (for weights/config),
        // - args[4 * i + 2] denotes the device type,
        // - args[4 * i + 3] denotes the device id.
        // And the last argument denotes the KV cache config in JSON string.

        engine_ = nullptr;
        ClearGlobalMemoryManager();
        engine_ = std::make_unique<Engine>(Engine());
        // num_models x (model lib, model path, device type, device id) + kv_cache_config
        std::vector<TVMArgValue> reload_libs;
        std::vector<String> model_paths;
        std::vector<DLDevice> devices;
        CHECK_EQ(args.size() % 4, 1)
            << "Unexpected number of reload arguments. "
               "Reload arguments should be one or many of (reload_lib, "
               "model_path, device_type, device_id) with a trailing KV cache config JSON string.";

        int num_models = args.size() / 4;
        reload_libs.reserve(num_models);
        model_paths.reserve(num_models);
        devices.reserve(num_models);
        for (int i = 0; i < num_models; ++i) {
          reload_libs.push_back(args[i * 4]);
          model_paths.push_back(args[i * 4 + 1]);
          int device_type = args[i * 4 + 2];
          int device_id = args[i * 4 + 3];
          devices.push_back(DLDevice{static_cast<DLDeviceType>(device_type), device_id});
        }
        engine_->Reload(reload_libs, model_paths, devices, args[num_models * 4]);
      });
    } else if (name == "add_request") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.size(), 1);
        GetEngine()->AddRequest(args[0]);
      });
    } else if (name == "abort") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.size(), 0);
        GetEngine()->AbortRequest(args[0]);
      });
    } else if (name == "step") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.size(), 0);
        GetEngine()->Step();
      });
    } else if (name == "stats") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.size(), 0);
        *rv = GetEngine()->estate_->stats.AsJSON();
      });
    } else if (name == "reset") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.size(), 0);
        GetEngine()->ResetEngine();
      });
    } else {
      return PackedFunc(nullptr);
    }
  }

  Engine* GetEngine() {
    ICHECK(engine_ != nullptr) << "Engine is not initialized via reload";
    return engine_.get();
  }

  const char* type_key() const final { return "mlc.serve.engine"; }

 private:
  std::unique_ptr<Engine> engine_ = nullptr;
};

tvm::runtime::Module CreateEngineModule() {
  ObjectPtr<EngineModule> n = make_object<EngineModule>();
  return Module(n);
}

// register as a system function that can be queried
TVM_REGISTER_GLOBAL("mlc.serve.create_engine").set_body_typed(CreateEngineModule);

}  // namespace serve
}  // namespace llm
}  // namespace mlc
