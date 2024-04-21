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
#include <tvm/runtime/threading_backend.h>

#include <optional>
#include <tuple>
#include <unordered_set>

#include "../tokenizers.h"
#include "engine_actions/action.h"
#include "engine_actions/action_commons.h"
#include "engine_state.h"
#include "event_trace_recorder.h"
#include "grammar/grammar_state_matcher.h"
#include "logit_processor.h"
#include "model.h"
#include "request.h"
#include "request_state.h"
#include "sampler/sampler.h"

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

  explicit EngineImpl(EngineConfig engine_config, Optional<PackedFunc> request_stream_callback,
                      Optional<EventTraceRecorder> trace_recorder) {
    // Step 1. Initialize metadata and singleton states inside the engine
    this->estate_->Reset();
    // Being "-1" means there is no limit on single sequence length.
    if (engine_config->max_single_sequence_length == -1) {
      engine_config->max_single_sequence_length = std::numeric_limits<int>::max();
    }
    this->request_stream_callback_ = std::move(request_stream_callback);
    this->trace_recorder_ = trace_recorder;
    this->tokenizer_ = Tokenizer::FromPath(engine_config->model);
    this->token_table_ = tokenizer_->TokenTable();
    this->grammar_init_context_storage_ = GrammarInitContextStorage(this->token_table_);
    // Step 2. Initialize each model independently.
    //         Create the logit processor and sampler.
    this->models_.clear();
    this->model_workspaces_.clear();

    auto f_create_model = [this, &engine_config, &trace_recorder](const String& model_path,
                                                                  const String& model_lib_path) {
      Model model = Model::Create(model_lib_path, std::move(model_path), engine_config->device,
                                  engine_config->max_num_sequence,
                                  /*trace_enabled=*/trace_recorder.defined());
      model->CreateKVCache(engine_config->kv_cache_page_size, engine_config->max_num_sequence,
                           engine_config->max_total_sequence_length,
                           engine_config->prefill_chunk_size);
      CHECK_GE(model->GetMaxWindowSize(), engine_config->max_single_sequence_length)
          << "The window size of the model, " << model->GetMaxWindowSize()
          << ", is smaller than the pre-defined max single sequence length, "
          << engine_config->max_single_sequence_length;
      this->models_.push_back(model);
      this->model_workspaces_.push_back(
          ModelWorkspace{model->AllocEmbeddingTensor(), model->AllocHiddenStatesTensor()});
    };

    f_create_model(engine_config->model, engine_config->model_lib_path);
    CHECK_EQ(engine_config->additional_models.size(),
             engine_config->additional_model_lib_paths.size())
        << "The additional model and lib path list has mismatched size.";
    for (int i = 0; i < static_cast<int>(engine_config->additional_models.size()); ++i) {
      f_create_model(engine_config->additional_models[i],
                     engine_config->additional_model_lib_paths[i]);
    }

    int max_num_tokens = engine_config->max_num_sequence;
    if (engine_config->speculative_mode != SpeculativeMode::kDisable) {
      max_num_tokens *= engine_config->spec_draft_length;
    }
    LogitProcessor logit_processor =
        this->models_[0]->CreateLogitProcessor(max_num_tokens, trace_recorder);
    Sampler sampler = this->models_[0]->CreateSampler(
        max_num_tokens, static_cast<int>(this->models_.size()), trace_recorder);
    // Step 3. Initialize engine actions that represent state transitions.
    if (engine_config->speculative_mode != SpeculativeMode::kDisable) {
      // Speculative decoding is only possible for more than one model.
      ICHECK_GT(this->models_.size(), 1U);
      switch (engine_config->speculative_mode) {
        case SpeculativeMode::kEagle:
          this->actions_ = {EngineAction::EagleNewRequestPrefill(this->models_,            //
                                                                 logit_processor,          //
                                                                 sampler,                  //
                                                                 this->model_workspaces_,  //
                                                                 engine_config,            //
                                                                 this->trace_recorder_),
                            EngineAction::EagleBatchDraft(
                                this->models_, logit_processor, sampler, this->model_workspaces_,
                                this->trace_recorder_, engine_config->spec_draft_length),
                            EngineAction::EagleBatchVerify(this->models_, logit_processor, sampler,
                                                           this->model_workspaces_, engine_config,
                                                           this->trace_recorder_)};
          break;
        default:
          this->actions_ = {EngineAction::NewRequestPrefill(this->models_,            //
                                                            logit_processor,          //
                                                            sampler,                  //
                                                            this->model_workspaces_,  //
                                                            engine_config,            //
                                                            this->trace_recorder_),
                            EngineAction::BatchDraft(this->models_, logit_processor, sampler,
                                                     this->trace_recorder_),
                            EngineAction::BatchVerify(this->models_, logit_processor, sampler,
                                                      engine_config, this->trace_recorder_)};
      }
    } else {
      this->actions_ = {EngineAction::NewRequestPrefill(this->models_,            //
                                                        logit_processor,          //
                                                        sampler,                  //
                                                        this->model_workspaces_,  //
                                                        engine_config,            //
                                                        this->trace_recorder_),
                        EngineAction::BatchDecode(this->models_, logit_processor, sampler,
                                                  this->trace_recorder_)};
    }
    // Step 4. Automatically set the threading backend max concurrency.
    this->engine_config_ = engine_config;
    SetThreadMaxConcurrency();
  }

  void Reset() final {
    AbortAllRequests();
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
    RECORD_EVENT(trace_recorder_, request->id, "request added to engine");
    // Get a request copy where all text inputs are tokenized.
    request = Request::FromUntokenized(request, tokenizer_);
    ICHECK_NE(request->input_total_length, -1);

    if (request->input_total_length >= engine_config_->max_single_sequence_length &&
        request_stream_callback_.defined()) {
      // If the request input length exceeds the maximum allowed single sequence length,
      // invoke callback and do not process the request.
      Array<RequestStreamOutput> output{RequestStreamOutput(
          request->id, std::vector<IntTuple>(request->generation_cfg->n),
          Optional<Array<Array<String>>>(),
          std::vector<Optional<String>>(request->generation_cfg->n, String("length")))};
      request_stream_callback_.value()(std::move(output));
      return;
    }

    // Append to the waiting queue and create the request state.
    estate_->waiting_queue.push_back(request);

    int n = request->generation_cfg->n;
    int rng_seed = request->generation_cfg->seed;
    auto grammar_state_init_ctx =
        ResponseFormatToGrammarInitContext(request->generation_cfg->response_format);

    std::vector<RequestStateEntry> rsentries;
    // Create the request state entry for the input.
    rsentries.emplace_back(request, models_.size(), estate_->id_manager.GetNewId(), rng_seed,
                           token_table_, grammar_state_init_ctx);
    if (n > 1) {
      // Then create a request state entry for each parallel generation branch.
      // We add a offset to the rng seed so that to make generations different.
      rsentries.reserve(n + 1);
      rsentries[0]->child_indices.reserve(n);
      for (int i = 0; i < n; ++i) {
        rsentries[0]->child_indices.push_back(rsentries.size());
        rsentries.emplace_back(request, models_.size(), estate_->id_manager.GetNewId(),
                               rng_seed + i + 1, token_table_, grammar_state_init_ctx,
                               /*parent_idx=*/0);
      }
    }
    estate_->request_states.emplace(request->id, RequestState(std::move(rsentries)));
  }

  void AbortRequest(const String& request_id) final {
    auto it_rstate = estate_->request_states.find(request_id);
    if (it_rstate == estate_->request_states.end()) {
      // The request to abort does not exist.
      return;
    }

    RequestState rstate = it_rstate->second;
    Request request = rstate->entries[0]->request;

    // - Check if the request is running or pending.
    auto it_running =
        std::find(estate_->running_queue.begin(), estate_->running_queue.end(), request);
    auto it_waiting =
        std::find(estate_->waiting_queue.begin(), estate_->waiting_queue.end(), request);

    for (const RequestStateEntry& rsentry : rstate->entries) {
      estate_->id_manager.RecycleId(rsentry->mstates[0]->internal_id);
    }
    estate_->request_states.erase(request->id);
    if (it_running != estate_->running_queue.end()) {
      // The request to abort is in running queue
      estate_->running_queue.erase(it_running);

      for (int i = static_cast<int>(rstate->entries.size()) - 1; i >= 0; --i) {
        if (rstate->entries[i]->status != RequestStateStatus::kAlive) {
          continue;
        }
        RemoveRequestFromModel(estate_, rstate->entries[i]->mstates[0]->internal_id, models_);
      }
    }
    if (it_waiting != estate_->waiting_queue.end()) {
      // The request to abort is in waiting queue
      estate_->waiting_queue.erase(it_waiting);
    }

    // Send a callback to notice the abortion.
    if (request_stream_callback_.defined()) {
      Array<RequestStreamOutput> output{RequestStreamOutput(
          request_id, std::vector<IntTuple>(request->generation_cfg->n),
          Optional<Array<Array<String>>>(),
          std::vector<Optional<String>>(request->generation_cfg->n, String("abort")))};
      request_stream_callback_.value()(std::move(output));
    }
  }

  void AbortAllRequests() final {
    // - Collect all the request ids.
    std::vector<String> request_ids;
    request_ids.reserve(estate_->request_states.size());
    for (const auto& kv : estate_->request_states) {
      request_ids.push_back(kv.first);
    }
    // - Abort all the requests.
    for (const String& request_id : request_ids) {
      AbortRequest(request_id);
    }
  }

  /*********************** Engine Action ***********************/

  void Step() final {
    CHECK(request_stream_callback_.defined())
        << "The request stream callback is not set. Engine cannot execute.";
    for (EngineAction action : actions_) {
      Array<Request> processed_requests = action->Step(estate_);
      if (!processed_requests.empty()) {
        ActionStepPostProcess(processed_requests, estate_, models_, tokenizer_,
                              request_stream_callback_.value(),
                              engine_config_->max_single_sequence_length);
        return;
      }
    }
    ICHECK(estate_->running_queue.empty())
        << "Internal assumption violated: It is expected that an engine step takes at least one "
           "action (e.g. prefill, decode, etc.) but it does not.";
  }

  /************** Debug/Profile **************/

  void DebugCallFuncOnAllAllWorker(const String& func_name) final {
    CHECK(!models_.empty()) << "There is no model running in Engine.";
    models_[0]->DebugCallFuncOnAllAllWorker(func_name);
  }

 private:
  /*! \brief Set the maximum threading backend concurrency. */
  void SetThreadMaxConcurrency() {
    int host_cpu_usage = 1;
    for (Model model : models_) {
      host_cpu_usage += model->EstimateHostCPURequirement();
    }
    int max_concurrency = tvm::runtime::threading::MaxConcurrency();
    tvm::runtime::threading::SetMaxConcurrency(
        std::min(std::max(max_concurrency - host_cpu_usage, 1), engine_config_->max_num_sequence));
  }

  /*! \brief Create a grammar init context according to the response format. If the response format
   * is not JSON, return std::nullopt. */
  std::optional<std::shared_ptr<GrammarStateInitContext>> ResponseFormatToGrammarInitContext(
      const ResponseFormat& response_format) {
    if (response_format.type != "json_object") {
      return std::nullopt;
    } else if (!response_format.schema) {
      return grammar_init_context_storage_->GetInitContextForJSON();
    } else {
      return grammar_init_context_storage_->GetInitContextForJSONSchema(
          response_format.schema.value());
    }
  }

  // Engine state, managing requests and request states.
  EngineState estate_;
  // Configurations and singletons
  EngineConfig engine_config_;
  Tokenizer tokenizer_;
  std::vector<std::string> token_table_;
  // Helper to get the grammar init context for requests.
  GrammarInitContextStorage grammar_init_context_storage_;
  // Models
  Array<Model> models_;
  // Workspace of each model.
  std::vector<ModelWorkspace> model_workspaces_;
  // Request stream callback function
  Optional<PackedFunc> request_stream_callback_;
  // Engine actions.
  Array<EngineAction> actions_;
  // Event trace recorder.
  Optional<EventTraceRecorder> trace_recorder_;
};

std::unique_ptr<Engine> Engine::Create(EngineConfig engine_config,
                                       Optional<PackedFunc> request_stream_callback,
                                       Optional<EventTraceRecorder> trace_recorder) {
  return std::make_unique<EngineImpl>(std::move(engine_config), std::move(request_stream_callback),
                                      std::move(trace_recorder));
}

/*! \brief Clear global memory manager */
void ClearGlobalMemoryManager() {
  static const char* kFunc = "vm.builtin.memory_manager.clear";
  const PackedFunc* f = tvm::runtime::Registry::Get(kFunc);
  CHECK(f != nullptr) << "ValueError: Cannot find function `" << kFunc << "` in TVM runtime";
  (*f)();
}

class EngineModule : public ModuleNode {
 public:
  TVM_MODULE_VTABLE_BEGIN("mlc.serve.engine");
  TVM_MODULE_VTABLE_ENTRY("init", &EngineModule::Init);
  TVM_MODULE_VTABLE_ENTRY("add_request", &EngineModule::AddRequest);
  TVM_MODULE_VTABLE_ENTRY("abort_request", &EngineModule::Abort);
  TVM_MODULE_VTABLE_ENTRY("step", &EngineModule::Step);
  TVM_MODULE_VTABLE_ENTRY("stats", &EngineModule::Stats);
  TVM_MODULE_VTABLE_ENTRY("reset", &EngineModule::Reset);
  TVM_MODULE_VTABLE_ENTRY("get_request_stream_callback", &EngineModule::GetRequestStreamCallback);
  TVM_MODULE_VTABLE_ENTRY("set_request_stream_callback", &EngineModule::SetRequestStreamCallback);
  TVM_MODULE_VTABLE_END();

  /*! \brief Initialize the engine with config and other fields. */
  void Init(EngineConfig engine_config, Optional<PackedFunc> request_stream_callback,
            Optional<EventTraceRecorder> trace_recorder) {
    this->engine_ = Engine::Create(std::move(engine_config), std::move(request_stream_callback),
                                   std::move(trace_recorder));
  }
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
