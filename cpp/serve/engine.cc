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
                      const String& kv_cache_config_json_str, const String& engine_mode_json_str,
                      Optional<PackedFunc> request_stream_callback,
                      Optional<EventTraceRecorder> trace_recorder,
                      const std::vector<std::tuple<TVMArgValue, String, DLDevice>>& model_infos) {
    CHECK_GE(model_infos.size(), 1) << "ValueError: No model is provided in the engine.";
    // Step 1. Initialize metadata and singleton states inside the engine
    this->estate_->Reset();
    this->max_single_sequence_length_ = max_single_sequence_length;
    this->kv_cache_config_ = KVCacheConfig(kv_cache_config_json_str, max_single_sequence_length);
    this->engine_mode_ = EngineMode(engine_mode_json_str);
    this->request_stream_callback_ = std::move(request_stream_callback);
    this->trace_recorder_ = trace_recorder;
    this->tokenizer_ = Tokenizer::FromPath(tokenizer_path);
    this->token_table_ = tokenizer_->TokenTable();
    this->json_grammar_state_init_ctx_ =
        GrammarStateMatcher::CreateInitContext(BNFGrammar::GetGrammarOfJSON(), this->token_table_);
    // Step 2. Initialize each model independently.
    //         Create the logit processor and sampler.
    this->models_.clear();
    this->model_workspaces_.clear();
    for (const auto& model_info : model_infos) {
      TVMArgValue model_lib = std::get<0>(model_info);
      String model_path = std::get<1>(model_info);
      DLDevice device = std::get<2>(model_info);
      Model model = Model::Create(model_lib, std::move(model_path), device,
                                  kv_cache_config_->max_num_sequence);
      model->CreateKVCache(this->kv_cache_config_);
      CHECK_GE(model->GetMaxWindowSize(), this->max_single_sequence_length_)
          << "The window size of the model, " << model->GetMaxWindowSize()
          << ", is smaller than the pre-defined max single sequence length, "
          << this->max_single_sequence_length_;
      this->models_.push_back(model);
      this->model_workspaces_.push_back(ModelWorkspace{model->AllocEmbeddingTensor()});
    }
    int max_logit_processor_num_token = kv_cache_config_->max_num_sequence;
    if (engine_mode_->enable_speculative) {
      max_logit_processor_num_token *= engine_mode_->spec_draft_length;
    }
    LogitProcessor logit_processor =
        this->models_[0]->CreateLogitProcessor(max_logit_processor_num_token, trace_recorder);
    Sampler sampler = Sampler::Create(/*sampler_kind=*/"cpu", trace_recorder_);
    // Step 3. Initialize engine actions that represent state transitions.
    if (this->engine_mode_->enable_speculative) {
      // Speculative decoding is only possible for more than one model.
      ICHECK_GT(this->models_.size(), 1U);
      this->actions_ = {
          EngineAction::NewRequestPrefill(this->models_,            //
                                          logit_processor,          //
                                          sampler,                  //
                                          this->model_workspaces_,  //
                                          this->kv_cache_config_,   //
                                          this->engine_mode_,       //
                                          this->trace_recorder_),
          EngineAction::BatchDraft(this->models_, logit_processor, sampler, this->trace_recorder_,
                                   this->engine_mode_->spec_draft_length),
          EngineAction::BatchVerify(this->models_, logit_processor, sampler, this->kv_cache_config_,
                                    this->trace_recorder_)};
    } else {
      this->actions_ = {EngineAction::NewRequestPrefill(this->models_,            //
                                                        logit_processor,          //
                                                        sampler,                  //
                                                        this->model_workspaces_,  //
                                                        this->kv_cache_config_,   //
                                                        this->engine_mode_,       //
                                                        this->trace_recorder_),
                        EngineAction::BatchDecode(this->models_, logit_processor, sampler,
                                                  this->trace_recorder_)};
    }
    // Step 4. Automatically set the threading backend max concurrency.
    SetThreadMaxConcurrency();
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
    RECORD_EVENT(trace_recorder_, request->id, "request added to engine");
    // Get a request copy where all text inputs are tokenized.
    request = Request::FromUntokenized(request, tokenizer_);
    ICHECK_NE(request->input_total_length, -1);
    // Append to the waiting queue and create the request state.
    estate_->waiting_queue.push_back(request);

    int n = request->generation_cfg->n;
    int rng_seed = request->generation_cfg->seed;

    std::vector<RequestStateEntry> rsentries;
    // Create the request state entry for the input.
    rsentries.emplace_back(request, models_.size(), estate_->id_manager.GetNewId(), rng_seed,
                           token_table_, json_grammar_state_init_ctx_);
    if (n > 1) {
      // Then create a request state entry for each parallel generation branch.
      // We add a offset to the rng seed so that to make generations different.
      rsentries.reserve(n + 1);
      rsentries[0]->child_indices.reserve(n);
      for (int i = 0; i < n; ++i) {
        rsentries[0]->child_indices.push_back(rsentries.size());
        rsentries.emplace_back(request, models_.size(), estate_->id_manager.GetNewId(),
                               rng_seed + i + 1, token_table_, json_grammar_state_init_ctx_,
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

      // Reduce the input length.
      estate_->stats.current_total_seq_len -= request->input_total_length;
      // Reduce the generated length.
      for (int i = 0; i < static_cast<int>(rstate->entries.size()); ++i) {
        if (rstate->entries[i]->status != RequestStateStatus::kAlive) {
          continue;
        }
        estate_->stats.current_total_seq_len -=
            rstate->entries[i]->mstates[0]->committed_tokens.size();
        RemoveRequestFromModel(estate_, rstate->entries[i]->mstates[0]->internal_id, models_);
        if (rstate->entries[i]->child_indices.empty()) {
          // For each running leaf state, length 1 is over reduced since the last
          // token is not added into KV cache. So we add the length back.
          ++estate_->stats.current_total_seq_len;
        }
      }
    }
    if (it_waiting != estate_->waiting_queue.end()) {
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
        ActionStepPostProcess(processed_requests, estate_, models_, tokenizer_,
                              request_stream_callback_.value(), max_single_sequence_length_);
        return;
      }
    }
    ICHECK(estate_->running_queue.empty())
        << "Internal assumption violated: It is expected that an engine step takes at least one "
           "action (e.g. prefill, decode, etc.) but it does not.";
  }

 private:
  /*! \brief Set the maximum threading backend concurrency. */
  void SetThreadMaxConcurrency() {
    int host_cpu_usage = 1;
    for (Model model : models_) {
      host_cpu_usage += model->EstimateHostCPURequirement();
    }
    int max_concurrency = tvm::runtime::threading::MaxConcurrency();
    tvm::runtime::threading::SetMaxConcurrency(std::min(
        std::max(max_concurrency - host_cpu_usage, 1), kv_cache_config_->max_num_sequence));
  }

  // Engine state, managing requests and request states.
  EngineState estate_;
  // Configurations and singletons
  KVCacheConfig kv_cache_config_;
  EngineMode engine_mode_;
  int max_single_sequence_length_;
  Tokenizer tokenizer_;
  std::vector<std::string> token_table_;
  // The initial context for the grammar state matching of JSON.
  std::shared_ptr<GrammarStateInitContext> json_grammar_state_init_ctx_;
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

std::unique_ptr<Engine> Engine::Create(
    int max_single_sequence_length, const String& tokenizer_path,
    const String& kv_cache_config_json_str, const String& engine_mode_json_str,
    Optional<PackedFunc> request_stream_callback, Optional<EventTraceRecorder> trace_recorder,
    const std::vector<std::tuple<TVMArgValue, String, DLDevice>>& model_infos) {
  return std::make_unique<EngineImpl>(
      max_single_sequence_length, tokenizer_path, kv_cache_config_json_str, engine_mode_json_str,
      request_stream_callback, std::move(trace_recorder), model_infos);
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
      "takes (6 + 4 * n) arguments. The first 6 arguments should be: "
      "1) (int) maximum length of a sequence, which must be equal or smaller than the context "
      "window size of each model; "
      "2) (string) path to tokenizer configuration files, which in MLC LLM, usually in a model "
      "weights directory; "
      "3) (string) JSON configuration for the KVCache; "
      "4) (string) JSON mode for Engine;"
      "5) (packed function, optional) global request stream callback function. "
      "6) (EventTraceRecorder, optional) the event trace recorder for requests."
      "The following (4 * n) arguments, 4 for each model, should be: "
      "1) (tvm.runtime.Module) The model library loaded into TVM's RelaxVM; "
      "2) (string) Model path which includes weights and mlc-chat-config.json; "
      "3) (int, enum DLDeviceType) Device type, e.g. CUDA, ROCm, etc; "
      "4) (int) Device id, i.e. the ordinal index of the device that exists locally.";

  ClearGlobalMemoryManager();
  const int num_non_model_args = 6;
  const int num_model_args = 4;
  int num_models = (args.size() - num_non_model_args) / num_model_args;
  int max_single_sequence_length;
  std::string tokenizer_path;
  std::string kv_cache_config_json_str;
  std::string engine_mode_json_str;
  Optional<PackedFunc> request_stream_callback;
  Optional<EventTraceRecorder> trace_recorder;
  std::vector<std::tuple<TVMArgValue, String, DLDevice>> model_infos;
  model_infos.reserve(num_models);
  try {
    CHECK_LE(num_models * num_model_args + num_non_model_args, args.size())
        << "Incorrect number of arguments.";
    max_single_sequence_length = args.At<int>(0);
    tokenizer_path = args.At<std::string>(1);
    kv_cache_config_json_str = args.At<std::string>(2);
    engine_mode_json_str = args.At<std::string>(3);
    request_stream_callback = args.At<Optional<PackedFunc>>(4);
    trace_recorder = args.At<Optional<EventTraceRecorder>>(5);
    for (int i = 0; i < num_models; ++i) {
      TVMArgValue model_lib = args[i * num_model_args + num_non_model_args];
      std::string model_path = args.At<std::string>(i * num_model_args + num_non_model_args + 1);
      DLDeviceType device_type =
          static_cast<DLDeviceType>(args.At<int>(i * num_model_args + num_non_model_args + 2));
      int device_id = args.At<int>(i * num_model_args + num_non_model_args + 3);
      model_infos.emplace_back(model_lib, model_path, DLDevice{device_type, device_id});
    }
  } catch (const dmlc::Error& e) {
    LOG(FATAL) << "ValueError: " << e.what() << kErrorMessage;
  }
  return Engine::Create(max_single_sequence_length, tokenizer_path, kv_cache_config_json_str,
                        engine_mode_json_str, request_stream_callback, std::move(trace_recorder),
                        model_infos);
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
