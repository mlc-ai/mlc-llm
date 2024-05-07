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

#include <numeric>
#include <optional>
#include <tuple>
#include <unordered_set>

#include "../support/json_parser.h"
#include "../support/result.h"
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

  static Result<EngineCreationOutput> Create(const std::string& engine_config_json_str,
                                             DLDevice device,
                                             Optional<PackedFunc> request_stream_callback,
                                             Optional<EventTraceRecorder> trace_recorder) {
    using TResult = Result<EngineCreationOutput>;
    std::unique_ptr<EngineImpl> n = std::make_unique<EngineImpl>();

    // - Read the models and model libs from the EngineConfig JSON string.
    Result<std::vector<std::pair<std::string, std::string>>> models_and_model_libs_res =
        EngineConfig::GetModelsAndModelLibsFromJSONString(engine_config_json_str);
    if (models_and_model_libs_res.IsErr()) {
      return TResult::Error(models_and_model_libs_res.UnwrapErr());
    }
    std::vector<std::pair<std::string, std::string>> models_and_model_libs =
        models_and_model_libs_res.Unwrap();
    ICHECK_GE(models_and_model_libs.size(), 1);
    // - Initialize singleton states inside the engine.
    n->estate_->Reset();
    n->request_stream_callback_ = std::move(request_stream_callback);
    n->trace_recorder_ = trace_recorder;
    n->device_ = device;
    // - Load model config, create a shared disco session when tensor
    // parallelism is enabled.
    std::vector<picojson::object> model_configs;
    for (int i = 0; i < static_cast<int>(models_and_model_libs.size()); ++i) {
      const auto& [model_str, model_lib] = models_and_model_libs[i];
      Result<picojson::object> model_config_res = Model::LoadModelConfig(model_str);
      if (model_config_res.IsErr()) {
        return TResult::Error("Model " + std::to_string(i) +
                              " has invalid mlc-chat-config.json: " + model_config_res.UnwrapErr());
      }
      model_configs.push_back(model_config_res.Unwrap());
    }
    Optional<Session> session = n->CreateDiscoSession(model_configs, device);
    // - Initialize each model independently.
    n->models_.clear();
    for (int i = 0; i < static_cast<int>(models_and_model_libs.size()); ++i) {
      const auto& [model_str, model_lib] = models_and_model_libs[i];
      Model model = Model::Create(model_lib, model_str, model_configs[i], device, session,
                                  /*trace_enabled=*/trace_recorder.defined());
      n->models_.push_back(model);
    }
    // - Automatically infer the missing fields in EngineConfig JSON strings
    // and get the final EngineConfig.
    Result<EngineConfig> engine_config_res =
        n->AutoDecideEngineConfig(engine_config_json_str, model_configs);
    if (engine_config_res.IsErr()) {
      return TResult::Error(engine_config_res.UnwrapErr());
    }
    EngineConfig engine_config = engine_config_res.Unwrap();
    // - Load model weights, create KV cache and workspace.
    n->model_workspaces_.clear();
    for (const Model& model : n->models_) {
      model->LoadParams();
      model->SetMaxNumSequence(engine_config->max_num_sequence);
      model->SetPrefillChunkSize(engine_config->prefill_chunk_size);
      model->CreateKVCache(engine_config->kv_cache_page_size, engine_config->max_num_sequence,
                           engine_config->max_total_sequence_length,
                           engine_config->prefill_chunk_size, engine_config->max_history_size,
                           engine_config->kv_state_kind);
      n->model_workspaces_.push_back(
          ModelWorkspace{model->AllocEmbeddingTensor(), model->AllocHiddenStatesTensor()});
    }
    // - Initialize tokenizer and grammar
    n->tokenizer_ = Tokenizer::FromPath(engine_config->model);
    std::string token_table_postproc_method;
    if (model_configs[0].count("token_table_postproc_method") == 0) {
      // Backward compatibility: use "byte_fallback" by default
      token_table_postproc_method = "byte_fallback";
    } else {
      token_table_postproc_method =
          model_configs[0].at("token_table_postproc_method").get<std::string>();
    }
    n->token_table_ =
        Tokenizer::PostProcessTokenTable(n->tokenizer_->TokenTable(), token_table_postproc_method);
    n->grammar_init_context_storage_ = GrammarInitContextStorage(n->token_table_);
    // - Create the logit processor and sampler, and
    // the DraftTokenWorkspaceManager for speculative decoding.
    int max_num_tokens = engine_config->max_num_sequence;
    DraftTokenWorkspaceManager draft_token_workspace_manager{nullptr};
    if (engine_config->speculative_mode != SpeculativeMode::kDisable) {
      max_num_tokens *= engine_config->spec_draft_length + 1;
      // multiply max num_tokens by two so we can do ping-pong swaping during draft/verify process
      draft_token_workspace_manager =
          n->models_[0]->CreateDraftTokenWorkspaceManager(max_num_tokens * 2);
      draft_token_workspace_manager->AllocWorkspace(
          &n->model_workspaces_[0],
          /*require_hidden_states=*/engine_config->speculative_mode == SpeculativeMode::kEagle);
    }
    LogitProcessor logit_processor =
        n->models_[0]->CreateLogitProcessor(max_num_tokens, trace_recorder);
    Sampler sampler = n->models_[0]->CreateSampler(
        max_num_tokens, static_cast<int>(n->models_.size()), trace_recorder);
    // - Initialize engine actions that represent state transitions.
    if (engine_config->speculative_mode != SpeculativeMode::kDisable) {
      // Speculative decoding is only possible for more than one model.
      ICHECK_GT(n->models_.size(), 1U);
      switch (engine_config->speculative_mode) {
        case SpeculativeMode::kEagle:
          n->actions_ = {
              EngineAction::EagleNewRequestPrefill(n->models_,                     //
                                                   logit_processor,                //
                                                   sampler,                        //
                                                   n->model_workspaces_,           //
                                                   draft_token_workspace_manager,  //
                                                   engine_config,                  //
                                                   n->trace_recorder_),
              EngineAction::EagleBatchDraft(n->models_, logit_processor, sampler,
                                            n->model_workspaces_, draft_token_workspace_manager,
                                            n->trace_recorder_, engine_config->spec_draft_length),
              EngineAction::EagleBatchVerify(n->models_, logit_processor, sampler,
                                             n->model_workspaces_, draft_token_workspace_manager,
                                             engine_config, n->trace_recorder_)};
          break;
        default:
          n->actions_ = {
              EngineAction::NewRequestPrefill(n->models_,            //
                                              logit_processor,       //
                                              sampler,               //
                                              n->model_workspaces_,  //
                                              engine_config,         //
                                              n->trace_recorder_),
              EngineAction::BatchDraft(n->models_, logit_processor, sampler, n->model_workspaces_,
                                       draft_token_workspace_manager, n->trace_recorder_,
                                       engine_config->spec_draft_length),
              EngineAction::BatchVerify(n->models_, logit_processor, sampler, n->model_workspaces_,
                                        draft_token_workspace_manager, engine_config,
                                        n->trace_recorder_)};
      }
    } else {
      n->actions_ = {
          EngineAction::NewRequestPrefill(n->models_,            //
                                          logit_processor,       //
                                          sampler,               //
                                          n->model_workspaces_,  //
                                          engine_config,         //
                                          n->trace_recorder_),
          EngineAction::BatchDecode(n->models_, logit_processor, sampler, n->trace_recorder_)};
    }
    // - Automatically set the threading backend max concurrency.
    n->engine_config_ = engine_config;
    n->SetThreadMaxConcurrency();
    // - Get the default generation config from the first model.
    GenerationConfig default_generation_cfg =
        GenerationConfig::GetDefaultFromModelConfig(model_configs[0]);
    return TResult::Ok({std::move(n), std::move(engine_config), std::move(default_generation_cfg)});
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

  /************** Utility Functions **************/
  Optional<Session> CreateDiscoSession(const std::vector<picojson::object>& model_configs,
                                       Device device) {
    const auto& base_model_config = model_configs[0];

    auto f_get_num_shards = [](const picojson::object& model_config) -> int {
      constexpr auto kNumShardsKey = "tensor_parallel_shards";
      if (model_config.count(kNumShardsKey)) {
        const auto& val = model_config.at(kNumShardsKey);
        CHECK(val.is<int64_t>());
        return static_cast<int>(val.get<int64_t>());
      } else {
        LOG(FATAL) << "Key \"tensor_parallel_shards\" not found.";
      }
      throw;
    };

    int num_shards = std::transform_reduce(
        model_configs.begin(), model_configs.end(), 1, [](int a, int b) { return std::max(a, b); },
        f_get_num_shards);
    Optional<Session> session = NullOpt;
    if (num_shards > 1) {
      constexpr const char* f_create_process_pool = "runtime.disco.create_process_pool";
      if (Registry::Get(f_create_process_pool) == nullptr) {
        LOG(FATAL) << "Cannot find process launcher `" << f_create_process_pool << "`. "
                   << "Multi-GPU inference depends on MLC LLM Python API to launch process.";
      }
      std::string ccl;
      if (device.device_type == kDLCUDA) {
        ccl = "nccl";
      } else if (device.device_type == kDLROCM) {
        ccl = "rccl";
      } else {
        LOG(FATAL) << "ValueError: Multi-GPU on device " << DLDeviceType2Str(device.device_type)
                   << " is not supported. Currently, only NCCL and RCCL are integrated.";
      }
      std::vector<int64_t> device_ids(num_shards);
      for (int i = 0; i < num_shards; ++i) {
        device_ids[i] = i;
      }
      session = Session::ProcessSession(num_shards, f_create_process_pool, "mlc_llm.cli.worker");
      session.value()->InitCCL(ccl, ShapeTuple(device_ids));
    }
    return session;
  }

  /************** Debug/Profile **************/

  void DebugCallFuncOnAllAllWorker(const String& func_name) final {
    CHECK(!models_.empty()) << "There is no model running in Engine.";
    models_[0]->DebugCallFuncOnAllAllWorker(func_name);
  }

 private:
  Result<EngineConfig> AutoDecideEngineConfig(const std::string& engine_config_json_str,
                                              const std::vector<picojson::object>& model_configs) {
    using TResult = Result<EngineConfig>;
    picojson::value config_json;
    std::string err = picojson::parse(config_json, engine_config_json_str);
    if (!err.empty()) {
      return TResult::Error(err);
    }
    picojson::object config = config_json.get<picojson::object>();
    ObjectPtr<EngineConfigNode> n = make_object<EngineConfigNode>();

    // - Get the engine mode and maximum GPU utilization for inference.
    EngineMode mode = EngineModeFromString(json::Lookup<std::string>(config, "mode"));
    double gpu_memory_utilization =
        json::LookupOrDefault<double>(config, "gpu_memory_utilization", n->gpu_memory_utilization);
    bool verbose = json::LookupOrDefault<bool>(config, "verbose", n->verbose);

    // - Get the config fields that can be automatically inferred.
    std::optional<int64_t> max_num_sequence =
        json::LookupOptional<int64_t>(config, "max_num_sequence");
    std::optional<int64_t> max_total_sequence_length =
        json::LookupOptional<int64_t>(config, "max_total_sequence_length");
    std::optional<int64_t> max_single_sequence_length =
        json::LookupOptional<int64_t>(config, "max_single_sequence_length");
    std::optional<int64_t> prefill_chunk_size =
        json::LookupOptional<int64_t>(config, "prefill_chunk_size");
    std::optional<int64_t> max_history_size =
        json::LookupOptional<int64_t>(config, "max_history_size");
    std::optional<std::string> kv_state_kind_str =
        json::LookupOptional<std::string>(config, "kv_state_kind");
    std::optional<KVStateKind> kv_state_kind;
    if (kv_state_kind_str.has_value()) {
      kv_state_kind = KVStateKindFromString(kv_state_kind_str.value());
    }
    InferrableEngineConfig inferrable_cfg{max_num_sequence,           max_total_sequence_length,
                                          max_single_sequence_length, prefill_chunk_size,
                                          max_history_size,           kv_state_kind};

    // - Get the model metadata.
    std::vector<ModelMetadata> model_metadata;
    for (const Model& model : models_) {
      model_metadata.push_back(model->GetMetadata());
    }
    // - Select from kv cache or RNN state.
    Result<bool> use_kv_cache = ModelsUseKVCache(model_configs);
    if (use_kv_cache.IsErr()) {
      return TResult::Error(use_kv_cache.UnwrapErr());
    }
    KVStateKind inferred_kv_state_kind;
    Result<InferrableEngineConfig> inferrable_cfg_res;
    if (use_kv_cache.Unwrap()) {
      inferred_kv_state_kind = KVStateKind::kKVCache;
      // - Check if the kv state kind from config is valid.
      if (kv_state_kind.has_value() && kv_state_kind.value() != inferred_kv_state_kind) {
        return TResult::Error(
            "Invalid kv state kind in EngineConfig. The models use KV cache, but RNN state is "
            "specified in EngineConfig.");
      }
      // - Infer configuration.
      inferrable_cfg_res = InferrableEngineConfig::InferForKVCache(
          mode, device_, gpu_memory_utilization, model_configs, model_metadata, inferrable_cfg,
          verbose);
    } else {
      inferred_kv_state_kind = KVStateKind::kRNNState;
      // - Check if the kv state kind from config is valid.
      if (kv_state_kind.has_value() && kv_state_kind.value() != inferred_kv_state_kind) {
        return TResult::Error(
            "Invalid kv state kind in EngineConfig. The models use RNN state, but KV cache is "
            "specified in EngineConfig.");
      }
      // - Infer configuration.
      inferrable_cfg_res = InferrableEngineConfig::InferForRNNState(
          mode, device_, gpu_memory_utilization, model_configs, model_metadata, inferrable_cfg,
          verbose);
    }

    if (inferrable_cfg_res.IsErr()) {
      return TResult::Error(inferrable_cfg_res.UnwrapErr());
    }
    inferrable_cfg = inferrable_cfg_res.Unwrap();
    ICHECK(inferrable_cfg.max_num_sequence.has_value());
    ICHECK(inferrable_cfg.max_total_sequence_length.has_value());
    ICHECK(inferrable_cfg.max_single_sequence_length.has_value());
    ICHECK(inferrable_cfg.prefill_chunk_size.has_value());
    ICHECK(inferrable_cfg.max_history_size.has_value());
    ICHECK(inferrable_cfg.kv_state_kind.has_value());
    return TResult::Ok(EngineConfig::FromJSONAndInferredConfig(config, inferrable_cfg));
  }

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
  // Device that the models run on.
  Device device_;
  // Workspace of each model.
  std::vector<ModelWorkspace> model_workspaces_;
  // Request stream callback function
  Optional<PackedFunc> request_stream_callback_;
  // Engine actions.
  Array<EngineAction> actions_;
  // Event trace recorder.
  Optional<EventTraceRecorder> trace_recorder_;
};

Result<EngineCreationOutput> Engine::Create(const std::string& engine_config_json_str,
                                            Device device,
                                            Optional<PackedFunc> request_stream_callback,
                                            Optional<EventTraceRecorder> trace_recorder) {
  return EngineImpl::Create(engine_config_json_str, device, std::move(request_stream_callback),
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
  TVM_MODULE_VTABLE_ENTRY("get_default_generation_config",
                          &EngineModule::GetDefaultGenerationConfigJSONString);
  TVM_MODULE_VTABLE_END();

  /*! \brief Initialize the engine with config and other fields. */
  void Init(const std::string& engine_config_json_str, Device device,
            Optional<PackedFunc> request_stream_callback,
            Optional<EventTraceRecorder> trace_recorder) {
    Result<EngineCreationOutput> output_res =
        Engine::Create(engine_config_json_str, device, std::move(request_stream_callback),
                       std::move(trace_recorder));
    CHECK(output_res.IsOk()) << output_res.UnwrapErr();
    EngineCreationOutput output = output_res.Unwrap();
    this->engine_ = std::move(output.reloaded_engine);
    this->default_generation_cfg_json_str_ = output.default_generation_cfg->AsJSONString();
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
  /*! \brief Return the default generation config string. */
  String GetDefaultGenerationConfigJSONString() {
    CHECK(!default_generation_cfg_json_str_.empty())
        << "The default generation config has not been set.";
    return default_generation_cfg_json_str_;
  }

 private:
  Engine* GetEngine() {
    ICHECK(engine_ != nullptr) << "Engine is not initialized via init";
    return engine_.get();
  }

  std::unique_ptr<Engine> engine_ = nullptr;
  String default_generation_cfg_json_str_;
};

TVM_REGISTER_GLOBAL("mlc.serve.create_engine").set_body_typed(EngineModule::Create);

}  // namespace serve
}  // namespace llm
}  // namespace mlc
