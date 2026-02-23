/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/engine.cc
 * \brief The implementation for runtime module of serving engine module in MLC LLM.
 */
#include "engine.h"

#include <dlpack/dlpack.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/nvtx.h>
#include <tvm/runtime/threading_backend.h>
#include <xgrammar/xgrammar.h>

#include <cstdlib>
#include <functional>
#include <numeric>
#include <optional>
#include <tuple>
#include <unordered_set>

#include "../support/json_parser.h"
#include "../support/result.h"
#include "../support/utils.h"
#include "../tokenizers/tokenizers.h"
#include "engine_actions/action.h"
#include "engine_actions/action_commons.h"
#include "engine_state.h"
#include "event_trace_recorder.h"
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
using tvm::ffi::Function;

class EngineModule;

// get tokenizer info from model config
inline std::optional<TokenizerInfo> GetTokenizerInfo(const picojson::object& model_config) {
  if (model_config.count("tokenizer_info") == 0) {
    LOG(WARNING) << "Tokenizer info not found in mlc-chat-config.json. "
                 << "Trying to automatically detect the tokenizer info";
    return std::nullopt;
  }
  const picojson::object& tokenizer_info_obj =
      model_config.at("tokenizer_info").get<picojson::object>();
  auto info = tvm::ffi::make_object<TokenizerInfoNode>();
  if (tokenizer_info_obj.count("token_postproc_method")) {
    info->token_postproc_method = tokenizer_info_obj.at("token_postproc_method").get<std::string>();
  }
  if (tokenizer_info_obj.count("prepend_space_in_encode")) {
    info->prepend_space_in_encode = tokenizer_info_obj.at("prepend_space_in_encode").get<bool>();
  }
  if (tokenizer_info_obj.count("strip_space_in_decode")) {
    info->strip_space_in_decode = tokenizer_info_obj.at("strip_space_in_decode").get<bool>();
  }
  return TokenizerInfo(info);
}

inline std::pair<std::optional<std::string>, int> GetEnvSocketHostPort() {
  char* host_str = std::getenv("MLC_SOCKET_HOST");
  char* port_str = std::getenv("MLC_SOCKET_PORT");
  if (host_str == nullptr || port_str == nullptr) {
    return {std::nullopt, -1};
  }
  std::string host(host_str);
  if (host.empty()) {
    return {std::nullopt, -1};
  }
  return {host, std::atoi(port_str)};
}

// string back error node
void StreamBackErrorImpl(Request request, FRequestStreamCallback request_stream_callback,
                         String finish_reason) {
  // If the request input length exceeds the maximum allowed single sequence length,
  // invoke callback and do not process the request.
  Array<RequestStreamOutput> output{RequestStreamOutput(
      request->id, std::vector<std::vector<int64_t>>(request->generation_cfg->n), std::nullopt,
      std::vector<Optional<String>>(request->generation_cfg->n, finish_reason),
      std::vector<String>(request->generation_cfg->n))};
  // NOTE: Invariant requirement
  // always stream back final usage
  // otherwise frontend may have issues deciding
  String dummy_usage = ("{ \"prompt_tokens\": 0, \"completion_tokens\": 0, \"total_tokens\": 0 }");
  output.push_back(RequestStreamOutput::Usage(request->id, dummy_usage));
  if (request_stream_callback != nullptr) {
    request_stream_callback(output);
  }
}

void AbortRequestImpl(EngineState estate, const Array<Model>& models, const String& request_id,
                      String finish_reason) {
  auto it_rstate = estate->request_states.find(request_id);
  if (it_rstate == estate->request_states.end()) {
    // The request to abort does not exist.
    return;
  }

  RequestState rstate = it_rstate->second;
  Request request = rstate->entries[0]->request;

  // - Check if the request is running or pending.
  auto it_running = std::find(estate->running_queue.begin(), estate->running_queue.end(), request);
  auto it_waiting = std::find(estate->waiting_queue.begin(), estate->waiting_queue.end(), request);

  estate->request_states.erase(request->id);
  if (it_running != estate->running_queue.end()) {
    // The request to abort is in running queue
    estate->running_queue.erase(it_running);

    for (int i = static_cast<int>(rstate->entries.size()) - 1; i >= 0; --i) {
      if (estate->prefix_cache->HasSequence(rstate->entries[i]->mstates[0]->internal_id)) {
        estate->prefix_cache->RecycleSequence(rstate->entries[i]->mstates[0]->internal_id,
                                              /*lazy=*/false);
      } else {
        if (rstate->entries[i]->status != RequestStateStatus::kAlive) {
          estate->id_manager.RecycleId(rstate->entries[i]->mstates[0]->internal_id);
          continue;
        }
        RemoveRequestFromModel(estate, rstate->entries[i]->mstates[0]->internal_id, models);
        estate->id_manager.RecycleId(rstate->entries[i]->mstates[0]->internal_id);
      }
    }
  }
  if (it_waiting != estate->waiting_queue.end()) {
    // The request to abort is in waiting queue
    estate->waiting_queue.erase(it_waiting);
  }
  // Todo: abortion when the request is not in either queue?

  // Send a callback to notice the abortion.
  StreamBackErrorImpl(request, estate->request_stream_callback_, finish_reason);
  estate->running_rsentries_changed = true;
}

/*!
 *  \brief This a mock engine that always echo back the inputs
 *   and attaches the generation config to usage.extra
 *
 * \note: mock engine test cannot replace real engine test.
 *
 * It only tests that parameters are converted and
 * passed correctly to the backend.
 */
class MockEchoEngineImpl : public Engine {
 public:
  static Result<EngineCreationOutput> Create(const std::string& engine_config_json_str,
                                             FRequestStreamCallback request_stream_callback,
                                             const picojson::object& model_config) {
    using TResult = Result<EngineCreationOutput>;
    // set dummy values
    InferrableEngineConfig inferrable_config;
    inferrable_config.max_num_sequence = 32;
    inferrable_config.max_total_sequence_length = 32 * 4096;
    inferrable_config.max_single_sequence_length = 4096;
    inferrable_config.prefill_chunk_size = 1024;
    inferrable_config.max_history_size = 1024;
    picojson::value config_json;
    std::string err = picojson::parse(config_json, engine_config_json_str);
    if (!err.empty()) {
      return TResult::Error(err);
    }
    EngineConfig engine_config = EngineConfig::FromJSONAndInferredConfig(
        config_json.get<picojson::object>(), inferrable_config);

    auto n = std::make_unique<MockEchoEngineImpl>();
    n->request_stream_callback_ = request_stream_callback;
    n->tokenizer_ = Tokenizer::FromPath(engine_config->model, GetTokenizerInfo(model_config));
    // - Get the default generation config from the first model.
    GenerationConfig default_generation_cfg =
        GenerationConfig::GetDefaultFromModelConfig(model_config);
    return TResult::Ok({std::move(n), std::move(engine_config), std::move(default_generation_cfg)});
  }

  void Reset() final {}

  bool Empty() final { return request_map_.empty(); }

  void SetRequestStreamCallback(FRequestStreamCallback request_stream_callback) final {
    request_stream_callback_ = request_stream_callback;
  }

  FRequestStreamCallback GetRequestStreamCallback() final { return request_stream_callback_; }

  void AddRequest(Request request) final {
    // precompute the stream back results and store them in the request_map
    request = Request::FromUntokenized(request, tokenizer_);
    std::vector<RequestStreamOutput> outputs;
    int64_t completion_tokens = 0;
    int64_t prompt_tokens = 0;

    for (Data input : request->inputs) {
      // only stream back token data
      if (auto* token_data = input.as<TokenDataNode>()) {
        for (int64_t token_id : token_data->token_ids) {
          prompt_tokens += 1;
          completion_tokens += 1;
          if (request->generation_cfg->max_tokens == -1 ||
              completion_tokens <= request->generation_cfg->max_tokens) {
            outputs.push_back(RequestStreamOutput(
                request->id,
                std::vector<std::vector<int64_t>>(request->generation_cfg->n, {token_id}),
                std::nullopt,
                std::vector<Optional<String>>(request->generation_cfg->n, std::nullopt),
                std::vector<String>(request->generation_cfg->n)));
          }
        }
      }
    }

    // output go beyond max tokens
    String finish_reason = "stop";
    if (request->generation_cfg->max_tokens != -1 &&
        prompt_tokens > request->generation_cfg->max_tokens) {
      finish_reason = "length";
    }
    std::vector<std::vector<int64_t>> group_delta_token_ids;

    // correct the last output with right finish reason
    if (outputs.size() > 0) {
      group_delta_token_ids = outputs.back()->group_delta_token_ids;
      outputs.pop_back();
    }
    outputs.push_back(RequestStreamOutput(
        request->id, group_delta_token_ids, std::nullopt,
        std::vector<Optional<String>>(request->generation_cfg->n, finish_reason),
        std::vector<String>(request->generation_cfg->n)));

    // attach usage and config
    picojson::object usage;
    usage["prompt_tokens"] = picojson::value(static_cast<int64_t>(prompt_tokens));
    usage["completion_tokens"] =
        picojson::value(static_cast<int64_t>(completion_tokens * request->generation_cfg->n));
    usage["total_tokens"] = picojson::value(
        static_cast<int64_t>(prompt_tokens + completion_tokens * request->generation_cfg->n));
    usage["extra"] = picojson::value(request->generation_cfg->AsJSON());
    // NOTE: Invariant requirement
    // always stream back final usage
    // otherwise frontend may have issues deciding termination
    outputs.push_back(RequestStreamOutput::Usage(request->id, picojson::value(usage).serialize()));
    // reverse the stream back so we can just pop back and get out
    std::reverse(outputs.begin(), outputs.end());

    request_map_[request->id] = MockRequestState{request, std::move(outputs)};
  }

  void AbortRequest(const String& request_id) {
    auto it = request_map_.find(request_id);
    if (it == request_map_.end()) return;
    Request request = it->second.request;

    // If the request input length exceeds the maximum allowed single sequence length,
    // invoke callback and do not process the request.
    Array<RequestStreamOutput> output{RequestStreamOutput(
        request_id, std::vector<std::vector<int64_t>>(request->generation_cfg->n), std::nullopt,
        std::vector<Optional<String>>(request->generation_cfg->n, String("abort")),
        std::vector<String>(request->generation_cfg->n))};
    // NOTE: Invariant requirement
    // always stream back final usage
    // otherwise frontend may have issues deciding
    String dummy_usage =
        ("{ \"prompt_tokens\": 0, \"completion_tokens\": 0, \"total_tokens\": 0 }");
    output.push_back(RequestStreamOutput::Usage(request->id, dummy_usage));
    request_map_.erase(it);
    if (request_stream_callback_ != nullptr) {
      request_stream_callback_(output);
    }
  }

  void AbortAllRequests() final {
    // avoid deletion during iteraton
    std::vector<String> request_ids;
    for (const auto& kv : request_map_) {
      request_ids.push_back(kv.first);
    }
    for (String req_id : request_ids) {
      AbortRequest(req_id);
    }
  }

  void Step() final {
    Array<RequestStreamOutput> outputs;
    std::vector<String> finished_request_ids;
    for (auto& kv : request_map_) {
      MockRequestState& state = kv.second;
      TVM_FFI_ICHECK_GE(state.reversed_outputs.size(), 2);
      if (state.reversed_outputs.size() == 2) {
        outputs.push_back(state.reversed_outputs.back());
        state.reversed_outputs.pop_back();
        outputs.push_back(state.reversed_outputs.back());
        finished_request_ids.push_back(kv.first);
      } else {
        outputs.push_back(state.reversed_outputs.back());
        state.reversed_outputs.pop_back();
      }
    }
    for (String req_id : finished_request_ids) {
      request_map_.erase(req_id);
    }
    if (request_stream_callback_ != nullptr) {
      request_stream_callback_(outputs);
    }
  }

  /************** Debug/Profile **************/

  /*! \brief Internal engine metrics. */
  String JSONMetrics() final { return "{}"; }

  /*! \brief Call the given global function on all workers. Only for debug purpose. */
  void DebugCallFuncOnAllAllWorker(const String& func_name, Optional<String> func_args) final {}

 private:
  struct MockRequestState {
    Request request;
    std::vector<RequestStreamOutput> reversed_outputs;
  };

  // internal tokenizer
  // keep for future usage, in case we want to echo back the tokens
  Tokenizer tokenizer_;
  // callback stream
  FRequestStreamCallback request_stream_callback_;
  // active requests
  std::unordered_map<String, MockRequestState> request_map_;
};

/********************** Engine Impl **********************/

/*! \brief The implementation of Engine. */
class EngineImpl : public Engine {
  friend class EngineModule;

 public:
  /********************** Engine Management **********************/

  static Result<EngineCreationOutput> Create(const std::string& engine_config_json_str,
                                             DLDevice device,
                                             FRequestStreamCallback request_stream_callback,
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

    int num_model = models_and_model_libs.size();
    TVM_FFI_ICHECK_GE(num_model, 1);
    // - Initialize singleton states inside the engine.
    n->estate_->Reset();
    n->estate_->request_stream_callback_ = std::move(request_stream_callback);
    n->trace_recorder_ = trace_recorder;
    n->device_ = device;
    // - Load model config, create a shared disco session when tensor
    // parallelism is enabled.
    std::vector<std::string> model_libs;
    std::vector<picojson::object> model_configs;
    model_libs.reserve(num_model);
    model_configs.reserve(num_model);
    for (int i = 0; i < num_model; ++i) {
      const auto& [model_str, model_lib] = models_and_model_libs[i];
      Result<picojson::object> model_config_res = Model::LoadModelConfig(model_str);
      if (model_config_res.IsErr()) {
        return TResult::Error("Model " + std::to_string(i) +
                              " has invalid mlc-chat-config.json: " + model_config_res.UnwrapErr());
      }
      model_libs.push_back(model_lib);
      model_configs.push_back(model_config_res.Unwrap());
    }

    // kick in mock path so we don't have to load in models
    if (models_and_model_libs[0].second == "mock://echo") {
      return MockEchoEngineImpl::Create(engine_config_json_str,
                                        n->estate_->request_stream_callback_, model_configs[0]);
    }

    auto [session, num_shards, model_num_pipeline_stages] =
        n->CreateDiscoSession(model_libs, model_configs, device);

    // - Initialize each model independently.
    n->models_.clear();
    for (int i = 0; i < num_model; ++i) {
      const auto& [model_str, model_lib] = models_and_model_libs[i];
      Model model = Model::Create(model_lib, model_str, model_configs[i], device, session,
                                  num_shards, model_num_pipeline_stages[i],
                                  /*trace_enabled=*/trace_recorder.defined());
      n->models_.push_back(model);
    }
    // - Initialize NVSHMEM
    n->estate_->disaggregation = n->models_[0]->GetMetadata().disaggregation;
    if (n->estate_->disaggregation) {
      LOG(INFO) << "Intiailizing NVSHMEM";
      char* nvshmem_init_config_json_char = std::getenv("MLC_NVSHMEM_INIT_CONFIG_JSON_STR");
      CHECK(nvshmem_init_config_json_char != nullptr)
          << "The environment variables MLC_NVSHMEM_INIT_CONFIG_JSON_STR should be set.";
      std::string f_name = "runtime.disco.nvshmem.init_nvshmem_wrapper";
      if (session != nullptr) {
        n->DebugCallFuncOnAllAllWorker(f_name, String(nvshmem_init_config_json_char));
      } else {
        static Function func = Function::GetGlobalRequired(f_name);
        func(String(nvshmem_init_config_json_char));
      }
      LOG(INFO) << "NVSHMEM initialized successfully.";
    }

    // - Automatically infer the missing fields in EngineConfig JSON strings
    // and get the final EngineConfig.
    Result<EngineConfig> engine_config_res =
        n->AutoDecideEngineConfig(engine_config_json_str, model_configs);
    if (engine_config_res.IsErr()) {
      return TResult::Error(engine_config_res.UnwrapErr());
    }
    EngineConfig engine_config = engine_config_res.Unwrap();
    {
      if (engine_config->prefix_cache_mode == PrefixCacheMode::kRadix) {
        n->estate_->prefix_cache = PrefixCache::CreateRadixPrefixCache(
            static_cast<size_t>(engine_config->prefix_cache_max_num_recycling_seqs),
            [engine_ptr = n.get()](int64_t seq_id) {
              RemoveRequestFromModel(engine_ptr->estate_, seq_id, engine_ptr->models_);
              engine_ptr->estate_->id_manager.RecycleId(seq_id);
            });
      } else if (engine_config->prefix_cache_mode == PrefixCacheMode::kDisable) {
        n->estate_->prefix_cache = PrefixCache::CreateNoPrefixCache();
      } else {
        LOG(FATAL) << "Unsupported prefix cache mode: "
                   << static_cast<int>(engine_config->prefix_cache_mode);
      }
      if (engine_config->speculative_mode != SpeculativeMode::kDisable &&
          engine_config->prefill_mode == PrefillMode::kHybrid) {
        engine_config->prefill_mode = PrefillMode::kChunked;
        LOG(WARNING)
            << "Hybrid prefill mode fallbacks to chunked prefill, due to speculative mode is "
               "enabled and not implemented with hybrid prefill yet.";
      }
    }
    // - Load model weights, create KV cache and workspace.
    n->model_workspaces_.clear();
    for (const Model& model : n->models_) {
      model->LoadParams();
      model->SetMaxNumSequence(engine_config->max_num_sequence);
      model->SetPrefillChunkSize(engine_config->prefill_chunk_size);
      model->CreateKVCache(engine_config->kv_cache_page_size, engine_config->max_num_sequence,
                           engine_config->max_total_sequence_length,
                           engine_config->prefill_chunk_size, engine_config->max_history_size);
      n->model_workspaces_.push_back(
          ModelWorkspace{model->AllocEmbeddingTensor(), model->AllocHiddenStatesTensor()});
    }
    // - Initialize tokenizer and grammar
    n->tokenizer_ = Tokenizer::FromPath(engine_config->model, GetTokenizerInfo(model_configs[0]));
    n->token_table_ = n->tokenizer_->PostProcessedTokenTable();
    n->cached_grammar_compiler_ = xgrammar::CachedGrammarCompiler(n->token_table_);
    // - Create the logit processor and sampler, and
    // the DraftTokenWorkspaceManager for speculative decoding.
    int max_num_tokens = engine_config->max_num_sequence;
    DraftTokenWorkspaceManager draft_token_workspace_manager{nullptr};
    if (engine_config->speculative_mode != SpeculativeMode::kDisable) {
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
      n->estate_->spec_draft_length = engine_config->spec_draft_length;
    }
    n->actions_ =
        CreateEngineActions(n->models_, engine_config, model_configs, n->model_workspaces_,
                            logit_processor, sampler, draft_token_workspace_manager, n->tokenizer_,
                            n->trace_recorder_, n->estate_->request_stream_callback_, device);
    n->draft_token_workspace_manager_ = draft_token_workspace_manager;
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

  bool Empty() final { return estate_->running_queue.empty() && estate_->waiting_queue.empty(); }

  String JSONMetrics() final { return picojson::value(estate_->metrics.AsJSON()).serialize(true); }

  FRequestStreamCallback GetRequestStreamCallback() final {
    return estate_->request_stream_callback_;
  }

  void SetRequestStreamCallback(FRequestStreamCallback request_stream_callback) final {
    estate_->request_stream_callback_ = std::move(request_stream_callback);
  }

  // string back error node
  void StreamBackError(Request request, String finish_reason) {
    StreamBackErrorImpl(request, estate_->request_stream_callback_, finish_reason);
  }

  /***************** High-level Request Management *****************/

  void HandleSpecialRequests(Request request) {
    auto special_request = request->generation_cfg->debug_config.special_request;
    switch (special_request) {
      case SpecialRequestKind::kQueryEngineMetrics: {
        Array<RequestStreamOutput> output = {
            RequestStreamOutput::Usage(request->id, estate_->metrics.AsUsageJSONStr())};
        estate_->request_stream_callback_(output);
        break;
      }
      default:
        break;
    }
  }

  /*!
   * \brief Handle the given disaggregation request.
   * Return true if skipping the subsequent AddRequest process.
   */
  bool HandleDisaggRequest(Request request) {
    DisaggConfig disagg_config = request->generation_cfg->debug_config.disagg_config;
    DisaggRequestKind kind = disagg_config.kind;
    if (kind == DisaggRequestKind::kPrepareReceive) {
      // No-op.
      return false;
    } else if (kind == DisaggRequestKind::kRemoteSend) {
      int input_length = 0;
      for (Data input : request->inputs) {
        input_length += input->GetLength();
      }
      // - Truncate the inputs to the desired prefill length (specified by "end").
      int kv_window_begin = disagg_config.kv_window_begin.value_or(0);
      int kv_window_end = disagg_config.kv_window_end.value_or(input_length);
      CHECK_GE(kv_window_begin, 0);
      if (kv_window_end < 0) {
        kv_window_end = input_length + kv_window_end;
      }
      CHECK_LT(kv_window_end, input_length)
          << "Prefill the full input on the remote machine is not supported.";
      CHECK_LT(kv_window_begin, kv_window_end)
          << "\"begin >= end\" is not supported by remote prefill";
      request->inputs = SplitData(request->inputs, input_length, kv_window_end).first;
      // - Check the invariant: "end - begin" equals the expanded metadata length.
      CHECK_EQ(disagg_config.kv_append_metadata.size(), models_.size());
      for (const IntTuple& compressed_kv_append_metadata : disagg_config.kv_append_metadata) {
        TVM_FFI_ICHECK(!compressed_kv_append_metadata.empty());
        int num_segments = compressed_kv_append_metadata[0];
        TVM_FFI_ICHECK_EQ(compressed_kv_append_metadata.size(), num_segments * 2 + 1);
        int transmission_length = 0;
        for (int i = 0; i < num_segments; ++i) {
          transmission_length += compressed_kv_append_metadata[i * 2 + 2];
        }
        CHECK_EQ(transmission_length, kv_window_end - kv_window_begin);
      }
      // - Override the "n" in generation config to 1.
      ObjectPtr<GenerationConfigNode> updated_generation_cfg =
          tvm::ffi::make_object<GenerationConfigNode>(*request->generation_cfg.get());
      updated_generation_cfg->n = 1;
      request->generation_cfg = GenerationConfig(updated_generation_cfg);
      return false;
    } else if (kind == DisaggRequestKind::kStartGeneration) {
      auto it_rstate = estate_->request_states.find(request->id);
      CHECK(it_rstate != estate_->request_states.end());
      TVM_FFI_ICHECK(!it_rstate->second->entries.empty());
      request = it_rstate->second->entries[0]->request;
      CHECK(request->generation_cfg->debug_config.disagg_config.kind ==
            DisaggRequestKind::kPrepareReceive);
      int input_length = 0;
      for (Data input : request->inputs) {
        input_length += input->GetLength();
      }
      // - Truncate the inputs to the desired prefill length (specified by "end").
      int kv_window_begin = disagg_config.kv_window_begin.value_or(0);
      int kv_window_end = disagg_config.kv_window_end.value_or(input_length);
      CHECK_EQ(kv_window_end, input_length);
      if (kv_window_begin < 0) {
        kv_window_begin = input_length + kv_window_begin;
      }
      CHECK_GE(kv_window_begin, 0);
      CHECK_LT(kv_window_begin, input_length);
      // The request is not supposed to be in running queue nor waiting queue.
      auto it_running =
          std::find(estate_->running_queue.begin(), estate_->running_queue.end(), request);
      auto it_waiting =
          std::find(estate_->waiting_queue.begin(), estate_->waiting_queue.end(), request);
      CHECK(it_running == estate_->running_queue.end());
      CHECK(it_waiting == estate_->waiting_queue.end());

      RequestState rstate = it_rstate->second;
      ObjectPtr<GenerationConfigNode> updated_generation_cfg =
          tvm::ffi::make_object<GenerationConfigNode>(*request->generation_cfg.get());
      // - Split the input data into two parts at the position "kv_window_begin".
      CHECK(!request->inputs.empty());
      auto [lhs_data, rhs_data] = SplitData(request->inputs, input_length, kv_window_begin);
      if (input_length - kv_window_begin == 1 && request->generation_cfg->n == 1) {
        // - Commit the last token id to the request states.
        CHECK_EQ(rhs_data.size(), 1);
        const auto* token_data = rhs_data.back().as<TokenDataNode>();
        CHECK(token_data != nullptr);
        CHECK_EQ(token_data->GetLength(), 1);
        SampleResult last_token;
        last_token.sampled_token_id = {token_data->token_ids.back(), 1.0};
        for (RequestModelState mstate : rstate->entries[0]->mstates) {
          mstate->CommitToken(last_token);
          CHECK_EQ(mstate->committed_tokens.size(), 1);
        }
        // - Set "next_callback_token_pos" so that this token will not be streamed back to user.
        rstate->entries[0]->next_callback_token_pos = 1;
        // - Update the request input.
        request->inputs = lhs_data;
        // - Increment the max_tokens in generation config.
        if (request->generation_cfg->max_tokens != -1) {
          ++updated_generation_cfg->max_tokens;
        }
      } else {
        // Since there are multiple tokens to prefill, we add the remaining inputs
        // to the request's RequestModelStates for prefill.
        for (RequestModelState mstate : rstate->entries[0]->mstates) {
          mstate->inputs = rhs_data;
        }
        // Add to waiting queue for prefill.
        estate_->waiting_queue.insert(estate_->waiting_queue.begin(), request);
      }
      estate_->running_queue.push_back(request);
      // Erase the disaggregation request kind.
      updated_generation_cfg->debug_config.disagg_config.kind = DisaggRequestKind::kNone;
      request->generation_cfg = GenerationConfig(updated_generation_cfg);
      estate_->running_rsentries_changed = true;
      return true;
    }
    LOG(FATAL) << "Cannot reach here";
    throw;
  }

  void AddRequest(Request request) final {
    NVTXScopedRange nvtx_scope("Add request " + request->id);
    // special requests do not involve generation
    if (request->generation_cfg->debug_config.special_request != SpecialRequestKind::kNone) {
      this->HandleSpecialRequests(request);
      return;
    }

    RECORD_EVENT(trace_recorder_, request->id, "request added to engine");
    auto add_time_point = std::chrono::high_resolution_clock::now();

    // Get a request copy where all text inputs are tokenized.
    request = Request::FromUntokenized(request, tokenizer_);
    TVM_FFI_ICHECK_NE(request->prompt_tokens, -1);

    if (request->prompt_tokens >= engine_config_->max_single_sequence_length &&
        estate_->request_stream_callback_ != nullptr) {
      this->StreamBackError(request, "length");
      return;
    }

    // Handle disaggregation requests.
    if (request->generation_cfg->debug_config.disagg_config.kind != DisaggRequestKind::kNone) {
      bool return_now = this->HandleDisaggRequest(request);
      if (return_now) {
        return;
      }
    }

    // Append to the waiting queue and create the request state.
    estate_->waiting_queue.push_back(request);

    int n = request->generation_cfg->n;
    int rng_seed = request->generation_cfg->seed;
    auto compiled_grammar = GetGrammarFromResponseFormat(request->generation_cfg->response_format);

    std::vector<RequestStateEntry> rsentries;
    // Create the request state entry for the input.
    rsentries.emplace_back(request, models_.size(), estate_->id_manager.GetNewId(), rng_seed,
                           token_table_, compiled_grammar);
    if (n > 1) {
      // Then create a request state entry for each parallel generation branch.
      // We add a offset to the rng seed so that to make generations different.
      rsentries.reserve(n + 1);
      rsentries[0]->child_indices.reserve(n);
      for (int i = 0; i < n; ++i) {
        rsentries[0]->child_indices.push_back(rsentries.size());
        rsentries.emplace_back(request, models_.size(), estate_->id_manager.GetNewId(),
                               rng_seed + i + 1, token_table_, compiled_grammar,
                               /*parent_idx=*/0);
      }
    }
    RequestState rstate = RequestState(std::move(rsentries), n, add_time_point);
    for (const RequestStateEntry& rsentry : rstate->entries) {
      // Set the back reference.
      // note, we avoid cyclic reference and use raw ptr.
      rsentry->rstate = rstate.operator->();
    }
    request->rstate = rstate.operator->();
    estate_->request_states.emplace(request->id, rstate);
  }

  void AbortRequest(const String& request_id) final {
    AbortRequestImpl(estate_, models_, request_id);
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
    CHECK(estate_->request_stream_callback_ != nullptr)
        << "The request stream callback is not set. Engine cannot execute.";
    for (EngineAction action : actions_) {
      Array<Request> processed_requests;
      {
        NVTXScopedRange nvtx_scope("Action step");
        processed_requests = action->Step(estate_);
      }
      if (!processed_requests.empty()) {
        ActionStepPostProcess(processed_requests, estate_, models_, tokenizer_,
                              estate_->request_stream_callback_,
                              engine_config_->max_single_sequence_length,
                              draft_token_workspace_manager_, trace_recorder_);
        return;
      }
    }
    TVM_FFI_ICHECK(estate_->running_queue.empty())
        << "Internal assumption violated: It is expected that an engine step takes at least one "
           "action (e.g. prefill, decode, etc.) but it does not.";
  }

  /************** Utility Functions **************/
  std::tuple<Optional<Session>, int, std::vector<int>> CreateDiscoSession(
      const std::vector<std::string>& model_libs,
      const std::vector<picojson::object>& model_configs, Device device) {
    const auto& base_model_config = model_configs[0];

    auto f_get_num_shards_num_stages =
        [&device](const std::string& model_lib,
                  const picojson::object& model_config) -> std::pair<int, int> {
      if (!StartsWith(model_lib, "system://")) {
        Module executable = ffi::Module::LoadFromFile(model_lib);
        Optional<Function> fload_exec = executable->GetFunction("vm_load_executable");
        TVM_FFI_ICHECK(fload_exec.defined()) << "TVM runtime cannot find vm_load_executable";
        Module local_vm = fload_exec.value()().cast<Module>();
        local_vm->GetFunction("vm_initialization")
            .value()(static_cast<int>(device.device_type), device.device_id,
                     static_cast<int>(tvm::runtime::memory::AllocatorType::kPooled),
                     static_cast<int>(kDLCPU), 0,
                     static_cast<int>(tvm::runtime::memory::AllocatorType::kPooled));
        ModelMetadata metadata = ModelMetadata::FromModule(local_vm, std::move(model_config));
        return {metadata.tensor_parallel_shards, metadata.pipeline_parallel_stages};
      } else {
        return {1, 1};
      }
    };

    int num_shards = -1;
    int max_num_stages = 1;
    std::vector<int> model_num_pipeline_stages;
    model_num_pipeline_stages.reserve(model_libs.size());
    TVM_FFI_ICHECK_EQ(model_libs.size(), model_configs.size());
    for (int i = 0; i < static_cast<int>(model_libs.size()); ++i) {
      auto [model_num_shards, model_num_stages] =
          f_get_num_shards_num_stages(model_libs[i], model_configs[i]);
      model_num_pipeline_stages.push_back(model_num_stages);
      max_num_stages = std::max(max_num_stages, model_num_stages);
      if (i == 0) {
        num_shards = model_num_shards;
      } else {
        CHECK_EQ(model_num_shards, num_shards)
            << "Inconsistent tensor_parallel_shards values across models. Some model is compiled "
               "with tensor_parallel_shards "
            << num_shards << " and some other model is compiled with tensor_parallel_shards "
            << model_num_shards;
      }
    }

    Optional<Session> session = std::nullopt;
    int num_workers = num_shards * max_num_stages;
    if (num_workers > 1) {
#ifndef MLC_SINGLE_GPU_ONLY
      constexpr const char* f_create_process_pool = "runtime.disco.create_process_pool";
      if (!Function::GetGlobal(f_create_process_pool).has_value()) {
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
      std::vector<int64_t> device_ids(num_workers);
      for (int i = 0; i < num_workers; ++i) {
        // device.device_id is the start of the worker 0 of this model
        device_ids[i] = device.device_id + i;
      }
      const std::string& green_text_begin = "\033[92m";
      const std::string& yellow_text_begin = "\033[93m";
      const std::string& colored_text_end = "\033[0m";
      auto [socket_host, socket_port] = GetEnvSocketHostPort();
      if (max_num_stages > 1 && socket_host.has_value()) {
        // Use SocketSession when pipeline parallelism enabled and socket host and port are set.
        CHECK_GT(socket_port, 0)
            << "Invalid MLC socket port " << socket_port
            << ". Please set a valid port value in environment variable \"MLC_SOCKET_PORT\".";
        LOG(INFO) << "Creating MLC socket session with socket host " << socket_host.value()
                  << " and port " << socket_port;
        LOG(INFO) << "Please launch " << green_text_begin << max_num_stages - 1 << colored_text_end
                  << " remote socket node(s) with the following command to proceed:\n\t"
                  << green_text_begin << "python -m mlc_llm.cli.disco_remote_socket_session "
                  << (socket_host.value() == "0.0.0.0" ? "<YOUR_NODE_IP>" : socket_host.value())
                  << " " << socket_port << " " << num_shards << colored_text_end;
        static Function f_create_socket_sess =
            Function::GetGlobalRequired("runtime.disco.SocketSession");
        Session sess =
            f_create_socket_sess(max_num_stages, num_shards, /*num_groups=*/max_num_stages,
                                 socket_host.value(), socket_port)
                .cast<Session>();
        session = std::move(sess);
      } else {
        if (max_num_stages > 1) {
          LOG(INFO)
              << yellow_text_begin
              << "Model is enabled with \"pipeline_parallel_stages\" but the socket host/port is "
                 "not set. If you intend to run the model on multiple nodes, please set "
                 "environment variable \"MLC_SOCKET_HOST\" and \"MLC_SOCKET_PORT\" and run again."
              << colored_text_end;
        }
        // Use ProcessSession otherwise.
        session = Session::ProcessSession(num_workers, max_num_stages, f_create_process_pool,
                                          "mlc_llm.cli.worker");
      }
      session.value()->InitCCL(ccl, Shape(device_ids));
#else
      LOG(FATAL) << "MLC_SINGLE_GPU_ONLY is specified. Multi-GPU is not enabled.";
#endif  // MLC_SINGLE_GPU_ONLY
    }
    return {session, num_shards, model_num_pipeline_stages};
  }

  /************** Debug/Profile **************/

  void DebugCallFuncOnAllAllWorker(const String& func_name, Optional<String> func_args) final {
    CHECK(!models_.empty()) << "There is no model running in Engine.";
    models_[0]->DebugCallFuncOnAllAllWorker(func_name, func_args);
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
    ObjectPtr<EngineConfigNode> n = tvm::ffi::make_object<EngineConfigNode>();

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
    InferrableEngineConfig inferrable_cfg{max_num_sequence, max_total_sequence_length,
                                          max_single_sequence_length, prefill_chunk_size,
                                          max_history_size};

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
    Result<InferrableEngineConfig> inferrable_cfg_res;
    if (use_kv_cache.Unwrap()) {
      // - Infer configuration.
      inferrable_cfg_res = InferrableEngineConfig::InferForKVCache(
          mode, device_, gpu_memory_utilization, model_configs, model_metadata, inferrable_cfg,
          verbose);
    } else {
      // - Infer configuration.
      inferrable_cfg_res = InferrableEngineConfig::InferForRNNState(
          mode, device_, gpu_memory_utilization, model_configs, model_metadata, inferrable_cfg,
          verbose);
    }

    if (inferrable_cfg_res.IsErr()) {
      return TResult::Error(inferrable_cfg_res.UnwrapErr());
    }
    inferrable_cfg = inferrable_cfg_res.Unwrap();
    TVM_FFI_ICHECK(inferrable_cfg.max_num_sequence.has_value());
    TVM_FFI_ICHECK(inferrable_cfg.max_total_sequence_length.has_value());
    use_kv_cache = ModelsUseKVCache(model_configs);
    if (use_kv_cache.Unwrap()) {
      TVM_FFI_ICHECK(inferrable_cfg.max_single_sequence_length.has_value());
    }
    TVM_FFI_ICHECK(inferrable_cfg.prefill_chunk_size.has_value());
    TVM_FFI_ICHECK(inferrable_cfg.max_history_size.has_value());
    return TResult::Ok(EngineConfig::FromJSONAndInferredConfig(config, inferrable_cfg));
  }

  /*! \brief Set the maximum threading backend concurrency. */
  void SetThreadMaxConcurrency() {
    int host_cpu_usage = 1;
    for (Model model : models_) {
      host_cpu_usage += model->EstimateHostCPURequirement();
    }
    if (host_cpu_usage > 1) {
      int max_concurrency = tvm::runtime::threading::MaxConcurrency();
      tvm::runtime::threading::SetMaxConcurrency(std::min(
          std::max(max_concurrency - host_cpu_usage, 1), engine_config_->max_num_sequence));
    }
  }

  /*! \brief Create a grammar init context according to the response format. If the response format
   * is not JSON, return std::nullopt. */
  std::optional<xgrammar::CompiledGrammar> GetGrammarFromResponseFormat(
      const ResponseFormat& response_format) {
    if (response_format.type != "json_object") {
      return std::nullopt;
    } else if (!response_format.schema) {
      return cached_grammar_compiler_.GetCompiledGrammarForJSON();
    } else {
      return cached_grammar_compiler_.GetCompiledGrammarForJSONSchema(
          response_format.schema.value());
    }
  }

  // Engine state, managing requests and request states.
  EngineState estate_;
  // Configurations and singletons
  EngineConfig engine_config_;
  // internal tokenizer
  Tokenizer tokenizer_;
  std::vector<std::string> token_table_;
  // Cached grammar compiler for grammar matching.
  xgrammar::CachedGrammarCompiler cached_grammar_compiler_;
  // Models
  Array<Model> models_;
  // Device that the models run on.
  Device device_;
  // Workspace of each model.
  std::vector<ModelWorkspace> model_workspaces_;
  // Engine actions.
  Array<EngineAction> actions_;
  // Draft token workspace manager for speculative decoding.
  Optional<DraftTokenWorkspaceManager> draft_token_workspace_manager_;
  // Event trace recorder.
  Optional<EventTraceRecorder> trace_recorder_;
};

Result<EngineCreationOutput> Engine::Create(const std::string& engine_config_json_str,
                                            Device device,
                                            FRequestStreamCallback request_stream_callback,
                                            Optional<EventTraceRecorder> trace_recorder) {
  return EngineImpl::Create(engine_config_json_str, device, request_stream_callback,
                            std::move(trace_recorder));
}

/*! \brief Clear global memory manager */
void ClearGlobalMemoryManager() {
  static const char* kFunc = "vm.builtin.memory_manager.clear";
  static Function f = Function::GetGlobalRequired(kFunc);
  f();
}

class EngineModule : public ffi::ModuleObj {
 public:
  TVM_MODULE_VTABLE_BEGIN("mlc.serve.engine");
  TVM_MODULE_VTABLE_ENTRY("init", &EngineModule::Init);
  TVM_MODULE_VTABLE_ENTRY("add_request", &EngineModule::AddRequest);
  TVM_MODULE_VTABLE_ENTRY("create_request", &EngineModule::CreateRequest);
  TVM_MODULE_VTABLE_ENTRY("abort_request", &EngineModule::Abort);
  TVM_MODULE_VTABLE_ENTRY("step", &EngineModule::Step);
  TVM_MODULE_VTABLE_ENTRY("reset", &EngineModule::Reset);
  TVM_MODULE_VTABLE_ENTRY("json_metrics", &EngineModule::JSONMetrics);
  TVM_MODULE_VTABLE_ENTRY("get_request_stream_callback", &EngineModule::GetRequestStreamCallback);
  TVM_MODULE_VTABLE_ENTRY("set_request_stream_callback", &EngineModule::SetRequestStreamCallback);
  TVM_MODULE_VTABLE_END();

  /*! \brief Initialize the engine with config and other fields. */
  void Init(const std::string& engine_config_json_str, Device device,
            FRequestStreamCallback request_stream_callback,
            Optional<EventTraceRecorder> trace_recorder) {
    Result<EngineCreationOutput> output_res = Engine::Create(
        engine_config_json_str, device, request_stream_callback, std::move(trace_recorder));
    CHECK(output_res.IsOk()) << output_res.UnwrapErr();
    EngineCreationOutput output = output_res.Unwrap();
    this->engine_ = std::move(output.reloaded_engine);
    this->default_generation_config_ = output.default_generation_cfg;
  }
  /*! \brief Construct an EngineModule. */
  static ffi::Module Create() { return ffi::Module(tvm::ffi::make_object<EngineModule>()); }
  /*! \brief Redirection to `Engine::AddRequest`. */
  void AddRequest(Request request) { return GetEngine()->AddRequest(std::move(request)); }
  /*! \brief Redirection to `Engine::AbortRequest`. */
  void Abort(const String& request_id) { return GetEngine()->AbortRequest(request_id); }
  /*! \brief Create request with given arguments and the engine default generation config. */
  Request CreateRequest(String id, Array<Data> inputs, String generation_cfg_json_str) {
    auto config = json::ParseToJSONObject(generation_cfg_json_str);
    auto gen_config = GenerationConfig::FromJSON(config, default_generation_config_);
    CHECK(gen_config.IsOk()) << gen_config.UnwrapErr();
    return Request(std::move(id), std::move(inputs), gen_config.Unwrap());
  }
  /*! \brief Redirection to `Engine::Step`. */
  void Step() { return GetEngine()->Step(); }
  /*! \brief Redirection to `Engine::GetRequestStreamCallback`. */
  FRequestStreamCallback GetRequestStreamCallback() {
    return GetEngine()->GetRequestStreamCallback();
  }
  /*! \brief Redirection to `Engine::SetRequestStreamCallback` */
  void SetRequestStreamCallback(FRequestStreamCallback request_stream_callback) {
    GetEngine()->SetRequestStreamCallback(request_stream_callback);
  }
  /*! \brief Redirection to `Engine::Reset`. */
  void Reset() { return GetEngine()->Reset(); }

  /*! \brief Redirection to `Engine::JSONMetrics`. */
  String JSONMetrics() { return GetEngine()->JSONMetrics(); }

 private:
  Engine* GetEngine() {
    TVM_FFI_ICHECK(engine_ != nullptr) << "Engine is not initialized via init";
    return engine_.get();
  }

  std::unique_ptr<Engine> engine_ = nullptr;
  GenerationConfig default_generation_config_;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("mlc.serve.create_engine", EngineModule::Create);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
