/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/config.cc
 */
#include "config.h"

#include <picojson.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/device_api.h>

#include <limits>
#include <random>

#include "../json_ffi/openai_api_protocol.h"
#include "../support/json_parser.h"
#include "../support/utils.h"
#include "data.h"

namespace mlc {
namespace llm {
namespace serve {

TVM_FFI_STATIC_INIT_BLOCK() {
  GenerationConfigNode::RegisterReflection();
  EngineConfigNode::RegisterReflection();
}

uint64_t TotalDetectGlobalMemory(DLDevice device) {
  // Get single-card GPU size.
  tvm::ffi::Any rv;
  DeviceAPI::Get(device)->GetAttr(device, DeviceAttrKind::kTotalGlobalMemory, &rv);
  int64_t gpu_size_bytes = rv.cast<int64_t>();
  // Since the memory size returned by the OpenCL runtime is smaller than the actual available
  // memory space, we set a best available space so that MLC LLM can run 7B or 8B models on Android
  // with OpenCL.
  if (device.device_type == kDLOpenCL) {
    int64_t min_size_bytes = 5LL * 1024 * 1024 * 1024;  //  Minimum size is 5 GB
    gpu_size_bytes = std::max(gpu_size_bytes, min_size_bytes);
  }
  return gpu_size_bytes;
}

/****************** ResponseFormat ******************/

Result<ResponseFormat> ResponseFormat::FromJSON(const picojson::object& config) {
  using TResult = Result<ResponseFormat>;
  ResponseFormat res;
  res.type = json::LookupOrDefault<std::string>(config, "type", "text");

  std::optional<std::string> schema = json::LookupOptional<std::string>(config, "schema");
  if (schema.has_value()) {
    res.schema = schema.value();
  }

  if (res.type != "text" && res.type != "function" && res.type != "json_object") {
    return TResult::Error("Uknonwn response_format type " + res.type);
  }

  return TResult::Ok(res);
}

picojson::object ResponseFormat::AsJSON() const {
  picojson::object config;
  config["type"] = picojson::value(type);
  if (schema.has_value()) {
    config["schema"] = picojson::value(schema.value().operator std::string());
  }
  return config;
}

/****************** DisaggConfig ******************/

Result<DisaggConfig> DisaggConfig::FromJSON(const picojson::object& config) {
  using TResult = Result<DisaggConfig>;
  DisaggConfig res;
  std::optional<std::string> kind = json::LookupOptional<std::string>(config, "kind");
  if (kind.has_value()) {
    if (kind.value() == "prepare_receive") {
      res.kind = DisaggRequestKind::kPrepareReceive;
    } else if (kind.value() == "remote_send") {
      res.kind = DisaggRequestKind::kRemoteSend;
    } else if (kind.value() == "start_generation") {
      res.kind = DisaggRequestKind::kStartGeneration;
    } else {
      return TResult::Error("Unknown disaggregation request kind " + kind.value());
    }
  }
  std::optional<std::string> kv_append_metadata_encoded =
      json::LookupOptional<std::string>(config, "kv_append_metadata");
  if (kv_append_metadata_encoded.has_value()) {
    picojson::value parse_result;
    std::string err =
        picojson::parse(parse_result, Base64Decode(kv_append_metadata_encoded.value()));
    if (!err.empty()) {
      return TResult::Error("kv_append_metadata parse error: " + err);
    }
    if (!parse_result.is<picojson::array>()) {
      return TResult::Error("kv_append_metadata is not array of integer.");
    }
    picojson::array kv_append_metadata_arr = parse_result.get<picojson::array>();
    std::vector<IntTuple> kv_append_metadata;
    int ptr = 0;
    while (ptr < static_cast<int>(kv_append_metadata_arr.size())) {
      if (!kv_append_metadata_arr[ptr].is<int64_t>()) {
        return TResult::Error("Invalid kv append metadata value in kv_append_metadata array");
      }
      int num_segments = kv_append_metadata_arr[ptr].get<int64_t>();
      if (ptr + num_segments * 2 + 1 > static_cast<int>(kv_append_metadata_arr.size())) {
        return TResult::Error("Invalid kv append metadata compression in kv_append_metadata");
      }
      std::vector<int64_t> compressed_kv_append_metadata{num_segments};
      compressed_kv_append_metadata.reserve(num_segments * 2 + 1);
      for (int i = 1; i <= num_segments * 2; ++i) {
        if (!kv_append_metadata_arr[ptr + i].is<int64_t>()) {
          return TResult::Error("Invalid kv append metadata value in kv_append_metadata array");
        }
        compressed_kv_append_metadata.push_back(kv_append_metadata_arr[ptr + i].get<int64_t>());
      }
      kv_append_metadata.push_back(IntTuple(std::move(compressed_kv_append_metadata)));
      ptr += num_segments * 2 + 1;
    }
    res.kv_append_metadata = std::move(kv_append_metadata);
  }
  res.kv_window_begin = json::LookupOptional<int64_t>(config, "kv_window_begin");
  res.kv_window_end = json::LookupOptional<int64_t>(config, "kv_window_end");
  res.dst_group_offset = json::LookupOptional<int64_t>(config, "dst_group_offset");
  return TResult::Ok(res);
}

picojson::object DisaggConfig::AsJSON() const {
  picojson::object config;
  switch (kind) {
    case DisaggRequestKind::kPrepareReceive: {
      config["kind"] = picojson::value("prepare_receive");
      break;
    }
    case DisaggRequestKind::kRemoteSend: {
      config["kind"] = picojson::value("remote_send");
      break;
    }
    case DisaggRequestKind::kStartGeneration: {
      config["kind"] = picojson::value("start_generation");
      break;
    }
    default:
      break;
  }
  if (!kv_append_metadata.empty()) {
    picojson::array kv_append_metadata_arr;
    for (const IntTuple& compressed_kv_append_metadata : kv_append_metadata) {
      for (int64_t value : compressed_kv_append_metadata) {
        kv_append_metadata_arr.push_back(picojson::value(value));
      }
    }
    config["kv_append_metadata"] =
        picojson::value(Base64Encode(picojson::value(kv_append_metadata_arr).serialize()));
  }
  if (kv_window_begin.has_value()) {
    config["kv_window_begin"] = picojson::value(static_cast<int64_t>(kv_window_begin.value()));
  }
  if (kv_window_end.has_value()) {
    config["kv_window_end"] = picojson::value(static_cast<int64_t>(kv_window_end.value()));
  }
  if (dst_group_offset.has_value()) {
    config["dst_group_offset"] = picojson::value(static_cast<int64_t>(dst_group_offset.value()));
  }
  return config;
}

/****************** DebugConfig ******************/

Result<DebugConfig> DebugConfig::FromJSON(const picojson::object& config) {
  using TResult = Result<DebugConfig>;
  DebugConfig res;
  res.ignore_eos = json::LookupOrDefault<bool>(config, "ignore_eos", false);
  res.pinned_system_prompt = json::LookupOrDefault<bool>(config, "pinned_system_prompt", false);
  std::string special_request = json::LookupOrDefault<std::string>(config, "special_request", "");
  if (special_request.length() != 0) {
    if (special_request == "query_engine_metrics") {
      res.special_request = SpecialRequestKind::kQueryEngineMetrics;
    } else {
      return TResult::Error("Unknown special request " + special_request);
    }
  }
  std::string grammar_execution_mode =
      json::LookupOrDefault<std::string>(config, "grammar_execution_mode", "jump_forward");
  if (grammar_execution_mode == "jump_forward") {
    res.grammar_execution_mode = GrammarExecutionMode::kJumpForward;
  } else if (grammar_execution_mode == "constraint") {
    res.grammar_execution_mode = GrammarExecutionMode::kConstraint;
  } else {
    return TResult::Error("Unknown grammar execution mode " + grammar_execution_mode);
  }
  if (auto disagg_config_obj = json::LookupOptional<picojson::object>(config, "disagg_config")) {
    Result<DisaggConfig> disagg_config = DisaggConfig::FromJSON(disagg_config_obj.value());
    if (disagg_config.IsErr()) {
      return TResult::Error(disagg_config.UnwrapErr());
    }
    res.disagg_config = disagg_config.Unwrap();
  }
  return TResult::Ok(res);
}

/**
 * \return serialized json value of the config.
 */
picojson::object DebugConfig::AsJSON() const {
  picojson::object config;
  config["ignore_eos"] = picojson::value(ignore_eos);
  config["pinned_system_prompt"] = picojson::value(pinned_system_prompt);
  switch (special_request) {
    case SpecialRequestKind::kQueryEngineMetrics: {
      config["special_request"] = picojson::value("query_engine_metrics");
      break;
    }
    case SpecialRequestKind::kNone:
      break;
  }
  switch (grammar_execution_mode) {
    case GrammarExecutionMode::kJumpForward: {
      config["grammar_execution_mode"] = picojson::value("jump_forward");
      break;
    }
    case GrammarExecutionMode::kConstraint: {
      config["grammar_execution_mode"] = picojson::value("constraint");
      break;
    }
  }
  if (disagg_config.kind != DisaggRequestKind::kNone) {
    config["disagg_config"] = picojson::value(disagg_config.AsJSON());
  }
  return config;
}

/****************** GenerationConfig ******************/

Result<GenerationConfig> GenerationConfig::Validate(GenerationConfig cfg) {
  using TResult = Result<GenerationConfig>;
  if (cfg->n <= 0) {
    return TResult::Error("\"n\" should be at least 1");
  }
  if (cfg->temperature < 0) {
    return TResult::Error("\"temperature\" should be non-negative");
  }
  if (cfg->top_p < 0 || cfg->top_p > 1) {
    return TResult::Error("\"top_p\" should be in range [0, 1]");
  }
  if (std::fabs(cfg->frequency_penalty) > 2.0) {
    return TResult::Error("frequency_penalty must be in [-2, 2]!");
  }
  if (cfg->repetition_penalty <= 0) {
    return TResult::Error("\"repetition_penalty\" must be positive");
  }
  if (cfg->top_logprobs < 0 || cfg->top_logprobs > 20) {
    return TResult::Error("At most 20 top logprob tokens are supported");
  }
  if (cfg->top_logprobs != 0 && !(cfg->logprobs)) {
    return TResult::Error("\"logprobs\" must be true to support \"top_logprobs\"");
  }
  for (const auto& item : cfg->logit_bias) {
    double bias_value = item.second;
    if (std::fabs(bias_value) > 100.0) {
      return TResult::Error("Logit bias value should be in range [-100, 100].");
    }
  }
  return TResult::Ok(cfg);
}

Result<GenerationConfig> GenerationConfig::FromJSON(const picojson::object& config,
                                                    const GenerationConfig& default_config) {
  using TResult = Result<GenerationConfig>;
  ObjectPtr<GenerationConfigNode> n = tvm::ffi::make_object<GenerationConfigNode>();
  n->n = json::LookupOrDefault<int64_t>(config, "n", default_config->n);
  n->temperature =
      json::LookupOrDefault<double>(config, "temperature", default_config->temperature);
  n->top_p = json::LookupOrDefault<double>(config, "top_p", default_config->top_p);
  n->frequency_penalty =
      json::LookupOrDefault<double>(config, "frequency_penalty", default_config->frequency_penalty);
  n->presence_penalty =
      json::LookupOrDefault<double>(config, "presence_penalty", default_config->presence_penalty);
  n->repetition_penalty = json::LookupOrDefault<double>(config, "repetition_penalty",
                                                        default_config->repetition_penalty);
  n->logprobs = json::LookupOrDefault<bool>(config, "logprobs", default_config->logprobs);
  n->top_logprobs =
      json::LookupOrDefault<int64_t>(config, "top_logprobs", default_config->top_logprobs);

  std::optional<picojson::object> logit_bias_obj =
      json::LookupOptional<picojson::object>(config, "logit_bias");
  if (logit_bias_obj.has_value()) {
    std::vector<std::pair<int, float>> logit_bias;
    logit_bias.reserve(logit_bias_obj.value().size());
    for (auto [token_id_str, bias] : logit_bias_obj.value()) {
      CHECK(bias.is<double>());
      double bias_value = bias.get<double>();
      logit_bias.emplace_back(std::stoi(token_id_str), bias_value);
    }
    n->logit_bias = std::move(logit_bias);
  } else {
    n->logit_bias = default_config->logit_bias;
  }

  n->seed = json::LookupOrDefault<int64_t>(config, "seed", std::random_device{}());
  // "-1" means the generation will not stop until exceeding
  // model capability or hit any stop criteria.
  n->max_tokens = json::LookupOrDefault<int64_t>(config, "max_tokens", -1);

  std::optional<picojson::array> stop_strs_arr =
      json::LookupOptional<picojson::array>(config, "stop_strs");
  if (stop_strs_arr.has_value()) {
    Array<String> stop_strs;
    stop_strs.reserve(stop_strs_arr.value().size());
    for (const picojson::value& v : stop_strs_arr.value()) {
      if (!v.is<std::string>()) {
        return TResult::Error("Invalid stop string in stop_strs");
      }
      stop_strs.push_back(v.get<std::string>());
    }
    n->stop_strs = std::move(stop_strs);
  } else {
    n->stop_strs = default_config->stop_strs;
  }
  std::optional<picojson::array> stop_token_ids_arr =
      json::LookupOptional<picojson::array>(config, "stop_token_ids");
  if (stop_token_ids_arr.has_value()) {
    std::vector<int> stop_token_ids;
    stop_token_ids.reserve(stop_token_ids_arr.value().size());
    for (const picojson::value& v : stop_token_ids_arr.value()) {
      if (!v.is<int64_t>()) {
        return TResult::Error("Invalid stop token in stop_token_ids");
      }
      stop_token_ids.push_back(v.get<int64_t>());
    }
    n->stop_token_ids = std::move(stop_token_ids);
  } else {
    n->stop_token_ids = default_config->stop_token_ids;
  }

  std::optional<picojson::object> response_format_obj =
      json::LookupOptional<picojson::object>(config, "response_format");
  if (response_format_obj.has_value()) {
    Result<ResponseFormat> response_format_res =
        ResponseFormat::FromJSON(response_format_obj.value());
    if (response_format_res.IsErr()) {
      return TResult::Error(response_format_res.UnwrapErr());
    }
    n->response_format = response_format_res.Unwrap();
  } else {
    n->response_format = default_config->response_format;
  }
  // "debug_config" is for internal usage. Not the part of OpenAI API spec.
  std::optional<picojson::object> debug_config_obj =
      json::LookupOptional<picojson::object>(config, "debug_config");

  if (debug_config_obj.has_value()) {
    Result<DebugConfig> debug_config_res = DebugConfig::FromJSON(debug_config_obj.value());
    if (debug_config_res.IsErr()) {
      return TResult::Error(debug_config_res.UnwrapErr());
    }
    n->debug_config = debug_config_res.Unwrap();
  }
  return Validate(GenerationConfig(n));
}

GenerationConfig GenerationConfig::GetDefaultFromModelConfig(
    const picojson::object& model_config_json) {
  ObjectPtr<GenerationConfigNode> n = tvm::ffi::make_object<GenerationConfigNode>();
  n->max_tokens = -1;
  n->temperature = json::LookupOrDefault<double>(model_config_json, "temperature", n->temperature);
  n->top_p = json::LookupOrDefault<double>(model_config_json, "top_p", n->top_p);
  n->frequency_penalty =
      json::LookupOrDefault<double>(model_config_json, "frequency_penalty", n->frequency_penalty);
  n->presence_penalty =
      json::LookupOrDefault<double>(model_config_json, "presence_penalty", n->presence_penalty);
  return GenerationConfig(n);
}

picojson::object GenerationConfigNode::AsJSON() const {
  picojson::object config;
  config["n"] = picojson::value(static_cast<int64_t>(this->n));
  config["temperature"] = picojson::value(this->temperature);
  config["top_p"] = picojson::value(this->top_p);
  config["frequency_penalty"] = picojson::value(this->frequency_penalty);
  config["presence_penalty"] = picojson::value(this->presence_penalty);
  config["repetition_penalty"] = picojson::value(this->repetition_penalty);
  config["logprobs"] = picojson::value(this->logprobs);
  config["top_logprobs"] = picojson::value(static_cast<int64_t>(this->top_logprobs));
  config["max_tokens"] = picojson::value(static_cast<int64_t>(this->max_tokens));
  config["seed"] = picojson::value(static_cast<int64_t>(this->seed));

  picojson::object logit_bias_obj;
  for (auto [token_id, bias] : logit_bias) {
    logit_bias_obj[std::to_string(token_id)] = picojson::value(static_cast<double>(bias));
  }
  config["logit_bias"] = picojson::value(logit_bias_obj);

  picojson::array stop_strs_arr;
  for (String stop_str : this->stop_strs) {
    stop_strs_arr.push_back(picojson::value(stop_str));
  }
  config["stop_strs"] = picojson::value(stop_strs_arr);

  picojson::array stop_token_ids_arr;
  for (int stop_token_id : this->stop_token_ids) {
    stop_token_ids_arr.push_back(picojson::value(static_cast<int64_t>(stop_token_id)));
  }
  config["stop_token_ids"] = picojson::value(stop_token_ids_arr);

  picojson::object response_format;
  response_format["type"] = picojson::value(this->response_format.type);
  response_format["schema"] = this->response_format.schema
                                  ? picojson::value(this->response_format.schema.value())
                                  : picojson::value();
  config["response_format"] = picojson::value(response_format);
  config["debug_config"] = picojson::value(debug_config.AsJSON());
  return config;
}

/****************** EngineConfig ******************/

EngineConfig EngineConfig::FromJSONAndInferredConfig(
    const picojson::object& json, const InferrableEngineConfig& inferred_config) {
  CHECK(inferred_config.max_num_sequence.has_value());
  CHECK(inferred_config.max_total_sequence_length.has_value());
  CHECK(inferred_config.prefill_chunk_size.has_value());
  CHECK(inferred_config.max_history_size.has_value());
  ObjectPtr<EngineConfigNode> n = tvm::ffi::make_object<EngineConfigNode>();

  // - Get models and model libs.
  n->model = json::Lookup<std::string>(json, "model");
  n->model_lib = json::Lookup<std::string>(json, "model_lib");
  std::vector<String> additional_models;
  std::vector<String> additional_model_libs;
  picojson::array additional_models_arr =
      json::LookupOrDefault<picojson::array>(json, "additional_models", picojson::array());
  int num_additional_models = additional_models_arr.size();
  additional_models.reserve(num_additional_models);
  additional_model_libs.reserve(num_additional_models);
  for (int i = 0; i < num_additional_models; ++i) {
    picojson::array additional_model_pair = json::Lookup<picojson::array>(additional_models_arr, i);
    additional_models.push_back(json::Lookup<std::string>(additional_model_pair, 0));
    additional_model_libs.push_back(json::Lookup<std::string>(additional_model_pair, 1));
  }
  n->additional_models = additional_models;
  n->additional_model_libs = additional_model_libs;
  n->mode = EngineModeFromString(json::Lookup<std::string>(json, "mode"));

  // - Other fields with default value.
  n->gpu_memory_utilization =
      json::LookupOrDefault<double>(json, "gpu_memory_utilization", n->gpu_memory_utilization);
  n->kv_cache_page_size =
      json::LookupOrDefault<int64_t>(json, "kv_cache_page_size", n->kv_cache_page_size);
  n->speculative_mode = SpeculativeModeFromString(json::LookupOrDefault<std::string>(
      json, "speculative_mode", SpeculativeModeToString(n->speculative_mode)));
  n->spec_draft_length =
      json::LookupOrDefault<int64_t>(json, "spec_draft_length", n->spec_draft_length);
  n->spec_tree_width = json::LookupOrDefault<int64_t>(json, "spec_tree_width", n->spec_tree_width);
  n->prefill_mode = PrefillModeFromString(json::LookupOrDefault<std::string>(
      json, "prefill_mode", PrefillModeToString(n->prefill_mode)));
  n->verbose = json::LookupOrDefault<bool>(json, "verbose", n->verbose);

  // - Fields from the inferred engine config.
  n->max_num_sequence = inferred_config.max_num_sequence.value();
  n->max_total_sequence_length = inferred_config.max_total_sequence_length.value();
  if (inferred_config.max_single_sequence_length.has_value()) {
    n->max_single_sequence_length = inferred_config.max_single_sequence_length.value();
  }
  n->prefill_chunk_size = inferred_config.prefill_chunk_size.value();
  n->max_history_size = inferred_config.max_history_size.value();

  n->prefix_cache_mode = PrefixCacheModeFromString(json::LookupOrDefault<std::string>(
      json, "prefix_cache_mode", PrefixCacheModeToString(n->prefix_cache_mode)));
  n->prefix_cache_max_num_recycling_seqs = json::LookupOrDefault<int64_t>(
      json, "prefix_cache_max_num_recycling_seqs", n->max_num_sequence);
  return EngineConfig(n);
}

Result<std::vector<std::pair<std::string, std::string>>>
EngineConfig::GetModelsAndModelLibsFromJSONString(const std::string& json_str) {
  using TResult = Result<std::vector<std::pair<std::string, std::string>>>;
  picojson::value config_json;
  std::string err = picojson::parse(config_json, json_str);
  if (!err.empty()) {
    return TResult::Error(err);
  }

  // Get the models and model libs from JSON.
  picojson::object config = config_json.get<picojson::object>();
  String model = json::Lookup<std::string>(config, "model");
  String model_lib = json::Lookup<std::string>(config, "model_lib");
  picojson::array additional_models_arr =
      json::LookupOrDefault<picojson::array>(config, "additional_models", picojson::array());

  int num_additional_models = additional_models_arr.size();
  std::vector<std::pair<std::string, std::string>> models_and_model_libs;
  models_and_model_libs.reserve(num_additional_models + 1);
  models_and_model_libs.emplace_back(model, model_lib);
  for (int i = 0; i < num_additional_models; ++i) {
    picojson::array additional_model_pair = json::Lookup<picojson::array>(additional_models_arr, i);
    models_and_model_libs.emplace_back(json::Lookup<std::string>(additional_model_pair, 0),
                                       json::Lookup<std::string>(additional_model_pair, 1));
  }
  return TResult::Ok(models_and_model_libs);
}

String EngineConfigNode::AsJSONString() const {
  picojson::object config;

  // - Models and model libs
  config["model"] = picojson::value(this->model);
  config["model_lib"] = picojson::value(this->model_lib);
  picojson::array additional_models_arr;
  additional_models_arr.reserve(this->additional_models.size());
  for (int i = 0; i < static_cast<int>(this->additional_models.size()); ++i) {
    additional_models_arr.push_back(
        picojson::value(picojson::array{picojson::value(this->additional_models[i]),
                                        picojson::value(this->additional_model_libs[i])}));
  }
  config["additional_models"] = picojson::value(additional_models_arr);

  // - Other fields
  config["mode"] = picojson::value(EngineModeToString(this->mode));
  config["gpu_memory_utilization"] = picojson::value(this->gpu_memory_utilization);
  config["kv_cache_page_size"] = picojson::value(static_cast<int64_t>(this->kv_cache_page_size));
  config["max_num_sequence"] = picojson::value(static_cast<int64_t>(this->max_num_sequence));
  config["max_total_sequence_length"] =
      picojson::value(static_cast<int64_t>(this->max_total_sequence_length));
  config["max_single_sequence_length"] =
      picojson::value(static_cast<int64_t>(this->max_single_sequence_length));
  config["prefill_chunk_size"] = picojson::value(static_cast<int64_t>(this->prefill_chunk_size));
  config["max_history_size"] = picojson::value(static_cast<int64_t>(this->max_history_size));
  config["prefix_cache_mode"] = picojson::value(PrefixCacheModeToString(this->prefix_cache_mode));
  config["prefix_cache_max_num_recycling_seqs"] =
      picojson::value(static_cast<int64_t>(this->prefix_cache_max_num_recycling_seqs));
  config["speculative_mode"] = picojson::value(SpeculativeModeToString(this->speculative_mode));
  config["spec_draft_length"] = picojson::value(static_cast<int64_t>(this->spec_draft_length));
  config["prefill_mode"] = picojson::value(PrefillModeToString(this->prefill_mode));
  config["verbose"] = picojson::value(static_cast<bool>(this->verbose));

  return picojson::value(config).serialize(true);
}

/****************** InferrableEngineConfig ******************/

/*! \brief The class for config limitation from models. */
struct ModelConfigLimits {
  int64_t model_compile_time_max_single_sequence_length;
  int64_t model_runtime_max_single_sequence_length;
  int64_t model_compile_time_max_prefill_chunk_size;
  int64_t model_runtime_max_prefill_chunk_size;
  int64_t model_max_sliding_window_size;
  int64_t model_max_batch_size;
};

/*! \brief Convert the bytes to megabytes, keeping 3 decimals. */
inline std::string BytesToMegabytesString(double bytes) {
  std::ostringstream os;
  os << std::setprecision(3) << std::fixed << (bytes / 1024 / 1024);
  return os.str();
}

/*!
 * \brief Get the upper bound of single sequence length, prefill size and batch size
 * from model config.
 */
Result<ModelConfigLimits> GetModelConfigLimits(const std::vector<picojson::object>& model_configs,
                                               const std::vector<ModelMetadata>& model_metadata) {
  TVM_FFI_ICHECK_EQ(model_configs.size(), model_metadata.size());
  int64_t model_compile_time_max_single_sequence_length = std::numeric_limits<int64_t>::max();
  int64_t model_runtime_max_single_sequence_length = std::numeric_limits<int64_t>::max();
  int64_t model_compile_time_max_prefill_chunk_size = std::numeric_limits<int64_t>::max();
  int64_t model_runtime_max_prefill_chunk_size = std::numeric_limits<int64_t>::max();
  int64_t model_max_batch_size = std::numeric_limits<int64_t>::max();
  int64_t model_max_sliding_window_size = std::numeric_limits<int64_t>::max();
  for (int i = 0; i < static_cast<int>(model_configs.size()); ++i) {
    // - The maximum single sequence length is the minimum context window size among all models.
    int64_t runtime_context_window_size =
        json::LookupOptional<int64_t>(model_configs[i], "context_window_size").value_or(-1);
    int64_t compile_time_context_window_size = model_metadata[i].context_window_size;

    // limit runtime setting by compile time setting
    if (compile_time_context_window_size != -1) {
      if (runtime_context_window_size == -1 ||
          runtime_context_window_size > compile_time_context_window_size) {
        runtime_context_window_size = compile_time_context_window_size;
      }
    }

    if (compile_time_context_window_size != -1) {
      model_compile_time_max_single_sequence_length =
          std::min(model_compile_time_max_single_sequence_length, compile_time_context_window_size);
    }
    if (runtime_context_window_size != -1) {
      model_runtime_max_single_sequence_length =
          std::min(model_runtime_max_single_sequence_length, runtime_context_window_size);
    }
    // - The maximum prefill chunk size is the minimum prefill chunk size among all models.
    int64_t runtime_prefill_chunk_size =
        json::Lookup<int64_t>(model_configs[i], "prefill_chunk_size");
    int64_t compile_time_prefill_chunk_size = model_metadata[i].prefill_chunk_size;

    // limit runtime setting by compile time setting
    if (compile_time_prefill_chunk_size != -1) {
      if (runtime_prefill_chunk_size == -1 ||
          runtime_prefill_chunk_size > compile_time_prefill_chunk_size) {
        runtime_prefill_chunk_size = compile_time_prefill_chunk_size;
      }
    }

    if (compile_time_prefill_chunk_size != -1) {
      model_compile_time_max_prefill_chunk_size =
          std::min(model_compile_time_max_prefill_chunk_size, compile_time_prefill_chunk_size);
    }
    if (runtime_prefill_chunk_size != -1) {
      model_runtime_max_prefill_chunk_size =
          std::min(model_runtime_max_prefill_chunk_size, runtime_prefill_chunk_size);
    }
    // - The maximum batch size is the minimum max batch size among all models.
    model_max_batch_size = std::min(model_max_batch_size, model_metadata[i].max_batch_size);
    // - The maximum sliding window size is the minimum among all models.
    int64_t runtime_sliding_window_size =
        json::LookupOptional<int64_t>(model_configs[i], "sliding_window_size").value_or(-1);
    if (runtime_sliding_window_size != -1) {
      model_max_sliding_window_size =
          std::min(model_max_sliding_window_size, runtime_sliding_window_size);
    }
  }
  TVM_FFI_ICHECK_NE(model_compile_time_max_prefill_chunk_size, std::numeric_limits<int64_t>::max());
  TVM_FFI_ICHECK_NE(model_runtime_max_prefill_chunk_size, std::numeric_limits<int64_t>::max());
  TVM_FFI_ICHECK_NE(model_max_batch_size, std::numeric_limits<int64_t>::max());
  TVM_FFI_ICHECK_GT(model_compile_time_max_prefill_chunk_size, 0);
  TVM_FFI_ICHECK_GT(model_runtime_max_prefill_chunk_size, 0);
  TVM_FFI_ICHECK_GT(model_max_batch_size, 0);
  return Result<ModelConfigLimits>::Ok(
      {model_compile_time_max_single_sequence_length, model_runtime_max_single_sequence_length,
       model_compile_time_max_prefill_chunk_size, model_runtime_max_prefill_chunk_size,
       model_max_sliding_window_size, model_max_batch_size});
}

/*! \brief The class for memory usage estimation result. */
struct MemUsageEstimationResult {
  double total_memory_bytes;
  double kv_cache_memory_bytes;
  double temp_memory_bytes;
  InferrableEngineConfig inferred_config;
};

Result<MemUsageEstimationResult> EstimateMemoryUsageOnMode(
    EngineMode mode, Device device, double gpu_memory_utilization, int64_t params_bytes,
    int64_t temp_buffer_bytes,
    const std::vector<picojson::object>& model_configs,  //
    const std::vector<ModelMetadata>& model_metadata,    //
    ModelConfigLimits model_config_limits,               //
    InferrableEngineConfig init_config, bool verbose) {
  std::ostringstream os;
  InferrableEngineConfig inferred_config = init_config;
  // - 1. max_num_sequence
  if (!init_config.max_num_sequence.has_value()) {
    if (mode == EngineMode::kLocal) {
      inferred_config.max_num_sequence =
          std::min(static_cast<int64_t>(4), model_config_limits.model_max_batch_size);
    } else if (mode == EngineMode::kInteractive) {
      inferred_config.max_num_sequence = 1;
    } else {
      inferred_config.max_num_sequence = model_config_limits.model_max_batch_size;
    }
    os << "max batch size will be set to " << inferred_config.max_num_sequence.value() << ", ";
  } else {
    os << "max batch size " << inferred_config.max_num_sequence.value()
       << " is specified by user, ";
  }
  int64_t max_num_sequence = inferred_config.max_num_sequence.value();
  // - 2. max_single_sequence_length
  if (!init_config.max_single_sequence_length.has_value()) {
    inferred_config.max_single_sequence_length =
        model_config_limits.model_runtime_max_single_sequence_length;
  } else {
    inferred_config.max_single_sequence_length =
        std::min(inferred_config.max_single_sequence_length.value(),
                 model_config_limits.model_compile_time_max_single_sequence_length);
  }
  // - 3. infer the maximum total sequence length that can fit GPU memory.
  double kv_bytes_per_token = 0;
  double kv_aux_workspace_bytes = 0;
  double model_workspace_bytes = 0;
  double logit_processor_workspace_bytes = 0;
  TVM_FFI_ICHECK_EQ(model_configs.size(), model_metadata.size());
  int num_models = model_configs.size();
  for (int i = 0; i < num_models; ++i) {
    // - Read the vocab size and compile-time prefill chunk size (which affects memory allocation).
    picojson::object compile_time_model_config =
        json::Lookup<picojson::object>(model_configs[i], "model_config");
    int64_t vocab_size = json::Lookup<int64_t>(compile_time_model_config, "vocab_size");
    int64_t prefill_chunk_size =
        json::Lookup<int64_t>(compile_time_model_config, "prefill_chunk_size");
    // - Calculate KV cache memory usage.
    int64_t num_layers = model_metadata[i].kv_cache_metadata.num_hidden_layers;
    int64_t head_dim = model_metadata[i].kv_cache_metadata.head_dim;
    int64_t num_qo_heads = model_metadata[i].kv_cache_metadata.num_attention_heads;
    int64_t num_kv_heads = model_metadata[i].kv_cache_metadata.num_key_value_heads;
    int64_t hidden_size = head_dim * num_qo_heads;
    kv_bytes_per_token +=
        head_dim * num_kv_heads * (num_layers / model_metadata[i].pipeline_parallel_stages) * 4 +
        1.25;
    kv_aux_workspace_bytes +=
        (max_num_sequence + 1) * 88 + prefill_chunk_size * (num_qo_heads + 1) * 8 +
        prefill_chunk_size * head_dim * (num_qo_heads + num_kv_heads) * 4 + 48 * 1024 * 1024;
    model_workspace_bytes += prefill_chunk_size * 4 + max_num_sequence * 4 +
                             (prefill_chunk_size * 2 + max_num_sequence) * hidden_size * 2;
    logit_processor_workspace_bytes +=
        max_num_sequence * 20 + max_num_sequence * vocab_size * 16.125;
  }
  int64_t gpu_size_bytes = TotalDetectGlobalMemory(device);
  // Compute the maximum total sequence length under the GPU memory budget.
  int64_t model_max_total_sequence_length =
      static_cast<int>((gpu_size_bytes * gpu_memory_utilization  //
                        - params_bytes                           //
                        - temp_buffer_bytes                      //
                        - kv_aux_workspace_bytes                 //
                        - model_workspace_bytes                  //
                        - logit_processor_workspace_bytes) /
                       kv_bytes_per_token);
  if (model_max_total_sequence_length <= 0) {
    if (verbose) {
      LOG(INFO) << "temp_buffer = " << BytesToMegabytesString(temp_buffer_bytes);
      LOG(INFO) << "kv_aux workspace = " << BytesToMegabytesString(kv_aux_workspace_bytes);
      LOG(INFO) << "model workspace = " << BytesToMegabytesString(model_workspace_bytes);
      LOG(INFO) << "logit processor workspace = "
                << BytesToMegabytesString(logit_processor_workspace_bytes);
    }
    return Result<MemUsageEstimationResult>::Error(
        "Insufficient GPU memory error: "
        "The available single GPU memory is " +
        BytesToMegabytesString(gpu_size_bytes * gpu_memory_utilization) +
        " MB, "
        "which is less than the sum of model weight size (" +
        BytesToMegabytesString(params_bytes) + " MB) and temporary buffer size (" +
        BytesToMegabytesString(temp_buffer_bytes + kv_aux_workspace_bytes + model_workspace_bytes +
                               logit_processor_workspace_bytes) +
        " MB).\n"
        "1. You can set a larger \"gpu_memory_utilization\" value.\n"
        "2. If the model weight size is too large, please enable tensor parallelism by passing "
        "`--tensor-parallel-shards $NGPU` to `mlc_llm gen_config` or use quantization.\n"
        "3. If the temporary buffer size is too large, please use a smaller `--prefill-chunk-size` "
        "in `mlc_llm gen_config`.");
  }
  if (device.device_type == DLDeviceType::kDLMetal) {
    // NOTE: Metal runtime has severe performance issues with large buffers.
    // To work around the issue, we limit the KV cache capacity to 32768.
    model_max_total_sequence_length =
        std::min(model_max_total_sequence_length, static_cast<int64_t>(32768));
  }
  // Compute the total memory usage except the KV cache part.
  double total_mem_usage_except_kv_cache =
      (params_bytes + temp_buffer_bytes + kv_aux_workspace_bytes + model_workspace_bytes +
       logit_processor_workspace_bytes);

  // - 4. max_total_sequence_length
  if (!init_config.max_total_sequence_length.has_value()) {
    if (mode == EngineMode::kLocal) {
      inferred_config.max_total_sequence_length = std::min(
          {model_max_total_sequence_length, inferred_config.max_single_sequence_length.value(),
           static_cast<int64_t>(8192)});
    } else if (mode == EngineMode::kInteractive) {
      inferred_config.max_total_sequence_length = std::min(
          {model_max_total_sequence_length, inferred_config.max_single_sequence_length.value()});
    } else {
      inferred_config.max_total_sequence_length =
          inferred_config.max_single_sequence_length.value() == std::numeric_limits<int64_t>::max()
              ? model_max_total_sequence_length
              : std::min(model_max_total_sequence_length,
                         max_num_sequence * inferred_config.max_single_sequence_length.value());
    }
    os << "max KV cache token capacity will be set to "
       << inferred_config.max_total_sequence_length.value() << ", ";
  } else {
    os << "max KV cache token capacity " << inferred_config.max_total_sequence_length.value()
       << " is specified by user, ";
  }
  // - 5. prefill_chunk_size
  if (!init_config.prefill_chunk_size.has_value()) {
    if (mode == EngineMode::kLocal || mode == EngineMode::kInteractive) {
      inferred_config.prefill_chunk_size =
          std::min({model_config_limits.model_runtime_max_prefill_chunk_size,
                    inferred_config.max_total_sequence_length.value(),
                    inferred_config.max_single_sequence_length.value()});
    } else {
      inferred_config.prefill_chunk_size = model_config_limits.model_runtime_max_prefill_chunk_size;
    }
    os << "prefill chunk size will be set to " << inferred_config.prefill_chunk_size.value()
       << ". ";
  } else {
    os << "prefill chunk size " << inferred_config.prefill_chunk_size.value()
       << " is specified by user. ";
  }

  // - Print logging message
  if (verbose) {
    LOG(INFO) << "Under mode \"" << EngineModeToString(mode) << "\", " << os.str();
  }

  return Result<MemUsageEstimationResult>::Ok(
      {total_mem_usage_except_kv_cache +
           inferred_config.max_total_sequence_length.value() * kv_bytes_per_token,
       kv_bytes_per_token * inferred_config.max_total_sequence_length.value() +
           kv_aux_workspace_bytes,
       model_workspace_bytes + logit_processor_workspace_bytes + temp_buffer_bytes,
       inferred_config});
}

Result<InferrableEngineConfig> InferrableEngineConfig::InferForKVCache(
    EngineMode mode, Device device, double gpu_memory_utilization,
    const std::vector<picojson::object>& model_configs,
    const std::vector<ModelMetadata>& model_metadata, InferrableEngineConfig init_config,
    bool verbose) {
  // - Check if max_history_size is not set.
  if (init_config.max_history_size.has_value() && init_config.max_history_size.value() != 0) {
    return Result<InferrableEngineConfig>::Error(
        "KV cache does not support max_history_size, while it is set to " +
        std::to_string(init_config.max_history_size.value()) + " in the input EngineConfig");
  }
  // - Get the upper bound of single sequence length, prefill size and batch size
  // from model config.
  Result<ModelConfigLimits> model_config_limits_res =
      GetModelConfigLimits(model_configs, model_metadata);

  if (model_config_limits_res.IsErr()) {
    return Result<InferrableEngineConfig>::Error(model_config_limits_res.UnwrapErr());
  }
  ModelConfigLimits model_config_limits = model_config_limits_res.Unwrap();
  // - Get total model parameter size and temporary in-function buffer
  // size in bytes on single GPU.
  int64_t params_bytes = 0;
  int64_t temp_buffer_bytes = 0;
  for (const ModelMetadata& metadata : model_metadata) {
    for (const ModelMetadata::Param& param : metadata.params) {
      int64_t param_size = param.dtype.bytes();
      for (int64_t v : param.shape) {
        TVM_FFI_ICHECK_GE(v, 0);
        param_size *= v;
      }
      params_bytes += param_size;
    }
    params_bytes /= metadata.pipeline_parallel_stages;
    for (const auto& [func_name, temp_buffer_size] : metadata.memory_usage) {
      temp_buffer_bytes = std::max(temp_buffer_bytes, temp_buffer_size);
    }
  }
  // Magnify the temp buffer by a factor of 2 for safety.
  temp_buffer_bytes *= 2;

  // - Infer the engine config and estimate memory usage for each mode.
  Result<MemUsageEstimationResult> local_mode_estimation_result = EstimateMemoryUsageOnMode(
      EngineMode::kLocal, device, gpu_memory_utilization, params_bytes, temp_buffer_bytes,
      model_configs, model_metadata, model_config_limits, init_config, verbose);
  Result<MemUsageEstimationResult> interactive_mode_estimation_result = EstimateMemoryUsageOnMode(
      EngineMode::kInteractive, device, gpu_memory_utilization, params_bytes, temp_buffer_bytes,
      model_configs, model_metadata, model_config_limits, init_config, verbose);
  Result<MemUsageEstimationResult> server_mode_estimation_result = EstimateMemoryUsageOnMode(
      EngineMode::kServer, device, gpu_memory_utilization, params_bytes, temp_buffer_bytes,
      model_configs, model_metadata, model_config_limits, init_config, verbose);
  // - Pick the estimation result according to the mode.
  std::string mode_name;
  Result<MemUsageEstimationResult> final_estimation_result;
  if (mode == EngineMode::kLocal) {
    final_estimation_result = std::move(local_mode_estimation_result);
  } else if (mode == EngineMode::kInteractive) {
    final_estimation_result = std::move(interactive_mode_estimation_result);
  } else {
    final_estimation_result = std::move(server_mode_estimation_result);
  }
  if (final_estimation_result.IsErr()) {
    return Result<InferrableEngineConfig>::Error(final_estimation_result.UnwrapErr());
  }
  // - Print log message.
  MemUsageEstimationResult final_estimation = final_estimation_result.Unwrap();
  InferrableEngineConfig inferred_config = std::move(final_estimation.inferred_config);

  if (verbose) {
    LOG(INFO) << "The actual engine mode is \"" << EngineModeToString(mode)
              << "\". So max batch size is " << inferred_config.max_num_sequence.value()
              << ", max KV cache token capacity is "
              << inferred_config.max_total_sequence_length.value() << ", prefill chunk size is "
              << inferred_config.prefill_chunk_size.value() << ".";
    LOG(INFO) << "Estimated total single GPU memory usage: "
              << BytesToMegabytesString(final_estimation.total_memory_bytes)
              << " MB (Parameters: " << BytesToMegabytesString(params_bytes)
              << " MB. KVCache: " << BytesToMegabytesString(final_estimation.kv_cache_memory_bytes)
              << " MB. Temporary buffer: "
              << BytesToMegabytesString(final_estimation.temp_memory_bytes)
              << " MB). The actual usage might be slightly larger than the estimated number.";
  }

  inferred_config.max_history_size = 0;
  return Result<InferrableEngineConfig>::Ok(inferred_config);
}

Result<InferrableEngineConfig> InferrableEngineConfig::InferForRNNState(
    EngineMode mode, Device device, double gpu_memory_utilization,
    const std::vector<picojson::object>& model_configs,
    const std::vector<ModelMetadata>& model_metadata, InferrableEngineConfig init_config,
    bool verbose) {
  // - Check max_single_sequence_length is not set.
  if (init_config.max_single_sequence_length.has_value()) {
    return Result<InferrableEngineConfig>::Error(
        "RNN state does not support max_single_sequence_length, while it is set to " +
        std::to_string(init_config.max_single_sequence_length.value()) +
        " in the input EngineConfig");
  }
  // - Get the upper bound of single sequence length, prefill size and batch size
  // from model config.
  Result<ModelConfigLimits> model_config_limits_res =
      GetModelConfigLimits(model_configs, model_metadata);
  if (model_config_limits_res.IsErr()) {
    return Result<InferrableEngineConfig>::Error(model_config_limits_res.UnwrapErr());
  }
  ModelConfigLimits model_config_limits = model_config_limits_res.Unwrap();

  std::ostringstream os;
  InferrableEngineConfig inferred_config = init_config;
  // - 1. prefill_chunk_size
  if (!init_config.prefill_chunk_size.has_value()) {
    inferred_config.prefill_chunk_size = std::min(
        model_config_limits.model_runtime_max_prefill_chunk_size, static_cast<int64_t>(4096));
    os << "prefill chunk size will be set to " << inferred_config.prefill_chunk_size.value()
       << ", ";
  } else {
    os << "prefill chunk size " << inferred_config.prefill_chunk_size.value()
       << " is specified by user, ";
  }
  // - 2. max_batch_size
  if (!init_config.max_num_sequence.has_value()) {
    inferred_config.max_num_sequence =
        mode == EngineMode::kInteractive
            ? 1
            : std::min(static_cast<int64_t>(4), model_config_limits.model_max_batch_size);
    os << "max batch size will be set to " << inferred_config.max_num_sequence.value() << ", ";
  } else {
    os << "max batch size " << inferred_config.max_num_sequence.value()
       << " is specified by user, ";
  }
  int64_t max_num_sequence = inferred_config.max_num_sequence.value();
  // - 3. max_total_sequence_length
  if (!init_config.max_total_sequence_length.has_value()) {
    inferred_config.max_total_sequence_length = 32768;
    os << "max RNN state token capacity will be set to "
       << inferred_config.max_total_sequence_length.value() << ". ";
  } else {
    os << "max RNN state token capacity " << inferred_config.max_total_sequence_length.value()
       << " is specified by user. ";
  }

  // - Extra logging message
  if (mode == EngineMode::kLocal) {
    os << "We choose small max batch size and RNN state capacity to use less GPU memory.";
  } else if (mode == EngineMode::kInteractive) {
    os << "We fix max batch size to 1 for interactive single sequence use.";
  } else {
    os << "We use as much GPU memory as possible (within the limit of gpu_memory_utilization).";
  }
  if (verbose) {
    LOG(INFO) << "Under mode \"" << EngineModeToString(mode) << "\", " << os.str();
  }

  // - Get total model parameter size and temporary in-function buffer
  // size in bytes on single GPU.
  int64_t params_bytes = 0;
  int64_t temp_buffer_bytes = 0;
  for (const ModelMetadata& metadata : model_metadata) {
    for (const ModelMetadata::Param& param : metadata.params) {
      int64_t param_size = param.dtype.bytes();
      for (int64_t v : param.shape) {
        TVM_FFI_ICHECK_GE(v, 0);
        param_size *= v;
      }
      params_bytes += param_size;
    }
    for (const auto& [func_name, temp_buffer_size] : metadata.memory_usage) {
      temp_buffer_bytes += temp_buffer_size;
    }
  }
  // - 4. max_history_size
  double rnn_state_base_bytes = 0;  // The memory usage for rnn state when history = 1.
  double model_workspace_bytes = 0;
  double logit_processor_workspace_bytes = 0;
  TVM_FFI_ICHECK_EQ(model_configs.size(), model_metadata.size());
  int num_models = model_configs.size();
  for (int i = 0; i < num_models; ++i) {
    // - Read the vocab size and compile-time prefill chunk size (which affects memory allocation).
    picojson::object compile_time_model_config =
        json::Lookup<picojson::object>(model_configs[i], "model_config");
    int64_t vocab_size = json::Lookup<int64_t>(compile_time_model_config, "vocab_size");
    int64_t prefill_chunk_size =
        json::Lookup<int64_t>(compile_time_model_config, "prefill_chunk_size");
    int64_t head_size = json::Lookup<int64_t>(compile_time_model_config, "head_size");
    int64_t num_heads = json::Lookup<int64_t>(compile_time_model_config, "num_heads");
    int64_t num_layers = json::Lookup<int64_t>(compile_time_model_config, "num_hidden_layers");
    int64_t hidden_size = json::Lookup<int64_t>(compile_time_model_config, "hidden_size");
    // - Calculate RNN state memory usage.
    rnn_state_base_bytes += (max_num_sequence * hidden_size * num_layers * 2 * 2 +
                             max_num_sequence * num_heads * head_size * head_size * num_layers * 2);
    model_workspace_bytes += prefill_chunk_size * 4 + max_num_sequence * 4 +
                             (prefill_chunk_size * 2 + max_num_sequence) * hidden_size * 2;
    logit_processor_workspace_bytes +=
        max_num_sequence * 20 + max_num_sequence * vocab_size * 16.125;
  }
  int64_t gpu_size_bytes = TotalDetectGlobalMemory(device);
  // Compute the maximum history size length under the GPU memory budget.
  int64_t model_max_history_size = static_cast<int>((gpu_size_bytes * gpu_memory_utilization  //
                                                     - params_bytes                           //
                                                     - temp_buffer_bytes                      //
                                                     - model_workspace_bytes                  //
                                                     - logit_processor_workspace_bytes) /
                                                    rnn_state_base_bytes);
  if (model_max_history_size <= 0) {
    return Result<InferrableEngineConfig>::Error(
        "Insufficient GPU memory error: "
        "The available single GPU memory is " +
        BytesToMegabytesString(gpu_size_bytes * gpu_memory_utilization) +
        " MB, "
        "which is less than the sum of model weight size (" +
        BytesToMegabytesString(params_bytes) + " MB) and temporary buffer size (" +
        BytesToMegabytesString(
            (temp_buffer_bytes + model_workspace_bytes + logit_processor_workspace_bytes)) +
        " MB). "
        "If the model weight size is too large, please use quantization. "
        "If the temporary buffer size is too large, please use a smaller `--prefill-chunk-size` in "
        "`mlc_llm gen_config`.");
  }
  if (!init_config.max_history_size.has_value()) {
    inferred_config.max_history_size = model_max_history_size;
  } else {
    inferred_config.max_history_size =
        std::min(inferred_config.max_history_size.value(), model_max_history_size);
  }
  if (verbose) {
    LOG(INFO) << "The actual engine mode is \"" << EngineModeToString(mode)
              << "\". So max batch size is " << inferred_config.max_num_sequence.value()
              << ", max RNN state token capacity is "
              << inferred_config.max_total_sequence_length.value() << ", prefill chunk size is "
              << inferred_config.prefill_chunk_size.value() << ".";
    LOG(INFO) << "Estimated total single GPU memory usage: "
              << BytesToMegabytesString(params_bytes + temp_buffer_bytes +
                                        inferred_config.max_history_size.value() *
                                            rnn_state_base_bytes)
              << " MB (Parameters: " << BytesToMegabytesString(params_bytes) << " MB. RNN state: "
              << BytesToMegabytesString(inferred_config.max_history_size.value() *
                                        rnn_state_base_bytes)
              << " MB. Temporary buffer: "
              << BytesToMegabytesString(model_workspace_bytes + logit_processor_workspace_bytes +
                                        temp_buffer_bytes)
              << " MB). The actual usage might be slightly larger than the estimated number.";
  }

  return Result<InferrableEngineConfig>::Ok(inferred_config);
}

/****************** Config utils ******************/

Result<bool> ModelsUseKVCache(const std::vector<picojson::object>& model_configs) {
  TVM_FFI_ICHECK_GE(model_configs.size(), 1);
  std::string model_type = json::Lookup<std::string>(model_configs[0], "model_type");
  bool use_kv_cache = model_type.find("rwkv") == std::string::npos;
  for (int i = 1; i < static_cast<int>(model_configs.size()); ++i) {
    if ((json::Lookup<std::string>(model_configs[i], "model_type").find("rwkv") ==
         std::string::npos) != use_kv_cache) {
      return Result<bool>::Error(
          "Invalid models in EngineConfig. Models must be all RNN model or none model is RNN "
          "model.");
    }
  }
  return Result<bool>::Ok(use_kv_cache);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
