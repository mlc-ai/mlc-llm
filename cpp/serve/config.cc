/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/config.cc
 */
#include "config.h"

#include <picojson.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include <limits>
#include <random>

#include "../json_ffi/openai_api_protocol.h"
#include "../support/json_parser.h"
#include "data.h"

namespace mlc {
namespace llm {
namespace serve {

/****************** GenerationConfig ******************/

TVM_REGISTER_OBJECT_TYPE(GenerationConfigNode);

GenerationConfig::GenerationConfig(
    std::optional<int> n, std::optional<double> temperature, std::optional<double> top_p,
    std::optional<double> frequency_penalty, std::optional<double> presense_penalty,
    std::optional<double> repetition_penalty, std::optional<bool> logprobs,
    std::optional<int> top_logprobs, std::optional<std::vector<std::pair<int, float>>> logit_bias,
    std::optional<int> seed, std::optional<bool> ignore_eos, std::optional<int> max_tokens,
    std::optional<Array<String>> stop_strs, std::optional<std::vector<int>> stop_token_ids,
    std::optional<ResponseFormat> response_format, std::optional<bool> pinned,
    Optional<String> default_config_json_str) {
  ObjectPtr<GenerationConfigNode> obj = make_object<GenerationConfigNode>();
  GenerationConfig default_config;
  if (default_config_json_str.defined()) {
    default_config = GenerationConfig(default_config_json_str.value(), NullOpt);
  } else {
    default_config = GenerationConfig(obj);
  }

  obj->n = n.value_or(default_config->n);
  CHECK_GT(obj->n, 0) << "\"n\" should be at least 1";
  obj->temperature = temperature.value_or(default_config->temperature);
  CHECK_GE(obj->temperature, 0) << "\"temperature\" should be non-negative";
  obj->top_p = top_p.value_or(default_config->top_p);
  CHECK(obj->top_p >= 0 && obj->top_p <= 1) << "\"top_p\" should be in range [0, 1]";
  obj->frequency_penalty = frequency_penalty.value_or(default_config->frequency_penalty);
  CHECK(std::fabs(obj->frequency_penalty) <= 2.0) << "Frequency penalty must be in [-2, 2]!";
  obj->presence_penalty = presense_penalty.value_or(default_config->presence_penalty);
  CHECK(std::fabs(obj->presence_penalty) <= 2.0) << "Presence penalty must be in [-2, 2]!";
  obj->repetition_penalty = repetition_penalty.value_or(default_config->repetition_penalty);
  CHECK(obj->repetition_penalty > 0) << "Repetition penalty must be a positive number!";
  obj->logprobs = logprobs.value_or(default_config->logprobs);
  obj->top_logprobs = top_logprobs.value_or(default_config->top_logprobs);
  CHECK(obj->top_logprobs >= 0 && obj->top_logprobs <= 5)
      << "At most 5 top logprob tokens are supported";
  CHECK(obj->top_logprobs == 0 || obj->logprobs)
      << "\"logprobs\" must be true to support \"top_logprobs\"";

  obj->logit_bias = logit_bias.value_or(default_config->logit_bias);
  for (auto [token_id_str, bias] : obj->logit_bias) {
    CHECK_LE(std::fabs(bias), 100.0) << "Logit bias value should be in range [-100, 100].";
  }

  obj->seed = seed.value_or(std::random_device{}());
  // "ignore_eos" is for benchmarking. Not the part of OpenAI API spec.
  obj->ignore_eos = ignore_eos.value_or(default_config->ignore_eos);
  // "-1" means the generation will not stop until exceeding
  // model capability or hit any stop criteria.
  obj->max_tokens = max_tokens.value_or(-1);

  obj->stop_strs = stop_strs.value_or(default_config->stop_strs);
  obj->stop_token_ids = stop_token_ids.value_or(default_config->stop_token_ids);
  obj->response_format = response_format.value_or(default_config->response_format);
  // "pinned" is for internal usage. Not the part of OpenAI API spec.
  obj->pinned = pinned.value_or(default_config->pinned);

  data_ = std::move(obj);
}

GenerationConfig::GenerationConfig(String config_json_str,
                                   Optional<String> default_config_json_str) {
  picojson::object config = json::ParseToJSONObject(config_json_str);
  ObjectPtr<GenerationConfigNode> n = make_object<GenerationConfigNode>();
  GenerationConfig default_config;
  if (default_config_json_str.defined()) {
    default_config = GenerationConfig(default_config_json_str.value(), NullOpt);
  } else {
    default_config = GenerationConfig(n);
  }

  n->n = json::LookupOrDefault<int64_t>(config, "n", default_config->n);
  CHECK_GT(n->n, 0) << "\"n\" should be at least 1";
  n->temperature =
      json::LookupOrDefault<double>(config, "temperature", default_config->temperature);
  CHECK_GE(n->temperature, 0) << "\"temperature\" should be non-negative";
  n->top_p = json::LookupOrDefault<double>(config, "top_p", default_config->top_p);
  CHECK(n->top_p >= 0 && n->top_p <= 1) << "\"top_p\" should be in range [0, 1]";
  n->frequency_penalty =
      json::LookupOrDefault<double>(config, "frequency_penalty", default_config->frequency_penalty);
  CHECK(std::fabs(n->frequency_penalty) <= 2.0) << "Frequency penalty must be in [-2, 2]!";
  n->presence_penalty =
      json::LookupOrDefault<double>(config, "presence_penalty", default_config->presence_penalty);
  CHECK(std::fabs(n->presence_penalty) <= 2.0) << "Presence penalty must be in [-2, 2]!";
  n->repetition_penalty = json::LookupOrDefault<double>(config, "repetition_penalty",
                                                        default_config->repetition_penalty);
  CHECK(n->repetition_penalty > 0) << "Repetition penalty must be a positive number!";
  n->logprobs = json::LookupOrDefault<bool>(config, "logprobs", default_config->logprobs);
  n->top_logprobs =
      json::LookupOrDefault<int64_t>(config, "top_logprobs", default_config->top_logprobs);
  CHECK(n->top_logprobs >= 0 && n->top_logprobs <= 5)
      << "At most 5 top logprob tokens are supported";
  CHECK(n->top_logprobs == 0 || n->logprobs)
      << "\"logprobs\" must be true to support \"top_logprobs\"";

  std::optional<picojson::object> logit_bias_obj =
      json::LookupOptional<picojson::object>(config, "logit_bias");
  if (logit_bias_obj.has_value()) {
    std::vector<std::pair<int, float>> logit_bias;
    logit_bias.reserve(logit_bias_obj.value().size());
    for (auto [token_id_str, bias] : logit_bias_obj.value()) {
      CHECK(bias.is<double>());
      double bias_value = bias.get<double>();
      CHECK_LE(std::fabs(bias_value), 100.0) << "Logit bias value should be in range [-100, 100].";
      logit_bias.emplace_back(std::stoi(token_id_str), bias_value);
    }
    n->logit_bias = std::move(logit_bias);
  } else {
    n->logit_bias = default_config->logit_bias;
  }

  n->seed = json::LookupOrDefault<int64_t>(config, "seed", std::random_device{}());
  // "ignore_eos" is for benchmarking. Not the part of OpenAI API spec.
  n->ignore_eos = json::LookupOrDefault<bool>(config, "ignore_eos", default_config->ignore_eos);
  // "-1" means the generation will not stop until exceeding
  // model capability or hit any stop criteria.
  n->max_tokens = json::LookupOrDefault<int64_t>(config, "max_tokens", -1);

  std::optional<picojson::array> stop_strs_arr =
      json::LookupOptional<picojson::array>(config, "stop_strs");
  if (stop_strs_arr.has_value()) {
    Array<String> stop_strs;
    stop_strs.reserve(stop_strs_arr.value().size());
    for (const picojson::value& v : stop_strs_arr.value()) {
      CHECK(v.is<std::string>()) << "Invalid stop string in stop_strs";
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
      CHECK(v.is<int64_t>()) << "Invalid stop token in stop_token_ids";
      stop_token_ids.push_back(v.get<int64_t>());
    }
    n->stop_token_ids = std::move(stop_token_ids);
  } else {
    n->stop_token_ids = default_config->stop_token_ids;
  }

  std::optional<picojson::object> response_format_obj =
      json::LookupOptional<picojson::object>(config, "response_format");
  if (response_format_obj.has_value()) {
    ResponseFormat response_format;
    response_format.type = json::LookupOrDefault<std::string>(response_format_obj.value(), "type",
                                                              response_format.type);
    std::optional<std::string> schema =
        json::LookupOptional<std::string>(response_format_obj.value(), "schema");
    if (schema.has_value()) {
      response_format.schema = schema.value();
    }
    n->response_format = response_format;
  } else {
    n->response_format = default_config->response_format;
  }
  // "pinned" is for internal usage. Not the part of OpenAI API spec.
  n->pinned = json::LookupOrDefault<bool>(config, "pinned", default_config->pinned);

  data_ = std::move(n);
}

GenerationConfig GenerationConfig::GetDefaultFromModelConfig(
    const picojson::object& model_config_json) {
  ObjectPtr<GenerationConfigNode> n = make_object<GenerationConfigNode>();
  n->temperature = json::LookupOrDefault<double>(model_config_json, "temperature", n->temperature);
  n->top_p = json::LookupOrDefault<double>(model_config_json, "top_p", n->top_p);
  n->frequency_penalty =
      json::LookupOrDefault<double>(model_config_json, "frequency_penalty", n->frequency_penalty);
  n->presence_penalty =
      json::LookupOrDefault<double>(model_config_json, "presence_penalty", n->presence_penalty);
  return GenerationConfig(n);
}

String GenerationConfigNode::AsJSONString() const {
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

  // Params for benchmarking. Not the part of openai spec.
  config["ignore_eos"] = picojson::value(this->ignore_eos);

  picojson::object response_format;
  response_format["type"] = picojson::value(this->response_format.type);
  response_format["schema"] = this->response_format.schema
                                  ? picojson::value(this->response_format.schema.value())
                                  : picojson::value();
  config["response_format"] = picojson::value(response_format);

  // Params for internal usage. Not the part of OpenAI API spec.
  config["pinned"] = picojson::value(this->pinned);

  return picojson::value(config).serialize(true);
}

/****************** EngineConfig ******************/

TVM_REGISTER_OBJECT_TYPE(EngineConfigNode);

EngineConfig EngineConfig::FromJSONAndInferredConfig(
    const picojson::object& json, const InferrableEngineConfig& inferred_config) {
  CHECK(inferred_config.max_num_sequence.has_value());
  CHECK(inferred_config.max_total_sequence_length.has_value());
  CHECK(inferred_config.max_single_sequence_length.has_value());
  CHECK(inferred_config.prefill_chunk_size.has_value());
  CHECK(inferred_config.max_history_size.has_value());
  ObjectPtr<EngineConfigNode> n = make_object<EngineConfigNode>();

  // - Get models and model libs.
  n->model = json::Lookup<std::string>(json, "model");
  n->model_lib = json::Lookup<std::string>(json, "model_lib");
  std::vector<String> additional_models;
  std::vector<String> additional_model_libs;
  picojson::array additional_models_arr =
      json::LookupOrDefault<picojson::array>(json, "additional_models", picojson::array());
  picojson::array additional_model_libs_arr =
      json::LookupOrDefault<picojson::array>(json, "additional_model_libs", picojson::array());
  CHECK_EQ(additional_models_arr.size(), additional_model_libs_arr.size())
      << "The number of additional model libs does not match the number of additional models";
  int num_additional_models = additional_models_arr.size();
  additional_models.reserve(num_additional_models);
  additional_model_libs.reserve(num_additional_models);
  for (int i = 0; i < num_additional_models; ++i) {
    additional_models.push_back(json::Lookup<std::string>(additional_models_arr, i));
    additional_model_libs.push_back(json::Lookup<std::string>(additional_model_libs_arr, i));
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
  n->verbose = json::LookupOrDefault<bool>(json, "verbose", n->verbose);

  // - Fields from the inferred engine config.
  n->max_num_sequence = inferred_config.max_num_sequence.value();
  n->max_total_sequence_length = inferred_config.max_total_sequence_length.value();
  n->max_single_sequence_length = inferred_config.max_single_sequence_length.value();
  n->prefill_chunk_size = inferred_config.prefill_chunk_size.value();
  n->max_history_size = inferred_config.max_history_size.value();

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
  picojson::array additional_model_libs_arr =
      json::LookupOrDefault<picojson::array>(config, "additional_model_libs", picojson::array());
  if (additional_models_arr.size() != additional_model_libs_arr.size()) {
    return TResult::Error(
        "The number of additional model libs does not match the number of additional models");
  }

  int num_additional_models = additional_models_arr.size();
  std::vector<std::pair<std::string, std::string>> models_and_model_libs;
  models_and_model_libs.reserve(num_additional_models + 1);
  models_and_model_libs.emplace_back(model, model_lib);
  for (int i = 0; i < num_additional_models; ++i) {
    models_and_model_libs.emplace_back(json::Lookup<std::string>(additional_models_arr, i),
                                       json::Lookup<std::string>(additional_model_libs_arr, i));
  }
  return TResult::Ok(models_and_model_libs);
}

String EngineConfigNode::AsJSONString() const {
  picojson::object config;

  // - Models and model libs
  config["model"] = picojson::value(this->model);
  config["model_lib"] = picojson::value(this->model_lib);
  picojson::array additional_models_arr;
  picojson::array additional_model_libs_arr;
  additional_models_arr.reserve(this->additional_models.size());
  additional_model_libs_arr.reserve(this->additional_models.size());
  for (int i = 0; i < static_cast<int>(this->additional_models.size()); ++i) {
    additional_models_arr.push_back(picojson::value(this->additional_models[i]));
    additional_model_libs_arr.push_back(picojson::value(this->additional_model_libs[i]));
  }
  config["additional_models"] = picojson::value(additional_models_arr);
  config["additional_model_libs"] = picojson::value(additional_model_libs_arr);

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
  config["speculative_mode"] = picojson::value(SpeculativeModeToString(this->speculative_mode));
  config["spec_draft_length"] = picojson::value(static_cast<int64_t>(this->spec_draft_length));
  config["verbose"] = picojson::value(static_cast<bool>(this->verbose));

  return picojson::value(config).serialize(true);
}

/****************** InferrableEngineConfig ******************/

/*! \brief The class for config limitation from models. */
struct ModelConfigLimits {
  int64_t model_max_single_sequence_length;
  int64_t model_max_prefill_chunk_size;
  int64_t model_max_batch_size;
};

/*! \brief Convert the bytes to megabytes, keeping 3 decimals. */
inline std::string BytesToMegabytesString(double bytes) {
  std::string str;
  str.resize(20);
  std::sprintf(&str[0], "%.3f", bytes / 1024 / 1024);
  str.resize(std::strlen(str.c_str()));
  return str;
}

/*!
 * \brief Get the upper bound of single sequence length, prefill size and batch size
 * from model config.
 */
Result<ModelConfigLimits> GetModelConfigLimits(const std::vector<picojson::object>& model_configs) {
  int64_t model_max_single_sequence_length = std::numeric_limits<int64_t>::max();
  int64_t model_max_prefill_chunk_size = std::numeric_limits<int64_t>::max();
  int64_t model_max_batch_size = std::numeric_limits<int64_t>::max();
  for (int i = 0; i < static_cast<int>(model_configs.size()); ++i) {
    picojson::object compile_time_model_config =
        json::Lookup<picojson::object>(model_configs[i], "model_config");
    // - The maximum single sequence length is the minimum context window size among all models.
    int64_t runtime_context_window_size =
        json::LookupOptional<int64_t>(model_configs[i], "context_window_size").value_or(-1);
    int64_t compile_time_context_window_size =
        json::LookupOptional<int64_t>(compile_time_model_config, "context_window_size")
            .value_or(-1);
    if (runtime_context_window_size > compile_time_context_window_size) {
      return Result<ModelConfigLimits>::Error(
          "Model " + std::to_string(i) + "'s runtime context window size (" +
          std::to_string(runtime_context_window_size) +
          ") is larger than the context window size used at compile time (" +
          std::to_string(compile_time_context_window_size) + ").");
    }
    if (runtime_context_window_size == -1 && compile_time_context_window_size != -1) {
      return Result<ModelConfigLimits>::Error(
          "Model " + std::to_string(i) +
          "'s runtime context window size (infinite) is larger than the context "
          "window size used at compile time (" +
          std::to_string(compile_time_context_window_size) + ").");
    }
    if (runtime_context_window_size != -1) {
      model_max_single_sequence_length =
          std::min(model_max_single_sequence_length, runtime_context_window_size);
    }
    // - The maximum prefill chunk size is the minimum prefill chunk size among all models.
    int64_t runtime_prefill_chunk_size =
        json::Lookup<int64_t>(model_configs[i], "prefill_chunk_size");
    int64_t compile_time_prefill_chunk_size =
        json::Lookup<int64_t>(compile_time_model_config, "prefill_chunk_size");
    if (runtime_prefill_chunk_size > compile_time_prefill_chunk_size) {
      return Result<ModelConfigLimits>::Error(
          "Model " + std::to_string(i) + "'s runtime prefill chunk size (" +
          std::to_string(runtime_prefill_chunk_size) +
          ") is larger than the prefill chunk size used at compile time (" +
          std::to_string(compile_time_prefill_chunk_size) + ").");
    }
    if (runtime_prefill_chunk_size != -1) {
      model_max_prefill_chunk_size =
          std::min(model_max_prefill_chunk_size, runtime_prefill_chunk_size);
    }
    // - The maximum batch size is the minimum max batch size among all models.
    model_max_batch_size = std::min(
        model_max_batch_size, json::Lookup<int64_t>(compile_time_model_config, "max_batch_size"));
  }
  ICHECK_NE(model_max_prefill_chunk_size, std::numeric_limits<int64_t>::max());
  ICHECK_NE(model_max_batch_size, std::numeric_limits<int64_t>::max());
  ICHECK_GT(model_max_prefill_chunk_size, 0);
  ICHECK_GT(model_max_batch_size, 0);
  return Result<ModelConfigLimits>::Ok(
      {model_max_single_sequence_length, model_max_prefill_chunk_size, model_max_batch_size});
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
        model_config_limits.model_max_single_sequence_length;
  } else {
    inferred_config.max_single_sequence_length =
        std::min(inferred_config.max_single_sequence_length.value(),
                 model_config_limits.model_max_single_sequence_length);
  }
  // - 3. infer the maximum total sequence length that can fit GPU memory.
  double kv_bytes_per_token = 0;
  double kv_aux_workspace_bytes = 0;
  double model_workspace_bytes = 0;
  double logit_processor_workspace_bytes = 0;
  ICHECK_EQ(model_configs.size(), model_metadata.size());
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
    kv_bytes_per_token += head_dim * num_kv_heads * num_layers * 4 + 1.25;
    kv_aux_workspace_bytes +=
        (max_num_sequence + 1) * 88 + prefill_chunk_size * (num_qo_heads + 1) * 8 +
        prefill_chunk_size * head_dim * (num_qo_heads + num_kv_heads) * 4 + 48 * 1024 * 1024;
    model_workspace_bytes += prefill_chunk_size * 4 + max_num_sequence * 4 +
                             (prefill_chunk_size * 2 + max_num_sequence) * hidden_size * 2;
    logit_processor_workspace_bytes +=
        max_num_sequence * 20 + max_num_sequence * vocab_size * 16.125;
  }
  // Get single-card GPU size.
  TVMRetValue rv;
  DeviceAPI::Get(device)->GetAttr(device, DeviceAttrKind::kTotalGlobalMemory, &rv);
  int64_t gpu_size_bytes = rv;
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
          {model_max_total_sequence_length, model_config_limits.model_max_single_sequence_length,
           static_cast<int64_t>(8192)});
    } else if (mode == EngineMode::kInteractive) {
      inferred_config.max_total_sequence_length = std::min(
          model_max_total_sequence_length, model_config_limits.model_max_single_sequence_length);
    } else {
      inferred_config.max_total_sequence_length =
          std::min(model_max_total_sequence_length,
                   max_num_sequence * model_config_limits.model_max_single_sequence_length);
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
          std::min({model_config_limits.model_max_prefill_chunk_size,
                    inferred_config.max_total_sequence_length.value(),
                    model_config_limits.model_max_single_sequence_length});
    } else {
      inferred_config.prefill_chunk_size = model_config_limits.model_max_prefill_chunk_size;
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
  Result<ModelConfigLimits> model_config_limits_res = GetModelConfigLimits(model_configs);
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
        ICHECK_GE(v, 0);
        param_size *= v;
      }
      params_bytes += param_size;
    }
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
  Result<ModelConfigLimits> model_config_limits_res = GetModelConfigLimits(model_configs);
  if (model_config_limits_res.IsErr()) {
    return Result<InferrableEngineConfig>::Error(model_config_limits_res.UnwrapErr());
  }
  ModelConfigLimits model_config_limits = model_config_limits_res.Unwrap();

  std::ostringstream os;
  InferrableEngineConfig inferred_config = init_config;
  // - 1. prefill_chunk_size
  if (!init_config.prefill_chunk_size.has_value()) {
    inferred_config.prefill_chunk_size =
        std::min(model_config_limits.model_max_prefill_chunk_size, static_cast<int64_t>(4096));
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
        ICHECK_GE(v, 0);
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
  ICHECK_EQ(model_configs.size(), model_metadata.size());
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
  // Get single-card GPU size.
  TVMRetValue rv;
  DeviceAPI::Get(device)->GetAttr(device, DeviceAttrKind::kTotalGlobalMemory, &rv);
  int64_t gpu_size_bytes = rv;
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
  ICHECK_GE(model_configs.size(), 1);
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
