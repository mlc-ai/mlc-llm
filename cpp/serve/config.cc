/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/config.cc
 */
#include "config.h"

#include <picojson.h>
#include <tvm/runtime/registry.h>

#include <random>

#include "../json_ffi/openai_api_protocol.h"
#include "../metadata/json_parser.h"
#include "data.h"

namespace mlc {
namespace llm {
namespace serve {

/****************** GenerationConfig ******************/

TVM_REGISTER_OBJECT_TYPE(GenerationConfigNode);

GenerationConfig::GenerationConfig(String config_json_str) {
  picojson::value config_json;
  std::string err = picojson::parse(config_json, config_json_str);
  if (!err.empty()) {
    LOG(FATAL) << err;
    return;
  }

  ObjectPtr<GenerationConfigNode> n = make_object<GenerationConfigNode>();

  picojson::object config = config_json.get<picojson::object>();
  if (config.count("n")) {
    CHECK(config["n"].is<int64_t>());
    n->n = config["n"].get<int64_t>();
    CHECK_GT(n->n, 0) << "\"n\" should be at least 1";
  }
  if (config.count("temperature")) {
    CHECK(config["temperature"].is<double>());
    n->temperature = config["temperature"].get<double>();
  }
  if (config.count("top_p")) {
    CHECK(config["top_p"].is<double>());
    n->top_p = config["top_p"].get<double>();
  }
  if (config.count("frequency_penalty")) {
    CHECK(config["frequency_penalty"].is<double>());
    n->frequency_penalty = config["frequency_penalty"].get<double>();
    CHECK(std::fabs(n->frequency_penalty) <= 2.0) << "Frequency penalty must be in [-2, 2]!";
  }
  if (config.count("presence_penalty")) {
    CHECK(config["presence_penalty"].is<double>());
    n->presence_penalty = config["presence_penalty"].get<double>();
    CHECK(std::fabs(n->presence_penalty) <= 2.0) << "Presence penalty must be in [-2, 2]!";
  }
  if (config.count("repetition_penalty")) {
    CHECK(config["repetition_penalty"].is<double>());
    n->repetition_penalty = config["repetition_penalty"].get<double>();
    CHECK(n->repetition_penalty > 0) << "Repetition penalty must be a positive number!";
  }
  if (config.count("logprobs")) {
    CHECK(config["logprobs"].is<bool>());
    n->logprobs = config["logprobs"].get<bool>();
  }
  if (config.count("top_logprobs")) {
    CHECK(config["top_logprobs"].is<int64_t>());
    n->top_logprobs = config["top_logprobs"].get<int64_t>();
    CHECK(n->top_logprobs >= 0 && n->top_logprobs <= 5)
        << "At most 5 top logprob tokens are supported";
    CHECK(n->top_logprobs == 0 || n->logprobs)
        << "\"logprobs\" must be true to support \"top_logprobs\"";
  }
  if (config.count("logit_bias")) {
    CHECK(config["logit_bias"].is<picojson::null>() || config["logit_bias"].is<picojson::object>());
    if (config["logit_bias"].is<picojson::object>()) {
      picojson::object logit_bias_json = config["logit_bias"].get<picojson::object>();
      std::vector<std::pair<int, float>> logit_bias;
      logit_bias.reserve(logit_bias_json.size());
      for (auto [token_id_str, bias] : logit_bias_json) {
        CHECK(bias.is<double>());
        double bias_value = bias.get<double>();
        CHECK_LE(std::fabs(bias_value), 100.0)
            << "Logit bias value should be in range [-100, 100].";
        logit_bias.emplace_back(std::stoi(token_id_str), bias_value);
      }
      n->logit_bias = std::move(logit_bias);
    }
  }
  if (config.count("max_tokens")) {
    if (config["max_tokens"].is<int64_t>()) {
      n->max_tokens = config["max_tokens"].get<int64_t>();
    } else {
      CHECK(config["max_tokens"].is<picojson::null>()) << "Unrecognized max_tokens";
      // "-1" means the generation will not stop until exceeding
      // model capability or hit any stop criteria.
      n->max_tokens = -1;
    }
  }
  if (config.count("seed")) {
    if (config["seed"].is<int64_t>()) {
      n->seed = config["seed"].get<int64_t>();
    } else {
      CHECK(config["seed"].is<picojson::null>()) << "Unrecognized seed";
      n->seed = std::random_device{}();
    }
  } else {
    n->seed = std::random_device{}();
  }
  if (config.count("stop_strs")) {
    CHECK(config["stop_strs"].is<picojson::array>())
        << "Invalid stop_strs. Stop strs should be an array of strings";
    picojson::array stop_strs_arr = config["stop_strs"].get<picojson::array>();
    Array<String> stop_strs;
    stop_strs.reserve(stop_strs_arr.size());
    for (const picojson::value& v : stop_strs_arr) {
      CHECK(v.is<std::string>()) << "Invalid stop string in stop_strs";
      stop_strs.push_back(v.get<std::string>());
    }
    n->stop_strs = std::move(stop_strs);
  }
  if (config.count("stop_token_ids")) {
    CHECK(config["stop_token_ids"].is<picojson::array>())
        << "Invalid stop_token_ids. Stop tokens should be an array of integers";
    picojson::array stop_token_ids_arr = config["stop_token_ids"].get<picojson::array>();
    std::vector<int> stop_token_ids;
    stop_token_ids.reserve(stop_token_ids_arr.size());
    for (const picojson::value& v : stop_token_ids_arr) {
      CHECK(v.is<int64_t>()) << "Invalid stop token in stop_token_ids";
      stop_token_ids.push_back(v.get<int64_t>());
    }
    n->stop_token_ids = std::move(stop_token_ids);
  }

  // Params for benchmarking. Not the part of openai spec.
  if (config.count("ignore_eos")) {
    CHECK(config["ignore_eos"].is<bool>());
    n->ignore_eos = config["ignore_eos"].get<bool>();
  }

  if (config.count("response_format")) {
    CHECK(config["response_format"].is<picojson::object>());
    picojson::object response_format_json = config["response_format"].get<picojson::object>();
    ResponseFormat response_format;
    if (response_format_json.count("type")) {
      CHECK(response_format_json["type"].is<std::string>());
      response_format.type = response_format_json["type"].get<std::string>();
    }
    if (response_format_json.count("schema")) {
      if (response_format_json["schema"].is<picojson::null>()) {
        response_format.schema = NullOpt;
      } else {
        CHECK(response_format_json["schema"].is<std::string>());
        response_format.schema = response_format_json["schema"].get<std::string>();
      }
    }
    n->response_format = response_format;
  }

  data_ = std::move(n);
}

Optional<GenerationConfig> GenerationConfig::Create(
    const std::string& json_str, std::string* err, const Conversation& conv_template,
    const ModelDefinedGenerationConfig& model_defined_gen_config) {
  std::optional<picojson::object> optional_json_obj = json::LoadJSONFromString(json_str, err);
  if (!err->empty() || !optional_json_obj.has_value()) {
    return NullOpt;
  }
  picojson::object& json_obj = optional_json_obj.value();
  ObjectPtr<GenerationConfigNode> n = make_object<GenerationConfigNode>();

  n->temperature =
      json::LookupOrDefault<double>(json_obj, "temperature", model_defined_gen_config->temperature);
  n->top_p = json::LookupOrDefault<double>(json_obj, "top_p", model_defined_gen_config->top_p);
  n->frequency_penalty = json::LookupOrDefault<double>(json_obj, "frequency_penalty",
                                                       model_defined_gen_config->frequency_penalty);
  n->presence_penalty = json::LookupOrDefault<double>(json_obj, "presence_penalty",
                                                      model_defined_gen_config->presence_penalty);
  n->logprobs = json::LookupOrDefault<bool>(json_obj, "logprobs", false);
  n->top_logprobs = static_cast<int>(json::LookupOrDefault<double>(json_obj, "top_logprobs", 0));
  n->ignore_eos = json::LookupOrDefault<bool>(json_obj, "ignore_eos", false);

  // Copy stop str from conversation template to generation config
  for (auto& stop_str : conv_template.stop_str) {
    n->stop_strs.push_back(stop_str);
  }
  for (auto& stop_token_id : conv_template.stop_token_ids) {
    n->stop_token_ids.push_back(stop_token_id);
  }

  GenerationConfig gen_config;
  gen_config.data_ = std::move(n);
  return gen_config;
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

  return picojson::value(config).serialize(true);
}

/****************** EngineConfig ******************/

TVM_REGISTER_OBJECT_TYPE(EngineConfigNode);

EngineConfig::EngineConfig(String model, String model_lib_path, Array<String> additional_models,
                           Array<String> additional_model_lib_paths, DLDevice device,
                           int kv_cache_page_size, int max_num_sequence,
                           int max_total_sequence_length, int max_single_sequence_length,
                           int prefill_chunk_size, int max_history_size, KVStateKind kv_state_kind,
                           SpeculativeMode speculative_mode, int spec_draft_length) {
  ObjectPtr<EngineConfigNode> n = make_object<EngineConfigNode>();
  n->model = std::move(model);
  n->model_lib_path = std::move(model_lib_path);
  n->additional_models = std::move(additional_models);
  n->additional_model_lib_paths = std::move(additional_model_lib_paths);
  n->device = device;
  n->kv_cache_page_size = kv_cache_page_size;
  n->max_num_sequence = max_num_sequence;
  n->max_total_sequence_length = max_total_sequence_length;
  n->max_single_sequence_length = max_single_sequence_length;
  n->prefill_chunk_size = prefill_chunk_size;
  n->max_history_size = max_history_size;
  n->kv_state_kind = kv_state_kind;
  n->spec_draft_length = spec_draft_length;
  n->speculative_mode = speculative_mode;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("mlc.serve.EngineConfig")
    .set_body_typed([](String model, String model_lib_path, Array<String> additional_models,
                       Array<String> additional_model_lib_paths, DLDevice device,
                       int kv_cache_page_size, int max_num_sequence, int max_total_sequence_length,
                       int max_single_sequence_length, int prefill_chunk_size, int max_history_size,
                       int kv_state_kind, int speculative_mode, int spec_draft_length) {
      return EngineConfig(std::move(model), std::move(model_lib_path), std::move(additional_models),
                          std::move(additional_model_lib_paths), device, kv_cache_page_size,
                          max_num_sequence, max_total_sequence_length, max_single_sequence_length,
                          prefill_chunk_size, max_history_size, KVStateKind(kv_state_kind),
                          SpeculativeMode(speculative_mode), spec_draft_length);
    });

}  // namespace serve
}  // namespace llm
}  // namespace mlc
