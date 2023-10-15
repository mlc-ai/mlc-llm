/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/config.cc
 */
#define PICOJSON_USE_INT64

#include "config.h"

#include <picojson.h>

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
  if (config.count("temperature")) {
    CHECK(config["temperature"].is<double>());
    n->temperature = config["temperature"].get<double>();
  }
  if (config.count("top_p")) {
    CHECK(config["top_p"].is<double>());
    n->top_p = config["top_p"].get<double>();
  }
  if (config.count("repetition_penalty")) {
    CHECK(config["repetition_penalty"].is<double>());
    n->repetition_penalty = config["repetition_penalty"].get<double>();
    CHECK(n->repetition_penalty > 0) << "Repetition penalty must be a positive number!";
  }
  if (config.count("max_new_tokens")) {
    CHECK(config["max_new_tokens"].is<int64_t>());
    n->max_new_tokens = config["max_new_tokens"].get<int64_t>();
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

  data_ = std::move(n);
}

String GenerationConfigNode::AsJSONString() const {
  picojson::object config;
  config["temperature"] = picojson::value(this->temperature);
  config["top_p"] = picojson::value(this->top_p);
  config["repetition_penalty"] = picojson::value(this->repetition_penalty);
  config["max_new_tokens"] = picojson::value(static_cast<int64_t>(this->max_new_tokens));

  picojson::array stop_strs_arr;
  for (String stop_str : this->stop_strs) {
    stop_strs_arr.push_back(picojson::value(stop_str));
  }
  config["stop_strs"] = picojson::value(stop_strs_arr);

  return picojson::value(config).serialize(true);
}

/****************** KVCacheConfig ******************/

TVM_REGISTER_OBJECT_TYPE(KVCacheConfigNode);

KVCacheConfig::KVCacheConfig(int page_size, int max_num_sequence, int max_total_sequence_length) {
  ObjectPtr<KVCacheConfigNode> n = make_object<KVCacheConfigNode>();
  n->page_size = page_size;
  n->max_num_sequence = max_num_sequence;
  n->max_total_sequence_length = max_total_sequence_length;
  data_ = std::move(n);
}

KVCacheConfig::KVCacheConfig(const std::string& config_str, int max_single_sequence_length) {
  int page_size;
  int max_total_sequence_length;
  int max_num_sequence = -1;

  picojson::value config_json;
  std::string err = picojson::parse(config_json, config_str);
  if (!err.empty()) {
    LOG(FATAL) << err;
  }

  // Get json fields.
  picojson::object config = config_json.get<picojson::object>();
  if (config.count("page_size")) {
    CHECK(config["page_size"].is<int64_t>());
    page_size = config["page_size"].get<int64_t>();
    CHECK_GE(page_size, 16) << "KV cache page size smaller than 16 is not supported.";
  } else {
    LOG(FATAL) << "Key \"page_size\" not found.";
  }
  if (config.count("max_total_sequence_length")) {
    CHECK(config["max_total_sequence_length"].is<int64_t>());
    max_total_sequence_length = config["max_total_sequence_length"].get<int64_t>();
  } else {
    LOG(FATAL) << "Key \"max_total_sequence_length\" not found.";
  }
  if (config.count("max_num_sequence")) {
    CHECK(config["max_num_sequence"].is<int64_t>());
    max_num_sequence = config["max_num_sequence"].get<int64_t>();
  }

  if (max_num_sequence == -1) {
    max_num_sequence = max_total_sequence_length / max_single_sequence_length;
  }

  ObjectPtr<KVCacheConfigNode> n = make_object<KVCacheConfigNode>();
  n->page_size = page_size;
  n->max_num_sequence = max_num_sequence;
  n->max_total_sequence_length = max_total_sequence_length;
  data_ = std::move(n);
}

String KVCacheConfigNode::AsJSONString() const {
  picojson::object config;
  config["page_size"] = picojson::value(static_cast<int64_t>(this->page_size));
  config["max_num_sequence"] = picojson::value(static_cast<int64_t>(this->max_num_sequence));
  config["max_total_sequence_length"] =
      picojson::value(static_cast<int64_t>(this->max_total_sequence_length));
  return picojson::value(config).serialize(true);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
