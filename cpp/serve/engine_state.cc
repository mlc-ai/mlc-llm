/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_state.cc
 */
#include "engine_state.h"

#include <picojson.h>

namespace mlc {
namespace llm {
namespace serve {

String EngineStats::AsJSON() const {
  picojson::object config;
  config["single_token_prefill_latency"] = picojson::value(
      total_prefill_length > 0 ? request_total_prefill_time / total_prefill_length : 0.0);
  config["single_token_decode_latency"] = picojson::value(
      total_decode_length > 0 ? request_total_decode_time / total_decode_length : 0.0);
  config["engine_total_prefill_time"] = picojson::value(engine_total_prefill_time);
  config["engine_total_decode_time"] = picojson::value(engine_total_decode_time);
  config["total_prefill_tokens"] = picojson::value(total_prefill_length);
  config["total_decode_tokens"] = picojson::value(total_decode_length);
  config["total_accepted_tokens"] = picojson::value(total_accepted_length);
  config["total_draft_tokens"] = picojson::value(total_draft_length);
  auto f_vector_to_array = [](const std::vector<int64_t>& vec) {
    picojson::array arr;
    for (int64_t v : vec) {
      arr.push_back(picojson::value(v));
    }
    return picojson::value(arr);
  };
  config["accept_count"] = f_vector_to_array(accept_count);
  config["draft_count"] = f_vector_to_array(draft_count);
  return picojson::value(config).serialize(true);
}

void EngineStats::Reset() {
  request_total_prefill_time = 0.0f;
  request_total_decode_time = 0.0f;
  engine_total_prefill_time = 0.0f;
  engine_total_decode_time = 0.0f;
  total_prefill_length = 0;
  total_decode_length = 0;
  total_accepted_length = 0;
  total_draft_length = 0;
  accept_count.clear();
  draft_count.clear();
}

TVM_REGISTER_OBJECT_TYPE(EngineStateObj);

EngineState::EngineState() { data_ = make_object<EngineStateObj>(); }

void EngineStateObj::Reset() {
  running_queue.clear();
  waiting_queue.clear();
  request_states.clear();
  id_manager.Reset();
  stats.Reset();
  if (prefix_cache.defined()) {
    prefix_cache->Reset();
  }
}

RequestState EngineStateObj::GetRequestState(Request request) {
  auto it = request_states.find(request->id);
  ICHECK(it != request_states.end());
  return it->second;
}

void EngineStats::UpdateSpecDecodingStats(int draft_length, int accept_length) {
  if (accept_count.size() < draft_length) {
    this->accept_count.resize(draft_length, 0);
    this->draft_count.resize(draft_length, 0);
  }
  for (int j = 0; j < draft_length; ++j) {
    if (j < accept_length) {
      this->accept_count[j]++;
    }
    this->draft_count[j]++;
  }
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
