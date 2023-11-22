/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_state.cc
 */
#define PICOJSON_USE_INT64

#include "engine_state.h"

#include <picojson.h>

namespace mlc {
namespace llm {
namespace serve {

String EngineStats::AsJSON() const {
  picojson::object config;
  config["single_token_prefill_latency"] =
      picojson::value(request_total_prefill_time / total_prefill_length);
  config["single_token_decode_latency"] =
      picojson::value(request_total_decode_time / total_decode_length);
  config["engine_total_prefill_time"] = picojson::value(engine_total_prefill_time);
  config["engine_total_decode_time"] = picojson::value(engine_total_decode_time);
  config["total_prefill_tokens"] = picojson::value(total_prefill_length);
  config["total_decode_tokens"] = picojson::value(total_decode_length);
  return picojson::value(config).serialize(true);
}

void EngineStats::Reset() {
  current_total_seq_len = 0;
  request_total_prefill_time = 0.0f;
  request_total_decode_time = 0.0f;
  engine_total_prefill_time = 0.0f;
  engine_total_decode_time = 0.0f;
  total_prefill_length = 0;
  total_decode_length = 0;
}

TVM_REGISTER_OBJECT_TYPE(EngineStateObj);

EngineState::EngineState() { data_ = make_object<EngineStateObj>(); }

void EngineStateObj::Reset() {
  running_queue.clear();
  waiting_queue.clear();
  request_states.clear();
  stats.Reset();
}

RequestState EngineStateObj::GetRequestState(Request request) {
  return request_states.at(request->id);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
