/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_state.cc
 */
#include "engine_state.h"

#include <picojson.h>

namespace mlc {
namespace llm {
namespace serve {

picojson::value EngineMetrics::AsJSON() const {
  picojson::object metrics;
  metrics["sum_request_prefill_time"] = picojson::value(sum_request_prefill_time.AsJSON());
  metrics["sum_request_decode_time"] = picojson::value(sum_request_decode_time.AsJSON());
  metrics["sum_engine_prefill_time"] = picojson::value(sum_engine_prefill_time.AsJSON());
  metrics["sum_engine_decode_time"] = picojson::value(sum_engine_decode_time.AsJSON());
  metrics["sum_num_input_tokens"] = picojson::value(sum_num_input_tokens.AsJSON());
  metrics["sum_num_prefill_tokens"] = picojson::value(sum_num_prefill_tokens.AsJSON());
  metrics["sum_num_output_tokens"] = picojson::value(sum_num_output_tokens.AsJSON());
  metrics["sum_num_accepted_tokens"] = picojson::value(sum_num_accepted_tokens.AsJSON());
  metrics["sum_num_draft_tokens"] = picojson::value(sum_num_draft_tokens.AsJSON());

  metrics["last_finished_req_prefill_time"] =
      picojson::value(last_finished_req_prefill_time.AsJSON());
  metrics["last_finished_req_decode_time"] =
      picojson::value(last_finished_req_decode_time.AsJSON());
  metrics["num_last_finished_req_input_tokens"] =
      picojson::value(num_last_finished_req_input_tokens.AsJSON());
  metrics["num_last_finished_req_prefill_tokens"] =
      picojson::value(num_last_finished_req_prefill_tokens.AsJSON());
  metrics["num_last_finished_req_output_tokens"] =
      picojson::value(num_last_finished_req_output_tokens.AsJSON());

  picojson::array batch_decode_time_obj;
  picojson::array batch_draft_time_obj;
  picojson::array batch_verification_time_obj;
  batch_decode_time_obj.reserve(batch_decode_time_list.size());
  batch_draft_time_obj.reserve(batch_draft_time_list.size());
  batch_verification_time_obj.reserve(batch_verification_time_list.size());
  for (const Metric& batch_decode_time : batch_decode_time_list) {
    if (batch_decode_time.label.empty()) {
      continue;
    }
    batch_decode_time_obj.push_back(picojson::value(batch_decode_time.AsJSON()));
  }
  for (const Metric& batch_draft_time : batch_draft_time_list) {
    if (batch_draft_time.label.empty()) {
      continue;
    }
    batch_draft_time_obj.push_back(picojson::value(batch_draft_time.AsJSON()));
  }
  for (const Metric& batch_verification_time : batch_verification_time_list) {
    if (batch_verification_time.label.empty()) {
      continue;
    }
    batch_verification_time_obj.push_back(picojson::value(batch_verification_time.AsJSON()));
  }
  metrics["batch_decode_time_per_batch_size"] = picojson::value(batch_decode_time_obj);
  metrics["batch_draft_time_per_batch_size"] = picojson::value(batch_draft_time_obj);
  metrics["batch_verification_time_per_batch_size"] = picojson::value(batch_verification_time_obj);

  auto f_vector_to_array = [](const std::vector<int64_t>& vec) {
    picojson::array arr;
    for (int64_t v : vec) {
      arr.push_back(picojson::value(v));
    }
    return picojson::value(arr);
  };
  metrics["accept_count"] = f_vector_to_array(accept_count);
  metrics["draft_count"] = f_vector_to_array(draft_count);
  return picojson::value(metrics);
}

void EngineMetrics::Reset() {
  sum_request_prefill_time.Reset(/*warmed_up=*/true);
  sum_request_decode_time.Reset(/*warmed_up=*/true);
  sum_engine_prefill_time.Reset(/*warmed_up=*/true);
  sum_engine_decode_time.Reset(/*warmed_up=*/true);
  sum_num_input_tokens.Reset(/*warmed_up=*/true);
  sum_num_prefill_tokens.Reset(/*warmed_up=*/true);
  sum_num_output_tokens.Reset(/*warmed_up=*/true);
  sum_num_accepted_tokens.Reset(/*warmed_up=*/true);
  sum_num_draft_tokens.Reset(/*warmed_up=*/true);
  last_finished_req_prefill_time.Reset(/*warmed_up=*/true);
  last_finished_req_decode_time.Reset(/*warmed_up=*/true);
  num_last_finished_req_input_tokens.Reset(/*warmed_up=*/true);
  num_last_finished_req_prefill_tokens.Reset(/*warmed_up=*/true);
  num_last_finished_req_output_tokens.Reset(/*warmed_up=*/true);
  batch_decode_time_list.clear();
  batch_draft_time_list.clear();
  batch_verification_time_list.clear();
  batch_decode_time_list.resize(kMaxEffectiveBatchSize);
  batch_draft_time_list.resize(kMaxEffectiveBatchSize);
  batch_verification_time_list.resize(kMaxEffectiveBatchSize);
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
  metrics.Reset();
  if (prefix_cache.defined()) {
    prefix_cache->Reset();
  }
}

RequestState EngineStateObj::GetRequestState(Request request) {
  auto it = request_states.find(request->id);
  ICHECK(it != request_states.end());
  return it->second;
}

void EngineMetrics::UpdateBatchDecodeTime(int batch_size, double time) {
  if (batch_size > kMaxEffectiveBatchSize) {
    return;
  }
  if (batch_decode_time_list[batch_size].label.empty()) {
    batch_decode_time_list[batch_size].label = std::to_string(batch_size);
  }
  batch_decode_time_list[batch_size].Update(time);
}

void EngineMetrics::UpdateBatchDraftTime(int batch_size, double time) {
  if (batch_size > kMaxEffectiveBatchSize) {
    return;
  }
  if (batch_draft_time_list[batch_size].label.empty()) {
    batch_draft_time_list[batch_size].label = std::to_string(batch_size);
  }
  batch_draft_time_list[batch_size].Update(time);
}

void EngineMetrics::UpdateBatchVerificationTime(int batch_size, double time) {
  if (batch_size > kMaxEffectiveBatchSize) {
    return;
  }
  if (batch_verification_time_list[batch_size].label.empty()) {
    batch_verification_time_list[batch_size].label = std::to_string(batch_size);
  }
  batch_verification_time_list[batch_size].Update(time);
}

void EngineMetrics::UpdateSpecDecodingStats(int draft_length, int accept_length) {
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
