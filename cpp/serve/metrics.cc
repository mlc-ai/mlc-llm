
/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/metrics.cc
 */
#include "metrics.h"

#include <tvm/runtime/logging.h>

#include <sstream>

namespace mlc {
namespace llm {
namespace serve {

tvm::ffi::json::Object TimeCost::AsJSON() const {
  tvm::ffi::json::Object config;
  config.Set("count", count);
  if (count != 0) {
    config.Set("mean", sum / count);
  }
  return config;
}

tvm::ffi::json::Object SpecDecodeMetrics::AsJSON() const {
  tvm::ffi::json::Object metrics;
  auto f_vector_to_array = [](const std::vector<int64_t>& vec) {
    tvm::ffi::json::Array arr;
    for (int64_t v : vec) {
      arr.push_back(v);
    }
    return tvm::ffi::json::Value(arr);
  };
  metrics.Set("draft_count", f_vector_to_array(draft_count));
  metrics.Set("accept_count", f_vector_to_array(accept_count));

  ICHECK_EQ(draft_count.size(), accept_count.size());
  // NOTE: label follows prometheus with full context
  // so it can be flattened and used in metrics reoorting end point
  tvm::ffi::json::Object accept_prob_metrics;
  tvm::ffi::json::Object accept_rate_metrics;
  tvm::ffi::json::Object accept_len_metrics;

  double accept_len_value = 0;

  for (size_t i = 0; i < draft_count.size(); ++i) {
    std::ostringstream accept_prob_label;
    accept_prob_label << "accept_prob{step=" << i << "}";
    double accept_prob_value =
        (static_cast<double>(accept_count[i]) / static_cast<double>(draft_count[i]));
    accept_prob_metrics.Set(accept_prob_label.str(), accept_prob_value);
    accept_len_value += accept_prob_value;

    std::ostringstream accept_len_label;
    accept_len_label << "accept_len{step=" << i << "}";
    accept_len_metrics.Set(accept_len_label.str(), accept_len_value);

    if (i != 0) {
      std::ostringstream accept_rate_label;
      accept_rate_label << "accept_rate{step=" << i << "}";
      double accept_rate_value =
          accept_count[i - 1] == 0
              ? 0.0f
              : (static_cast<double>(accept_count[i]) / static_cast<double>(accept_count[i - 1]));
      accept_rate_metrics.Set(accept_rate_label.str(), accept_rate_value);
    }
  }
  metrics.Set("accept_prob", accept_prob_metrics);
  metrics.Set("accept_rate", accept_rate_metrics);
  metrics.Set("accept_len", accept_len_metrics);

  return metrics;
}

tvm::ffi::json::Object RequestMetrics::AsJSON() const {
  tvm::ffi::json::Object metrics;
  metrics.Set("prompt_tokens", prompt_tokens);
  metrics.Set("completion_tokens", completion_tokens);
  metrics.Set("prefill_tokens", prefill_tokens);
  metrics.Set("decode_tokens", decode_tokens);
  metrics.Set("jump_forward_tokens", jump_forward_tokens);

  if (prefill_tokens != 0) {
    metrics.Set("prefill_tokens_per_s", prefill_tokens / this->GetPrefillTime());
  }
  if (decode_tokens != 0) {
    metrics.Set("decode_tokens_per_s", decode_tokens / this->GetDecodeTime());
  }
  metrics.Set("end_to_end_latency_s", this->GetTotalTime());
  metrics.Set("ttft_s", this->GetTTFT());
  metrics.Set("inter_token_latency_s", this->GetInterTokenLatency());
  return metrics;
}

std::string RequestMetrics::AsUsageJSONStr(bool include_extra) const {
  tvm::ffi::json::Object usage;
  usage.Set("prompt_tokens", prompt_tokens);
  usage.Set("completion_tokens", completion_tokens);
  usage.Set("total_tokens", prompt_tokens + completion_tokens);
  if (include_extra) {
    usage.Set("extra", this->AsJSON());
  }
  return tvm::ffi::json::Stringify(usage);
}

tvm::ffi::json::Object EngineMetrics::AsJSON() const {
  tvm::ffi::json::Object metrics;
  metrics.Set("engine_prefill_time_sum", engine_prefill_time_sum);
  metrics.Set("engine_decode_time_sum", engine_decode_time_sum);
  metrics.Set("engine_jump_forward_time_sum", engine_jump_forward_time_sum);
  metrics.Set("prompt_tokens_sum", prompt_tokens_sum);
  metrics.Set("completion_tokens_sum", completion_tokens_sum);
  metrics.Set("prefill_tokens_sum", prefill_tokens_sum);
  metrics.Set("decode_tokens_sum", decode_tokens_sum);
  metrics.Set("jump_forward_tokens_sum", jump_forward_tokens_sum);

  if (prefill_tokens_sum != 0) {
    metrics.Set("prefill_tokens_per_s", prefill_tokens_sum / engine_prefill_time_sum);
  }
  if (engine_decode_time_sum != 0) {
    metrics.Set("decode_tokens_per_s", decode_tokens_sum / engine_decode_time_sum);
  }

  metrics.Set("last_finished_request", last_finished_request.AsJSON());
  if (!spec_decode.IsEmpty()) {
    metrics.Set("spec_decode", spec_decode.AsJSON());
  }

  auto f_create_time_list = [](const std::vector<TimeCost>& time_list) {
    tvm::ffi::json::Object result;
    for (size_t i = 1; i < time_list.size(); ++i) {
      const TimeCost& item = time_list[i];
      if (item.count == 0) continue;
      std::ostringstream label_mean;
      label_mean << "mean{batch_size=" << i << "}";
      double mean = item.sum / item.count;
      result.Set(label_mean.str(), mean);
      std::ostringstream label_count;
      label_count << "count{batch_size=" << i << "}";
      result.Set(label_count.str(), item.count);
    }
    return tvm::ffi::json::Value(result);
  };

  metrics.Set("decode_time_by_batch_size", f_create_time_list(decode_time_by_batch_size));
  metrics.Set("draft_time_by_batch_size", f_create_time_list(draft_time_by_batch_size));
  metrics.Set("verify_time_by_batch_size", f_create_time_list(verify_time_by_batch_size));

  return metrics;
}

std::string EngineMetrics::AsUsageJSONStr() const {
  tvm::ffi::json::Object usage;
  // We return engine usage as a usage field according to the OpenAI API.
  // To comply with the API, just set prompt_tokens, completion_tokens, and total_tokens to 0.
  // And store the information in the extra field.
  usage.Set("prompt_tokens", static_cast<int64_t>(0));
  usage.Set("completion_tokens", static_cast<int64_t>(0));
  usage.Set("total_tokens", static_cast<int64_t>(0));
  usage.Set("extra", this->AsJSON());
  return tvm::ffi::json::Stringify(usage);
}

void EngineMetrics::Reset() {
  engine_prefill_time_sum = 0.0;
  engine_decode_time_sum = 0.0;
  engine_jump_forward_time_sum = 0;
  prompt_tokens_sum = 0;
  completion_tokens_sum = 0;
  prefill_tokens_sum = 0;
  decode_tokens_sum = 0;
  jump_forward_tokens_sum = 0;
  last_finished_request.Reset();
  spec_decode.Reset();
  decode_time_by_batch_size.clear();
  draft_time_by_batch_size.clear();
  verify_time_by_batch_size.clear();
  decode_time_by_batch_size.resize(kEndFineGrainedTrackingBatchSize);
  draft_time_by_batch_size.resize(kEndFineGrainedTrackingBatchSize);
  verify_time_by_batch_size.resize(kEndFineGrainedTrackingBatchSize);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
