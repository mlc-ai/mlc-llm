
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

picojson::object TimeCost::AsJSON() const {
  picojson::object config;
  config["count"] = picojson::value(count);
  if (count != 0) {
    config["mean"] = picojson::value(sum / count);
  }
  return config;
}

picojson::object SpecDecodeMetrics::AsJSON() const {
  picojson::object metrics;
  auto f_vector_to_array = [](const std::vector<int64_t>& vec) {
    picojson::array arr;
    for (int64_t v : vec) {
      arr.push_back(picojson::value(v));
    }
    return picojson::value(arr);
  };
  metrics["draft_count"] = f_vector_to_array(draft_count);
  metrics["accept_count"] = f_vector_to_array(accept_count);

  TVM_FFI_ICHECK_EQ(draft_count.size(), accept_count.size());
  // NOTE: label follows prometheus with full context
  // so it can be flattened and used in metrics reoorting end point
  picojson::object accept_prob_metrics;
  picojson::object accept_rate_metrics;
  picojson::object accept_len_metrics;

  double accept_len_value = 0;

  for (size_t i = 0; i < draft_count.size(); ++i) {
    std::ostringstream accept_prob_label;
    accept_prob_label << "accept_prob{step=" << i << "}";
    double accept_prob_value =
        (static_cast<double>(accept_count[i]) / static_cast<double>(draft_count[i]));
    accept_prob_metrics[accept_prob_label.str()] = picojson::value(accept_prob_value);
    accept_len_value += accept_prob_value;

    std::ostringstream accept_len_label;
    accept_len_label << "accept_len{step=" << i << "}";
    accept_len_metrics[accept_len_label.str()] = picojson::value(accept_len_value);

    if (i != 0) {
      std::ostringstream accept_rate_label;
      accept_rate_label << "accept_rate{step=" << i << "}";
      double accept_rate_value =
          accept_count[i - 1] == 0
              ? 0.0f
              : (static_cast<double>(accept_count[i]) / static_cast<double>(accept_count[i - 1]));
      accept_rate_metrics[accept_rate_label.str()] = picojson::value(accept_rate_value);
    }
  }
  metrics["accept_prob"] = picojson::value(accept_prob_metrics);
  metrics["accept_rate"] = picojson::value(accept_rate_metrics);
  metrics["accept_len"] = picojson::value(accept_len_metrics);

  return metrics;
}

picojson::object RequestMetrics::AsJSON() const {
  picojson::object metrics;
  metrics["prompt_tokens"] = picojson::value(prompt_tokens);
  metrics["completion_tokens"] = picojson::value(completion_tokens);
  metrics["prefill_tokens"] = picojson::value(prefill_tokens);
  metrics["decode_tokens"] = picojson::value(decode_tokens);
  metrics["jump_forward_tokens"] = picojson::value(jump_forward_tokens);

  if (prefill_tokens != 0) {
    metrics["prefill_tokens_per_s"] = picojson::value(prefill_tokens / this->GetPrefillTime());
  }
  if (decode_tokens != 0) {
    metrics["decode_tokens_per_s"] = picojson::value(decode_tokens / this->GetDecodeTime());
  }
  metrics["end_to_end_latency_s"] = picojson::value(this->GetTotalTime());
  metrics["ttft_s"] = picojson::value(this->GetTTFT());
  metrics["inter_token_latency_s"] = picojson::value(this->GetInterTokenLatency());
  return metrics;
}

std::string RequestMetrics::AsUsageJSONStr(bool include_extra) const {
  picojson::object usage;
  usage["prompt_tokens"] = picojson::value(prompt_tokens);
  usage["completion_tokens"] = picojson::value(completion_tokens);
  usage["total_tokens"] = picojson::value(prompt_tokens + completion_tokens);
  if (include_extra) {
    usage["extra"] = picojson::value(this->AsJSON());
  }
  return picojson::value(usage).serialize();
}

picojson::object EngineMetrics::AsJSON() const {
  picojson::object metrics;
  metrics["engine_prefill_time_sum"] = picojson::value(engine_prefill_time_sum);
  metrics["engine_decode_time_sum"] = picojson::value(engine_decode_time_sum);
  metrics["engine_jump_forward_time_sum"] = picojson::value(engine_jump_forward_time_sum);
  metrics["prompt_tokens_sum"] = picojson::value(prompt_tokens_sum);
  metrics["completion_tokens_sum"] = picojson::value(completion_tokens_sum);
  metrics["prefill_tokens_sum"] = picojson::value(prefill_tokens_sum);
  metrics["decode_tokens_sum"] = picojson::value(decode_tokens_sum);
  metrics["jump_forward_tokens_sum"] = picojson::value(jump_forward_tokens_sum);

  if (prefill_tokens_sum != 0) {
    metrics["prefill_tokens_per_s"] = picojson::value(prefill_tokens_sum / engine_prefill_time_sum);
  }
  if (engine_decode_time_sum != 0) {
    metrics["decode_tokens_per_s"] = picojson::value(decode_tokens_sum / engine_decode_time_sum);
  }

  metrics["last_finished_request"] = picojson::value(last_finished_request.AsJSON());
  if (!spec_decode.IsEmpty()) {
    metrics["spec_decode"] = picojson::value(spec_decode.AsJSON());
  }

  auto f_create_time_list = [](const std::vector<TimeCost>& time_list) {
    picojson::object result;
    for (size_t i = 1; i < time_list.size(); ++i) {
      const TimeCost& item = time_list[i];
      if (item.count == 0) continue;
      std::ostringstream label_mean;
      label_mean << "mean{batch_size=" << i << "}";
      double mean = item.sum / item.count;
      result[label_mean.str()] = picojson::value(mean);
      std::ostringstream label_count;
      label_count << "count{batch_size=" << i << "}";
      result[label_count.str()] = picojson::value(item.count);
    }
    return picojson::value(result);
  };

  metrics["decode_time_by_batch_size"] = f_create_time_list(decode_time_by_batch_size);
  metrics["draft_time_by_batch_size"] = f_create_time_list(draft_time_by_batch_size);
  metrics["verify_time_by_batch_size"] = f_create_time_list(verify_time_by_batch_size);

  return metrics;
}

std::string EngineMetrics::AsUsageJSONStr() const {
  picojson::object usage;
  // We return engine usage as a usage field according to the OpenAI API.
  // To comply with the API, just set prompt_tokens, completion_tokens, and total_tokens to 0.
  // And store the information in the extra field.
  usage["prompt_tokens"] = picojson::value(static_cast<int64_t>(0));
  usage["completion_tokens"] = picojson::value(static_cast<int64_t>(0));
  usage["total_tokens"] = picojson::value(static_cast<int64_t>(0));
  usage["extra"] = picojson::value(this->AsJSON());
  return picojson::value(usage).serialize();
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
