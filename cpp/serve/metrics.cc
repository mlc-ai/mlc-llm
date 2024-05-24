
/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/metrics.cc
 */
#include "metrics.h"

#include <tvm/runtime/logging.h>

#include <sstream>

namespace mlc {
namespace llm {
namespace serve {

picojson::value TimeCost::AsJSON() const {
  picojson::object config;
  config["count"] = picojson::value(count);
  if (count != 0) {
    config["mean"] = picojson::value(sum / count);
  }
  return picojson::value(config);
}

picojson::value SpecDecodeMetrics::AsJSON() const {
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

  ICHECK_EQ(draft_count.size(), accept_count.size());
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
          (static_cast<double>(accept_count[i]) / static_cast<double>(accept_count[i - 1]));
      accept_rate_metrics[accept_rate_label.str()] = picojson::value(accept_rate_value);
    }
  }
  metrics["accept_prob"] = picojson::value(accept_prob_metrics);
  metrics["accept_rate"] = picojson::value(accept_rate_metrics);
  metrics["accept_len"] = picojson::value(accept_len_metrics);

  return picojson::value(metrics);
}

picojson::value RequestMetrics::AsJSON() const {
  picojson::object metrics;
  metrics["num_input_tokens"] = picojson::value(num_input_tokens);
  metrics["num_prefill_tokens"] = picojson::value(num_prefill_tokens);
  metrics["num_output_tokens"] = picojson::value(num_output_tokens);

  if (num_output_tokens != 0) {
    metrics["decode_tokens_per_s"] = picojson::value(num_output_tokens / this->GetDecodeTime());
  }
  if (num_prefill_tokens != 0) {
    metrics["prefill_tokens_per_s"] = picojson::value(num_prefill_tokens / this->GetPrefillTime());
  }
  metrics["end_to_end_latency_s"] = picojson::value(this->GetTotalTime());
  return picojson::value(metrics);
}

picojson::value EngineMetrics::AsJSON() const {
  picojson::object metrics;
  metrics["engine_prefill_time_sum"] = picojson::value(engine_prefill_time_sum);
  metrics["engine_decode_time_sum"] = picojson::value(engine_decode_time_sum);
  metrics["num_input_tokens_sum"] = picojson::value(num_input_tokens_sum);
  metrics["num_prefill_tokens_sum"] = picojson::value(num_prefill_tokens_sum);
  metrics["num_output_tokens_sum"] = picojson::value(num_output_tokens_sum);
  metrics["last_finished_request"] = last_finished_request.AsJSON();
  if (!spec_decode.IsEmpty()) {
    metrics["spec_decode"] = spec_decode.AsJSON();
  }

  auto f_create_time_list = [](const std::vector<TimeCost>& time_list) {
    picojson::object result;
    for (size_t i = 1; i < time_list.size(); ++i) {
      const TimeCost& item = time_list[i];
      if (item.count == 0) continue;
      std::ostringstream label_mean;
      label_mean << "mean"
                 << "{batch_size=" << i << "}";
      double mean = item.sum / item.count;
      result[label_mean.str()] = picojson::value(mean);
      std::ostringstream label_count;
      label_count << "count"
                  << "{batch_size=" << i << "}";
      result[label_count.str()] = picojson::value(item.count);
    }
    return picojson::value(result);
  };

  metrics["decode_time_by_batch_size"] = f_create_time_list(decode_time_by_batch_size);
  metrics["draft_time_by_batch_size"] = f_create_time_list(draft_time_by_batch_size);
  metrics["verify_time_by_batch_size"] = f_create_time_list(verify_time_by_batch_size);

  return picojson::value(metrics);
}

void EngineMetrics::Reset() {
  engine_prefill_time_sum = 0.0;
  engine_decode_time_sum = 0.0;
  num_input_tokens_sum = 0;
  num_prefill_tokens_sum = 0;
  num_output_tokens_sum = 0;
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
