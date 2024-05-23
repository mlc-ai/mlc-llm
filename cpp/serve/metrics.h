/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/metric.h
 * \brief Metrics of serving engine/requests.
 */
#ifndef MLC_LLM_SERVE_METRICS_H_
#define MLC_LLM_SERVE_METRICS_H_

#include <picojson.h>

#include <string>

namespace mlc {
namespace llm {
namespace serve {

// We keep all metrics containers in this header (instead of in Engine and Request State)
// so we have a single central place to define all metrics across the engine.
// Conceptually, these statistics are derived from engine/request behaviors.

/*!
 * \brief The class for metric tracking in MLC.
 * - Each metric has a label string which can be empty.
 * - We maintain the number of updates (`count`) and the sum of updated values (`sum`).
 * - We support warmup. When `warmup` is false, the first update will be discarded.
 */
struct Metric {
  std::string label;
  double sum = 0.0;
  int64_t count = 0;
  bool warmed_up = false;

  explicit Metric(bool warmed_up = false, std::string label = "")
      : label(std::move(label)), warmed_up(warmed_up) {}

  /*! \brief Update the metric with given value. */
  void Update(double value) {
    if (warmed_up) {
      sum += value;
      count += 1;
    } else {
      warmed_up = true;
    }
  }

  /*! \brief Set the metric with the given value. */
  void Set(double value) {
    if (warmed_up) {
      sum = value;
      count = 1;
    } else {
      warmed_up = true;
    }
  }

  /*! \brief Reset the metric. */
  void Reset(bool warmed_up = false) {
    this->sum = 0.0;
    this->count = 0;
    this->warmed_up = warmed_up;
  }

  /*! \brief Overloading "+=" for quick update. */
  Metric& operator+=(double value) {
    this->Update(value);
    return *this;
  }

  /*! \brief Dump the metric as JSON. */
  picojson::value AsJSON() const {
    picojson::object config;
    config["label"] = picojson::value(label);
    config["sum"] = picojson::value(sum);
    config["count"] = picojson::value(count);
    config["warmed_up"] = picojson::value(warmed_up);
    return picojson::value(config);
  }
};

/*! \brief Runtime metrics of engine. */
struct EngineMetrics {
  /*! \brief The sum of "prefill time of each request". */
  Metric sum_request_prefill_time = Metric(/*warmed_up=*/true);
  /*! \brief The sum of "decode time of each request". */
  Metric sum_request_decode_time = Metric(/*warmed_up=*/true);
  /*! \brief The total engine time on prefill. */
  Metric sum_engine_prefill_time = Metric(/*warmed_up=*/true);
  /*! \brief The total engine time on decode. */
  Metric sum_engine_decode_time = Metric(/*warmed_up=*/true);
  /*! \brief The total number of request input tokens. */
  Metric sum_num_input_tokens = Metric(/*warmed_up=*/true);
  /*! \brief The total number of processed tokens (excluding the prefix-cached length) in prefill */
  Metric sum_num_prefill_tokens = Metric(/*warmed_up=*/true);
  /*! \brief The total number of request output tokens */
  Metric sum_num_output_tokens = Metric(/*warmed_up=*/true);
  /*! \brief The total number of accepted tokens in speculation verification. */
  Metric sum_num_accepted_tokens = Metric(/*warmed_up=*/true);
  /*! \brief The total number of speculated draft tokens. */
  Metric sum_num_draft_tokens = Metric(/*warmed_up=*/true);

  /*! \brief The prefill time of the latest finished request. */
  Metric last_finished_req_prefill_time = Metric(/*warmed_up=*/true);
  /*! \brief The decode time of the latest finished request. */
  Metric last_finished_req_decode_time = Metric(/*warmed_up=*/true);
  /*! \brief The number of input tokens of the latest finished request. */
  Metric num_last_finished_req_input_tokens = Metric(/*warmed_up=*/true);
  /*!
   * \brief The number of prefilled tokens (excluding the prefix-cached length) of the latest
   * finished request.
   */
  Metric num_last_finished_req_prefill_tokens = Metric(/*warmed_up=*/true);
  /*! \brief The number of output tokens of the latest finished request. */
  Metric num_last_finished_req_output_tokens = Metric(/*warmed_up=*/true);

  /*! \brief The maximum batch size we record for batch decode time. */
  static constexpr const int64_t kMaxEffectiveBatchSize = 64;
  /*! \brief The list of batch decode time under different batch size. */
  std::vector<Metric> batch_decode_time_list = std::vector<Metric>(kMaxEffectiveBatchSize);
  /*! \brief The list of batch draft time (a single decode step) under different batch size. */
  std::vector<Metric> batch_draft_time_list = std::vector<Metric>(kMaxEffectiveBatchSize);
  /*! \brief The list of batch verification time under different effective batch size. */
  std::vector<Metric> batch_verification_time_list = std::vector<Metric>(kMaxEffectiveBatchSize);

  /*! \brief The number of accepted tokens in speculative decoding. */
  std::vector<int64_t> accept_count;
  /*! \brief The number of draft tokens in speculative decoding. */
  std::vector<int64_t> draft_count;

  /*!
   * \brief Return the engine runtime metrics in JSON.
   * \return The metrics in JSON
   */
  picojson::value AsJSON() const;
  /*! \brief Reset all the metrics. */
  void Reset();

  // NOTE: we keep most update function in header
  // so they can be inlined effectively
  /*!
   * \brief Update the batch decode time for the given batch size.
   * The time will be ignored if the batch size is greater than `kMaxEffectiveBatchSize`.
   */
  void UpdateBatchDecodeTime(int batch_size, double time);
  /*!
   * \brief Update the single-step batch draft time for the given batch size.
   * The time will be ignored if the batch size is greater than `kMaxEffectiveBatchSize`.
   */
  void UpdateBatchDraftTime(int batch_size, double time);
  /*!
   * \brief Update the batch decode time for the given effective batch size.
   * The time will be ignored if the effective batch size is greater than `kMaxEffectiveBatchSize`.
   */
  void UpdateBatchVerificationTime(int effective_batch_size, double time);
  /*!
   * \brief Update the metrics of speculative decoding.
   * \param draft_length The number of draft tokens (including the last prediction by the base
   * model)
   * \param accept_length The number of accepted tokens in the speculative decoding.
   */
  void UpdateSpecDecodingStats(int draft_length, int accept_length);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_METRIC_H_
