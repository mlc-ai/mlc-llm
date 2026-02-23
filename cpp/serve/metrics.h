/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/metric.h
 * \brief Metrics of serving engine/requests.
 */
#ifndef MLC_LLM_SERVE_METRICS_H_
#define MLC_LLM_SERVE_METRICS_H_

#include <picojson.h>
#include <tvm/runtime/logging.h>

#include <chrono>
#include <string>

namespace mlc {
namespace llm {
namespace serve {

// We keep all metrics containers in this header (instead of in Engine and Request State)
// so we have a single central place to define all metrics across the engine.
// Conceptually, these statistics are derived from engine/request behaviors.

/*!
 * \brief The class for tracking mean time cost.
 * - We maintain the number of updates (`count`) and the sum of updated values (`sum`).
 * - We support warmup. When `warmup` is false, the first update will be discarded.
 */
struct TimeCost {
  /*! \brief the total amount of cost excluding warm up time */
  double sum = 0.0;
  /*! \brief the total count of events excluding warmup */
  int64_t count = 0;
  /*! \brief Whether we warmed up already, assuming one hit is enough */
  bool warmed_up = false;

  /*! \brief Update the metric with given value. */
  void Update(double value) {
    if (warmed_up) {
      sum += value;
      count += 1;
    } else {
      warmed_up = true;
    }
  }

  /*! \brief Reset the metric. */
  void Reset() {
    // NOTE: no need to redo warmup
    // assuming we are measuring the same thing
    this->sum = 0.0;
    this->count = 0;
  }

  /*! \brief Dump the metric as JSON. */
  picojson::object AsJSON() const;
};

/*! \brief Runtime metrics for speculative decoding */
struct SpecDecodeMetrics {
  /*! \brief The number of draft tokens in speculative decoding, per step */
  std::vector<int64_t> draft_count;
  /*! \brief The number of accepted tokens in speculative decoding, per step */
  std::vector<int64_t> accept_count;

  /*!
   * \brief Update the metrics of speculative decoding.
   * \param draft_length The number of draft tokens (including the last prediction by the base
   * model)
   * \param accept_length The number of accepted tokens in the speculative decoding.
   */
  void Update(int draft_length, int accept_length) {
    TVM_FFI_ICHECK_GE(accept_length, 1);
    if (accept_count.size() < draft_length) {
      this->accept_count.resize(draft_length, 0);
      this->draft_count.resize(draft_length, 0);
    }
    for (int j = 0; j < draft_length; ++j) {
      if (j < accept_length) {
        ++this->accept_count[j];
      }
      ++this->draft_count[j];
    }
  }

  bool IsEmpty() const { return draft_count.size() == 0; }

  void Reset() {
    accept_count.clear();
    draft_count.clear();
  }
  picojson::object AsJSON() const;
};

/*!
 * \brief Metrics attached to each request
 *
 * Sometimes requests can involve tree decode(e.g. parallel n).
 * The metrics is collected across all branches of the tree.
 */
struct RequestMetrics {
  /*! \brief Request input tokens. */
  int64_t prompt_tokens = 0;
  /*! \brief Total number of output tokens. */
  int64_t completion_tokens = 0;
  /*! \brief Total number of tokens that needs to be prefilled */
  int64_t prefill_tokens = 0;
  /*! \brief The number of processed tokens (including tokens rolled back later) in decode. */
  int64_t decode_tokens = 0;
  /*! \brief The number of tokens predicted by jump-forward decoding. */
  int64_t jump_forward_tokens = 0;

  /*! \brief The time of adding the request to engine. */
  std::chrono::high_resolution_clock::time_point add_time_point;
  /*! \brief The time of finishing prefill stage. */
  std::chrono::high_resolution_clock::time_point prefill_end_time_point;
  /*! \brief The time of finishing all decode. */
  std::chrono::high_resolution_clock::time_point finish_time_point;

  /*! \brief check whether the request metrics is a completed request */
  bool IsComplete() const { return prompt_tokens != 0 && completion_tokens != 0; }

  /*! \return the prefill time in seconds */
  double GetPrefillTime() const {
    return static_cast<double>((prefill_end_time_point - add_time_point).count()) / 1e9;
  }

  /*! \return the decode time in seconds */
  double GetDecodeTime() const {
    return static_cast<double>((finish_time_point - prefill_end_time_point).count()) / 1e9;
  }

  /*! \return the time to first token (TTFT) in seconds */
  double GetTTFT() const {
    return static_cast<double>((prefill_end_time_point - add_time_point).count()) / 1e9;
  }

  /*! \return the prefill time in seconds */
  double GetTotalTime() const {
    return static_cast<double>((finish_time_point - add_time_point).count()) / 1e9;
  }

  /*! \return the inter token latency (ITL) in seconds */
  double GetInterTokenLatency() const {
    return completion_tokens > 0 ? GetTotalTime() / completion_tokens : 0.0;
  }

  /*! \brief Reset the metric. */
  void Reset() {
    this->prompt_tokens = 0;
    this->prefill_tokens = 0;
    this->completion_tokens = 0;
  }
  /*!
   * \brief Return the request metrics in JSON.
   * \return The metrics in JSON
   */
  picojson::object AsJSON() const;
  /*!
   * \brief Return OpenAI compatible usage metrics
   * \param include_extra Whether to include extra set of metrics
   *
   * \return The usage metrics in json.
   */
  std::string AsUsageJSONStr(bool include_extra) const;
};

/*! \brief Runtime metrics of engine. */
struct EngineMetrics {
  /*! \brief The total engine time on prefill, including warmup */
  double engine_prefill_time_sum = 0;
  /*! \brief The total engine time on decode/draft/verify, including warmup */
  double engine_decode_time_sum = 0;
  /*! \brief The total engine time on jump-forward prediction. */
  double engine_jump_forward_time_sum = 0;
  /*! \brief The total number of request input tokens. */
  int64_t prompt_tokens_sum = 0;
  /*! \brief The total number of request output tokens */
  int64_t completion_tokens_sum = 0;
  /*! \brief The total number of processed tokens (excluding the prefix-cached length) in prefill */
  int64_t prefill_tokens_sum = 0;
  /*! \brief The total number of processed tokens (including tokens rolled back later) in decode. */
  int64_t decode_tokens_sum = 0;
  /*! \brief The total number of tokens predicted by jump-forward decoding. */
  int64_t jump_forward_tokens_sum = 0;
  /*! \brief metrics from last finished request. */
  RequestMetrics last_finished_request;
  /*! \brief speculative decoding metrics */
  SpecDecodeMetrics spec_decode;

  /*! \brief The maximum batch size we track for batch decode time. */
  static constexpr const int64_t kEndFineGrainedTrackingBatchSize = 65;
  /*! \brief The list of batch decode time under different batch size. */
  std::vector<TimeCost> decode_time_by_batch_size =
      std::vector<TimeCost>(kEndFineGrainedTrackingBatchSize);
  /*! \brief The list of batch draft time (a single decode step) under different batch size. */
  std::vector<TimeCost> draft_time_by_batch_size =
      std::vector<TimeCost>(kEndFineGrainedTrackingBatchSize);
  /*! \brief The list of batch verification time under different effective batch size. */
  std::vector<TimeCost> verify_time_by_batch_size =
      std::vector<TimeCost>(kEndFineGrainedTrackingBatchSize);

  // NOTE: we keep most update function in header
  // so they can be inlined effectively
  /*!
   * \brief Update the batch decode time for the given batch size.
   * The time will be ignored if the batch size is greater than `kMaxBatchSizeForTracking`.
   */
  void UpdateDecodeTimeByBatchSize(int batch_size, double time) {
    if (batch_size < kEndFineGrainedTrackingBatchSize) {
      decode_time_by_batch_size[batch_size].Update(time);
    }
  }
  /*!
   * \brief Update the single-step batch draft time for the given batch size.
   * The time will be ignored if the batch size is greater than `kMaxBatchSizeForTracking`.
   */
  void UpdateDraftTimeByBatchSize(int batch_size, double time) {
    if (batch_size < kEndFineGrainedTrackingBatchSize) {
      draft_time_by_batch_size[batch_size].Update(time);
    }
  }
  /*!
   * \brief Update the batch decode time for the given effective batch sizPe.
   * The time will be ignored if the effective batch size is greater than
   * `kMaxBatchSizeForTracking`.
   */
  void UpdateVerifyTimeByBatchSize(int effective_batch_size, double time) {
    if (effective_batch_size < kEndFineGrainedTrackingBatchSize) {
      verify_time_by_batch_size[effective_batch_size].Update(time);
    }
  }

  /*!
   * \brief Update global engine metrics as we finish a request
   *  by including the information from the finished request.
   */
  void RequestFinishUpdate(const RequestMetrics& request_metrics) {
    prompt_tokens_sum += request_metrics.prompt_tokens;
    prefill_tokens_sum += request_metrics.prefill_tokens;
    completion_tokens_sum += request_metrics.completion_tokens;
    decode_tokens_sum += request_metrics.decode_tokens;
    jump_forward_tokens_sum += request_metrics.jump_forward_tokens;
    last_finished_request = request_metrics;
  }
  /*!
   * \brief Return the engine runtime metrics in JSON.
   * \return The metrics in JSON
   */
  picojson::object AsJSON() const;

  /*!
   * \brief return engine metrics as usage json string.
   * \return The resulting usage json string.
   */
  std::string AsUsageJSONStr() const;

  /*! \brief Reset all the metrics. */
  void Reset();
};
}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_METRIC_H_
