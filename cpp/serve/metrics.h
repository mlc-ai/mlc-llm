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
  picojson::value AsJSON() const;
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

  void Reset() {
    accept_count.clear();
    draft_count.clear();
  }
  picojson::value AsJSON() const;
};

/*! \brief Runtime metrics of engine. */
struct EngineMetrics {
  /*! \brief The total engine time on prefill, including warmup */
  double sum_engine_prefill_time = 0;
  /*! \brief The total engine time on decode/draft/verify, including warmup */
  double sum_engine_decode_time = 0;
  /*! \brief The total number of request input tokens. */
  int64_t sum_num_input_tokens = 0;
  /*! \brief The total number of processed tokens (excluding the prefix-cached length) in prefill */
  int64_t sum_num_prefill_tokens = 0;
  /*! \brief The total number of request output tokens */
  int64_t sum_num_output_tokens = 0;

  /*! \brief The prefill time of the latest finished request. */
  double last_finished_req_prefill_time = 0.0;
  /*! \brief The decode time of the latest finished request. */
  double last_finished_req_decode_time = 0.0;
  /*! \brief The number of input tokens of the latest finished request. */
  double last_finished_req_num_input_tokens = 0.0;
  /*!
   * \brief The number of prefilled tokens (excluding the prefix-cached length) of the latest
   * finished request.
   */
  double last_finished_req_num_prefill_tokens = 0.0;
  /*! \brief The number of output tokens of the latest finished request. */
  double last_finished_req_num_output_tokens = 0.0;

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
   * \brief Return the engine runtime metrics in JSON.
   * \return The metrics in JSON
   */
  picojson::value AsJSON() const;
  /*! \brief Reset all the metrics. */
  void Reset();
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_METRIC_H_
