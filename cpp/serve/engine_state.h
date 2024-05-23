/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_state.h
 */
#ifndef MLC_LLM_SERVE_ENGINE_STATE_H_
#define MLC_LLM_SERVE_ENGINE_STATE_H_

#include <picojson.h>
#include <tvm/runtime/container/string.h>

#include "config.h"
#include "metric.h"
#include "prefix_cache.h"
#include "request.h"
#include "request_state.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

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
   * \brief Return the engine runtime metrics in JSON string.
   * \return The metrics in JSON.
   */
  picojson::value AsJSON() const;
  /*! \brief Reset all the metrics. */
  void Reset();

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

/*! \brief The manager of internal id for requests in engine. */
struct EngineInternalIDManager {
  std::vector<int64_t> available_ids;
  int64_t id_cnt = 0;

  /*! \brief Return an unused id. */
  int64_t GetNewId() {
    if (!available_ids.empty()) {
      int64_t id = available_ids.back();
      available_ids.pop_back();
      return id;
    } else {
      return id_cnt++;
    }
  }

  /*! \brief Recycle an id. */
  void RecycleId(int64_t id) { available_ids.push_back(id); }

  /*! \brief Reset the manager. */
  void Reset() {
    available_ids.clear();
    id_cnt = 0;
  }
};

/*!
 * \brief The state of the running engine.
 * It contains the requests and their states submitted to the Engine.
 */
class EngineStateObj : public Object {
 public:
  /*! \brief The requests being processed. */
  std::vector<Request> running_queue;
  /*! \brief The requests that have not started for process yet. */
  std::vector<Request> waiting_queue;
  /*! \brief The states of all requests. */
  std::unordered_map<String, RequestState> request_states;
  /*! \brief The internal id manager. */
  EngineInternalIDManager id_manager;
  /*! \brief Runtime metrics. */
  EngineMetrics metrics;
  /*! \brief The prefix cache. */
  PrefixCache prefix_cache{nullptr};

  /*! \brief Reset the engine state and clear the metrics. */
  void Reset();
  /*! \brief Get the request state of the given request. */
  RequestState GetRequestState(Request request);

  static constexpr const char* _type_key = "mlc.serve.EngineState";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_FINAL_OBJECT_INFO(EngineStateObj, Object);
};

/*!
 * \brief Managed reference of EngineStateObj.
 * \sa EngineStateObj
 */
class EngineState : public ObjectRef {
 public:
  explicit EngineState();

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(EngineState, ObjectRef, EngineStateObj);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_ENGINE_STATE_H_
