/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_state.h
 */
#ifndef MLC_LLM_SERVE_ENGINE_STATE_H_
#define MLC_LLM_SERVE_ENGINE_STATE_H_

#include <tvm/runtime/container/string.h>

#include "request.h"
#include "request_state.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/*! \brief Runtime statistics of engine. */
struct EngineStats {
  /*! \brief The current total sequence length in the first model. */
  int64_t current_total_seq_len = 0;
  /*! \brief The sum of "prefill time of each request". */
  double request_total_prefill_time = 0.0f;
  /*! \brief The sum of "decode time of each request". */
  double request_total_decode_time = 0.0f;
  /*! \brief The total engine time on prefill. */
  double engine_total_prefill_time = 0.0f;
  /*! \brief The total engine time on decode. */
  double engine_total_decode_time = 0.0f;
  /*! \brief The total number of processed tokens in prefill. */
  int64_t total_prefill_length = 0;
  /*! \brief The total number of processed tokens in decode. */
  int64_t total_decode_length = 0;

  /*!
   * \brief Return the engine runtime statistics in JSON string.
   * We collect the following entries:
   * - single token prefill latency (s/tok): avg latency of processing one token in prefill
   * - single token decode latency (s/tok): avg latency of processing one token in decode
   * - engine time for prefill (sec)
   * - engine time for decode (sec)
   * - total number of processed tokens in prefill.
   * - total number of processed tokens in decode.
   * \return The statistics in JSON string.
   */
  String AsJSON() const;
  /*! \brief Reset all the statistics. */
  void Reset();
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
  /*! \brief The requests to abort. */
  std::vector<Request> abort_queue;
  /*! \brief The states of all requests. */
  std::unordered_map<String, RequestState, ObjectPtrHash, ObjectPtrEqual> request_states;
  /*! \brief Runtime statistics. */
  EngineStats stats;

  /*! \brief Reset the engine state and clear the statistics. */
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
