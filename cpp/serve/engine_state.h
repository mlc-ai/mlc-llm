/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_state.h
 */
#ifndef MLC_LLM_SERVE_ENGINE_STATE_H_
#define MLC_LLM_SERVE_ENGINE_STATE_H_

#include <picojson.h>
#include <tvm/runtime/container/string.h>

#include "config.h"
#include "metrics.h"
#include "prefix_cache.h"
#include "request.h"
#include "request_state.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

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
