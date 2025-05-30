/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/engine_state.h
 */
#ifndef MLC_LLM_SERVE_ENGINE_STATE_H_
#define MLC_LLM_SERVE_ENGINE_STATE_H_

#include <picojson.h>
#include <tvm/ffi/string.h>

#include "config.h"
#include "metrics.h"
#include "prefix_cache.h"
#include "request.h"
#include "request_state.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

typedef TypedFunction<void(Array<RequestStreamOutput>)> FRequestStreamCallback;

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

/*! \brief The data structures used in the action post-process. */
struct ActionPostProcessWorkspace {
  std::vector<RequestStateEntry> finished_rsentries;
  Array<RequestStreamOutput> callback_delta_outputs;
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
  /*! \brief A boolean flag denoting whether the running request state entry list has changed. */
  bool running_rsentries_changed = true;
  /*!
   * \brief The current engine speculative decoding draft length.
   * The length may change across time under the auto speculative decoding mode.
   * Value 0 means undefined. It must have a positive value for speculative decoding to
   * properly work.
   */
  int spec_draft_length = 0;
  /*! \brief A boolean flag denoting whether the engine is in disaggregation mode. */
  bool disaggregation = false;
  // Request stream callback function
  FRequestStreamCallback request_stream_callback_;
  /*!
   * \brief The post-process data structures.
   * We make it a workspace to avoid repetitive memory allocation/free in the action post process.
   */
  ActionPostProcessWorkspace postproc_workspace;

  /*! \brief Reset the engine state and clear the metrics. */
  void Reset();
  /*! \brief Get the request state of the given request. */
  RequestState GetRequestState(Request request);
  /*! \brief Return the running request state entries*/
  const std::vector<RequestStateEntry>& GetRunningRequestStateEntries();

  static constexpr const char* _type_key = "mlc.serve.EngineState";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_FINAL_OBJECT_INFO(EngineStateObj, Object);

 private:
  std::vector<RequestStateEntry> cached_running_rsentries_;
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
