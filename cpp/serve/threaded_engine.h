/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/threaded_engine.h
 * \brief The header of threaded serving engine in MLC LLM.
 */
#ifndef MLC_LLM_SERVE_THREADED_ENGINE_H_
#define MLC_LLM_SERVE_THREADED_ENGINE_H_

#include <tvm/runtime/packed_func.h>

#include "data.h"
#include "engine.h"
#include "request.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/*!
 * \brief The interface threaded engine in MLC LLM.
 * The threaded engine keeps running a background request processing
 * loop on a standalone thread. Ensuring thread safety, it exposes
 * `AddRequest` and `AbortRequest` to receive new requests or
 * abortions from other threads, and the internal request processing
 * is backed by a normal engine wrapped inside.
 */
class ThreadedEngine {
 public:
  /*! \brief Create a ThreadedEngine. */
  static std::unique_ptr<ThreadedEngine> Create();

  virtual ~ThreadedEngine() = default;

  /*!
   * \brief Initialize the threaded engine from packed arguments in TVMArgs.
   * \param engine_config The engine config.
   * \param request_stream_callback The request stream callback function to.
   * \param trace_recorder Event trace recorder for requests.
   */
  virtual void InitBackgroundEngine(EngineConfig engine_config,
                                    Optional<PackedFunc> request_stream_callback,
                                    Optional<EventTraceRecorder> trace_recorder) = 0;

  /*! \brief Starts the background request processing loop. */
  virtual void RunBackgroundLoop() = 0;

  /*! \brief Starts the request stream callback loop. */
  virtual void RunBackgroundStreamBackLoop() = 0;

  /*!
   * \brief Notify the ThreadedEngine to exit the background
   * request processing loop. This method is invoked by threads
   * other than the engine-driving thread.
   */
  virtual void ExitBackgroundLoop() = 0;

  /*! \brief Add a new request to the engine. */
  virtual void AddRequest(Request request) = 0;

  /*! \brief Abort the input request (specified by id string) from engine. */
  virtual void AbortRequest(const String& request_id) = 0;

  /************** Debug/Profile **************/

  /*! \brief Call the given global function on all workers. Only for debug purpose. */
  virtual void DebugCallFuncOnAllAllWorker(const String& func_name) = 0;
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_THREADED_ENGINE_H_
