/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/async_threaded_engine.h
 * \brief The header of threaded asynchronous serving engine in MLC LLM.
 */
#ifndef MLC_LLM_SERVE_ASYNC_THREADED_ENGINE_H_
#define MLC_LLM_SERVE_ASYNC_THREADED_ENGINE_H_

#include <tvm/runtime/packed_func.h>

#include "data.h"
#include "engine.h"
#include "request.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/*!
 * \brief The interface asynchronous threaded engine in MLC LLM.
 * The threaded engine keeps running a background request processing
 * loop on a standalone thread. Ensuring thread safety, it exposes
 * `AddRequest` and `AbortRequest` to receive new requests or
 * abortions from other threads, and the internal request processing
 * is backed by a normal engine wrapped inside.
 */
class AsyncThreadedEngine {
 public:
  virtual ~AsyncThreadedEngine() = default;

  /*! \brief Starts the background request processing loop. */
  virtual void RunBackgroundLoop() = 0;

  /*!
   * \brief Notify the AsyncThreadedEngine to exit the background
   * request processing loop. This method is invoked by threads
   * other than the engine-driving thread.
   */
  virtual void ExitBackgroundLoop() = 0;

  /*! \brief Add a new request to the engine. */
  virtual void AddRequest(Request request) = 0;

  /*! \brief Abort the input request (specified by id string) from engine. */
  virtual void AbortRequest(const String& request_id) = 0;
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_ASYNC_THREADED_ENGINE_H_
