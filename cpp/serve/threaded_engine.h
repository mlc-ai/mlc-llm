/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/threaded_engine.h
 * \brief The header of threaded serving engine in MLC LLM.
 */
#ifndef MLC_LLM_SERVE_THREADED_ENGINE_H_
#define MLC_LLM_SERVE_THREADED_ENGINE_H_

#include <picojson.h>

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
   * \brief Initialize the threaded engine from packed arguments in PackedArgs.
   * \param device The device where to run models.
   * \param request_stream_callback The request stream callback function to.
   * \param trace_recorder Event trace recorder for requests.
   */
  virtual void InitThreadedEngine(Device device, Optional<Function> request_stream_callback,
                                  Optional<EventTraceRecorder> trace_recorder) = 0;

  /*!
   * \brief Reload the engine with the new engine config.
   * \param engine_config_json_str The engine config JSON string.
   */
  virtual void Reload(String engine_config_json_str) = 0;

  /*! \brief Unload the background engine. */
  virtual void Unload() = 0;

  /*! \brief Reset the engine to the initial state. */
  virtual void Reset() = 0;

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

  /************** Query/Profile/Debug **************/

  /*! \brief Return the default generation config. */
  virtual GenerationConfig GetDefaultGenerationConfig() const = 0;

  /*! \brief Return the complete engine config. */
  virtual EngineConfig GetCompleteEngineConfig() const = 0;

  /*! \brief Call the given global function on all workers. Only for debug purpose. */
  virtual void DebugCallFuncOnAllAllWorker(const String& func_name, Optional<String> func_args) = 0;
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_THREADED_ENGINE_H_
