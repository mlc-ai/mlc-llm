/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine.h
 * \brief The header of serving engine in MLC LLM.
 */
#ifndef MLC_LLM_SERVE_ENGINE_H_
#define MLC_LLM_SERVE_ENGINE_H_

#include <tvm/runtime/packed_func.h>

#include "data.h"
#include "event_trace_recorder.h"
#include "request.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

typedef TypedPackedFunc<void(Array<RequestStreamOutput>)> FRequestStreamCallback;

/*!
 * \brief The engine interface for request serving in MLC LLM.
 * The engine can run one or multiple LLM models internally for
 * text generation. Usually, when there are multiple models,
 * speculative inference will be activated, where the first model
 * (index 0) is the main "large model" that has better generation
 * quality, and all other models are "small" models that used for
 * speculation.
 * The engine receives requests from the "AddRequest" method. For
 * an given request, the engine will keep generating new tokens for
 * the request until finish (under certain criterion). After finish,
 * the engine will return the generation result through the callback
 * function provided by the request.
 * \note For now only one model run in the engine is supported.
 * Multiple model support such as speculative inference will
 * be followed soon in the future.
 *
 * The public interface of Engine has the following three categories:
 * - engine management,
 * - high-level request management,
 * - engine "step" action.
 */
class Engine {
 public:
  /********************** Engine Management **********************/
  virtual ~Engine() = default;

  /*!
   * \brief Create an engine in unique pointer.
   * \param max_single_sequence_length The maximum allowed single
   * sequence length supported by the engine.
   * \param tokenizer_path The tokenizer path on disk.
   * \param kv_cache_config_json_str The KV cache config in JSON string.
   * \param engine_mode_json_str The Engine execution mode in JSON string.
   * \param request_stream_callback The request stream callback function to
   * stream back generated output for requests.
   * \param trace_recorder Event trace recorder for requests.
   * \param model_infos The model info tuples. Each tuple contains
   * - the model library, which might be a path to the binary file or
   * an executable module that is pre-loaded,
   * - the path to the model weight parameters,
   * - the device to run the model on.
   * \return The created Engine in pointer.
   */
  static std::unique_ptr<Engine> Create(
      int max_single_sequence_length, const String& tokenizer_path,
      const String& kv_cache_config_json_str, const String& engine_mode_json_str,
      Optional<PackedFunc> request_stream_callback, Optional<EventTraceRecorder> trace_recorder,
      const std::vector<std::tuple<TVMArgValue, String, DLDevice>>& model_infos);

  /*! \brief Reset the engine, clean up all running data and statistics. */
  virtual void Reset() = 0;

  /*! \brief Check if the engine has no request to process. */
  virtual bool Empty() = 0;

  /*! \brief Get the statistics of the Engine in JSON string. */
  virtual String Stats() = 0;

  /*! \brief Get the request stream callback function of the engine. */
  virtual Optional<PackedFunc> GetRequestStreamCallback() = 0;

  /*! \brief Set the request stream callback function of the engine. */
  virtual void SetRequestStreamCallback(Optional<PackedFunc> request_stream_callback) = 0;

  /***************** High-level Request Management *****************/

  /*! \brief Add a new request to the engine. */
  virtual void AddRequest(Request request) = 0;

  /*! \brief Abort the input request (specified by id string) from engine. */
  virtual void AbortRequest(const String& request_id) = 0;

  /*********************** Engine Action ***********************/

  /*!
   * \brief The main function that the engine takes a step of action.
   * At each step, the engine may decide to
   * - run prefill for one (or more) requests,
   * - run one-step decode for the all existing requests
   * ...
   * In the end of certain actions (e.g., decode), the engine will
   * check if any request has finished, and will return the
   * generation results for those finished requests.
   */
  virtual void Step() = 0;
};

/*!
 * \brief Create an Engine from packed arguments in TVMArgs.
 * \param args The arguments of engine construction.
 * \return The constructed engine in unique pointer.
 */
std::unique_ptr<Engine> CreateEnginePacked(TVMArgs args);

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_ENGINE_H_
