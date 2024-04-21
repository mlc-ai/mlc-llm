/*!
 *  Copyright (c) 2023 by Contributors
 * \file json_ffi/json_ffi_engine.h
 * \brief The header of JSON FFI engine in MLC LLM.
 */
#ifndef MLC_LLM_JSON_FFI_JSON_FFI_ENGINE_H_
#define MLC_LLM_JSON_FFI_JSON_FFI_ENGINE_H_

#include <tvm/runtime/packed_func.h>

#include <string>

#include "../serve/threaded_engine.h"
#include "../streamer.h"
#include "conv_template.h"
#include "openai_api_protocol.h"

namespace mlc {
namespace llm {
namespace json_ffi {

using namespace tvm::runtime;
using namespace mlc::llm::serve;

/*!
 * \brief // Todo: document this class, fields and member functions
 */
class JSONFFIEngine {
 public:
  JSONFFIEngine();

  ~JSONFFIEngine();

  bool ChatCompletion(std::string request_json_str, std::string request_id);

  bool AddRequest(std::string request_json_str, std::string request_id);

  void StreamBackError(std::string request_id);

  bool Abort(std::string request_id);

  std::string GetLastError();

  void ExitBackgroundLoop();

 protected:
  std::unique_ptr<ThreadedEngine> engine_;
  std::string err_;
  PackedFunc request_stream_callback_;
  TextStreamer streamer_;  // TODO: Support "n", and support different streamers for each request
  Conversation conv_template_;
};

}  // namespace json_ffi
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_JSON_FFI_JSON_FFI_ENGINE_H_