/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/json_ffi_engine.h
 * \brief The header of JSON FFI engine in MLC LLM.
 */
#ifndef MLC_LLM_JSON_FFI_ENGINE_H_
#define MLC_LLM_JSON_FFI_ENGINE_H_

#include <dlpack/dlpack.h>
#include <picojson.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <thread>

#include "engine.h"
#include "openai_api_protocol.h"
#include "threaded_engine.h"

namespace mlc {
namespace llm {
namespace serve {
using namespace tvm::runtime;

class JSONFFIEngine {
 public:
  std::unique_ptr<ThreadedEngine> engine_;
  std::string err_;
  std::thread background_loop_thread_;
  std::thread background_stream_back_loop_thread_;
  PackedFunc request_stream_callback_;
  Tokenizer tokenizer_;
  //   JSONFFIEngine() {}
  //   JSONFFIEngine(TVMArgs args);

  ~JSONFFIEngine();

  bool ChatCompletion(std::string request_json_str, std::string request_id);

  bool Abort(std::string request_id);

  std::string GetLastError();

  void Terminate();
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_JSON_FFI_ENGINE_H_
