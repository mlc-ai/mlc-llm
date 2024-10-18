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
#include "../tokenizers/streamer.h"
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
  /*! \brief local request state entry, one per reply stream. */
  struct RequestState {
    /*! \brief model to fill in reply. */
    std::string model;
    /*! \brief text streamer for each stream */
    std::vector<TextStreamer> streamer;
  };

  std::unique_ptr<ThreadedEngine> engine_;
  std::string err_;
  PackedFunc request_stream_callback_;
  // tokenizer
  Tokenizer tokenizer_;
  // conversation template
  Conversation conv_template_;
  // generation config
  GenerationConfig default_generation_config_;
  // model config
  ModelConfig model_config_;
  // local device
  DLDevice device_;
  // request state map
  std::unordered_map<String, RequestState> request_map_;
};

class JSONFFIEngineImpl : public JSONFFIEngine, public ModuleNode {
 public:
  static Module Create();

  TVM_MODULE_VTABLE_BEGIN("mlc.json_ffi");
  TVM_MODULE_VTABLE_ENTRY("init_background_engine", &JSONFFIEngineImpl::InitBackgroundEngine);
  TVM_MODULE_VTABLE_ENTRY("reload", &JSONFFIEngineImpl::Reload);
  TVM_MODULE_VTABLE_ENTRY("unload", &JSONFFIEngineImpl::Unload);
  TVM_MODULE_VTABLE_ENTRY("reset", &JSONFFIEngineImpl::Reset);
  TVM_MODULE_VTABLE_ENTRY("chat_completion", &JSONFFIEngineImpl::ChatCompletion);
  TVM_MODULE_VTABLE_ENTRY("abort", &JSONFFIEngineImpl::Abort);
  TVM_MODULE_VTABLE_ENTRY("get_last_error", &JSONFFIEngineImpl::GetLastError);
  TVM_MODULE_VTABLE_ENTRY("run_background_loop", &JSONFFIEngineImpl::RunBackgroundLoop);
  TVM_MODULE_VTABLE_ENTRY("run_background_stream_back_loop",
                          &JSONFFIEngineImpl::RunBackgroundStreamBackLoop);
  TVM_MODULE_VTABLE_ENTRY("exit_background_loop", &JSONFFIEngineImpl::ExitBackgroundLoop);
  TVM_MODULE_VTABLE_END();

  void InitBackgroundEngine(int device_type, int device_id,
                            Optional<PackedFunc> request_stream_callback);
  void Reload(String engine_config_json_str);
  void Unload();
  void Reset();
  void RunBackgroundLoop();
  void RunBackgroundStreamBackLoop();

  // Implement the TVM_MODULE_VTABLE
  // TVM_DEFINE_OBJECT_REF_METHODS(JSONFFIEngineImpl, ModuleNode, JSONFFIEngineImplNode);

 private:
  String GetResponseFromStreamOutput(Array<RequestStreamOutput> delta_outputs);
};

}  // namespace json_ffi
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_JSON_FFI_JSON_FFI_ENGINE_H_
