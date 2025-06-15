/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file json_ffi/json_ffi_engine.h
 * \brief The header of JSON FFI engine in MLC LLM.
 */
#ifndef MLC_LLM_JSON_FFI_JSON_FFI_ENGINE_H_
#define MLC_LLM_JSON_FFI_JSON_FFI_ENGINE_H_

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
  Function request_stream_callback_;
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

}  // namespace json_ffi
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_JSON_FFI_JSON_FFI_ENGINE_H_
