#ifndef MLC_LLM_JSON_FFI_CONV_TEMPLATE_H
#define MLC_LLM_JSON_FFI_CONV_TEMPLATE_H

#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <typeinfo>
#include <variant>
#include <vector>

#include "../serve/data.h"
#include "../support/result.h"
#include "truffle_protocol.h"
#include "picojson.h"

using namespace mlc::llm::serve;

namespace mlc {
namespace llm {
namespace truffle_ffi {


/****************** Model config ******************/

/*! \brief Defines the config of the model.
Populated from "model_config" field in mlc-chat-config.json */
class ModelConfig {
 public:
  int vocab_size;
  int context_window_size;
  int sliding_window_size;
  int prefill_chunk_size;
  int tensor_parallel_shards;
  int pipeline_parallel_stages;
  int max_batch_size;
  

  static ModelConfig FromJSON(const picojson::object& json_obj);
};



}  // namespace json_ffi
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_JSON_FFI_CONV_TEMPLATE_H
