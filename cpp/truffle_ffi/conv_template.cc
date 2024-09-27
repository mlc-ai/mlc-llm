#include "conv_template.h"

#include <tvm/runtime/registry.h>

#include "../support/json_parser.h"


namespace mlc {
namespace llm {
namespace truffle_ffi {

using namespace mlc::llm;


/****************** Model config ******************/

ModelConfig ModelConfig::FromJSON(const picojson::object& json_obj) {
  ModelConfig config;

  Result<int64_t> vocab_size_res = json::LookupWithResultReturn<int64_t>(json_obj, "vocab_size");
  if (vocab_size_res.IsOk()) {
    config.vocab_size = vocab_size_res.Unwrap();
  }

  Result<int64_t> context_window_size_res =
      json::LookupWithResultReturn<int64_t>(json_obj, "context_window_size");
  if (context_window_size_res.IsOk()) {
    config.context_window_size = context_window_size_res.Unwrap();
  }

  Result<int64_t> sliding_window_size_res =
      json::LookupWithResultReturn<int64_t>(json_obj, "sliding_window_size");
  if (sliding_window_size_res.IsOk()) {
    config.sliding_window_size = sliding_window_size_res.Unwrap();
  }

  Result<int64_t> prefill_chunk_size_res =
      json::LookupWithResultReturn<int64_t>(json_obj, "prefill_chunk_size");
  if (prefill_chunk_size_res.IsOk()) {
    config.prefill_chunk_size = prefill_chunk_size_res.Unwrap();
  }

  Result<int64_t> tensor_parallel_shards_res =
      json::LookupWithResultReturn<int64_t>(json_obj, "tensor_parallel_shards");
  if (tensor_parallel_shards_res.IsOk()) {
    config.tensor_parallel_shards = tensor_parallel_shards_res.Unwrap();
  }

  Result<int64_t> pipeline_parallel_stages_res =
      json::LookupWithResultReturn<int64_t>(json_obj, "pipeline_parallel_stages");
  if (pipeline_parallel_stages_res.IsOk()) {
    config.pipeline_parallel_stages = pipeline_parallel_stages_res.Unwrap();
  }

  Result<int64_t> max_batch_size_res =
      json::LookupWithResultReturn<int64_t>(json_obj, "max_batch_size");
  if (max_batch_size_res.IsOk()) {
    config.max_batch_size = max_batch_size_res.Unwrap();
  }


  return config;
}



}  // namespace json_ffi
}  // namespace llm
}  // namespace mlc
