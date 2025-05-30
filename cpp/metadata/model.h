/*!
 * \file model.h
 * \brief Metadata stored in model lib
 */
#ifndef MLC_LLM_CPP_MODEL_METADATA_H_
#define MLC_LLM_CPP_MODEL_METADATA_H_

#include <picojson.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/module.h>

#include <unordered_map>

namespace mlc {
namespace llm {

using tvm::ffi::Shape;
using tvm::ffi::String;
using tvm::runtime::DataType;

/*! \brief The kind of cache. */
enum class KVStateKind : int {
  kKVCache = 0,
  kRNNState = 1,
  kNone = 2,
};

inline std::string KVStateKindToString(KVStateKind kv_state_kind) {
  if (kv_state_kind == KVStateKind::kKVCache) {
    return "kv_cache";
  } else if (kv_state_kind == KVStateKind::kRNNState) {
    return "rnn_state";
  } else if (kv_state_kind == KVStateKind::kNone) {
    return "none";
  } else {
    LOG(FATAL) << "Invalid kv state kind: " << static_cast<int>(kv_state_kind);
  }
}

inline KVStateKind KVStateKindFromString(const std::string& kv_state_kind) {
  if (kv_state_kind == "kv_cache") {
    return KVStateKind::kKVCache;
  } else if (kv_state_kind == "rnn_state") {
    return KVStateKind::kRNNState;
  } else if (kv_state_kind == "none") {
    return KVStateKind::kNone;
  } else {
    LOG(FATAL) << "Invalid kv state kind string: " << kv_state_kind;
  }
}
struct ModelMetadata {
  struct Param {
    struct Preproc {
      String func_name;
      Shape in_shape;
      Shape out_shape;
      DataType out_dtype;
      static Preproc FromJSON(const picojson::object& js, const picojson::object& model_config);
    };

    String name;
    Shape shape;
    DataType dtype;
    std::vector<Preproc> preprocs;
    std::vector<int> pipeline_stages;
    static Param FromJSON(const picojson::object& param_obj, const picojson::object& model_config);
  };

  struct KVCacheMetadata {
    int64_t num_hidden_layers;
    int64_t num_attention_heads;
    int64_t num_key_value_heads;
    int64_t head_dim;
    static KVCacheMetadata FromJSON(const picojson::object& json);
  };

  std::string model_type;
  std::string quantization;
  int64_t context_window_size;
  int64_t prefill_chunk_size;
  int64_t max_batch_size;
  int64_t sliding_window_size;
  int64_t tensor_parallel_shards;
  int64_t pipeline_parallel_stages;
  bool disaggregation;
  int64_t attention_sink_size;
  std::vector<Param> params;
  std::unordered_map<std::string, int64_t> memory_usage;
  KVStateKind kv_state_kind;
  KVCacheMetadata kv_cache_metadata;

  static ModelMetadata FromJSON(const picojson::object& json_str,
                                const picojson::object& model_config);
  static ModelMetadata FromModule(tvm::runtime::Module module,
                                  const picojson::object& model_config);
};

}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_CPP_MODEL_METADATA_H_
