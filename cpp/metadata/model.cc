#include "./model.h"

#include <unordered_map>

#include "../support/json_parser.h"

namespace mlc {
namespace llm {

using namespace tvm::runtime;
using tvm::ffi::Function;
using tvm::ffi::Optional;

ModelMetadata::Param::Preproc ModelMetadata::Param::Preproc::FromJSON(
    const tvm::ffi::json::Object& js, const tvm::ffi::json::Object& model_config) {
  Preproc preproc;
  CHECK_GE(js.size(), 3) << "ValueError: Invalid preprocessing info in JSON";
  preproc.func_name = json::Lookup<std::string>(js, "func_name");
  json::SymShapeTuple sym_out_shape = json::Lookup<json::SymShapeTuple>(js, "out_shape");
  preproc.out_shape = sym_out_shape.ToStatic(model_config);
  json::SymShapeTuple sym_in_shape =
      json::LookupOrDefault<json::SymShapeTuple>(js, "in_shape", sym_out_shape);
  preproc.in_shape = sym_in_shape.ToStatic(model_config);
  preproc.out_dtype = json::Lookup<DataType>(js, "out_dtype");
  return preproc;
}

ModelMetadata::Param ModelMetadata::Param::FromJSON(const tvm::ffi::json::Object& param,
                                                    const tvm::ffi::json::Object& model_config) {
  Param result;
  result.name = json::Lookup<std::string>(param, "name");
  result.dtype = json::Lookup<DataType>(param, "dtype");
  // A shape being `-1` means that it is dynamic
  json::SymShapeTuple sym_shape = json::Lookup<json::SymShapeTuple>(param, "shape");
  result.shape = sym_shape.ToStatic(model_config);
  // - "preproc"
  tvm::ffi::json::Array preprocs = json::Lookup<tvm::ffi::json::Array>(param, "preprocs");
  result.preprocs.reserve(preprocs.size());
  for (int i = 0; i < preprocs.size(); i++) {
    result.preprocs.emplace_back(ModelMetadata::Param::Preproc::FromJSON(
        json::Lookup<tvm::ffi::json::Object>(preprocs, i), model_config));
  }
  // - "pipeline_stages"
  int pipeline_parallel_stages =
      json::LookupOrDefault<int64_t>(model_config, "pipeline_parallel_stages", 1);
  std::optional<tvm::ffi::json::Array> opt_pipeline_stages =
      json::LookupOptional<tvm::ffi::json::Array>(param, "pipeline_stages");
  if (pipeline_parallel_stages > 1) {
    CHECK(opt_pipeline_stages.has_value())
        << "The pipeline stage is undefined for parameter \"" << result.name
        << "\" when the number of pipeline parallel stages is " << pipeline_parallel_stages;
  }
  if (opt_pipeline_stages.has_value()) {
    result.pipeline_stages.reserve(opt_pipeline_stages.value().size());
    for (const tvm::ffi::json::Value& v : opt_pipeline_stages.value()) {
      auto int_opt = v.try_cast<int64_t>();
      CHECK(int_opt.has_value()) << "Pipeline stage is not a integer.";
      result.pipeline_stages.push_back(*int_opt);
    }
  } else {
    result.pipeline_stages = {0};
  }
  return result;
}

ModelMetadata::KVCacheMetadata ModelMetadata::KVCacheMetadata::FromJSON(
    const tvm::ffi::json::Object& json) {
  KVCacheMetadata kv_cache_metadata;
  kv_cache_metadata.num_hidden_layers = json::Lookup<int64_t>(json, "num_hidden_layers");
  kv_cache_metadata.head_dim = json::Lookup<int64_t>(json, "head_dim");
  kv_cache_metadata.num_attention_heads = json::Lookup<int64_t>(json, "num_attention_heads");
  kv_cache_metadata.num_key_value_heads = json::Lookup<int64_t>(json, "num_key_value_heads");
  return kv_cache_metadata;
}

ModelMetadata ModelMetadata::FromJSON(const tvm::ffi::json::Object& metadata,
                                      const tvm::ffi::json::Object& model_config) {
  ModelMetadata result;
  result.model_type = json::Lookup<std::string>(metadata, "model_type");
  result.quantization = json::Lookup<std::string>(metadata, "quantization");
  result.context_window_size = json::Lookup<int64_t>(metadata, "context_window_size");
  result.prefill_chunk_size = json::Lookup<int64_t>(metadata, "prefill_chunk_size");
  result.max_batch_size = json::Lookup<int64_t>(metadata, "max_batch_size");
  if (metadata.count("sliding_window_size"))
    result.sliding_window_size = json::Lookup<int64_t>(metadata, "sliding_window_size");
  if (metadata.count("sliding_window"))  // to be removed after SLM migration
    result.sliding_window_size = json::Lookup<int64_t>(metadata, "sliding_window");
  if (metadata.count("attention_sink_size"))  // remove after sink is decoupled from model lib
    result.attention_sink_size = json::Lookup<int64_t>(metadata, "attention_sink_size");
  result.seqlen_padding_factor =
      json::LookupOrDefault<int64_t>(metadata, "seqlen_padding_factor", 1);
  result.tensor_parallel_shards = json::Lookup<int64_t>(metadata, "tensor_parallel_shards");
  result.pipeline_parallel_stages =
      json::LookupOrDefault<int64_t>(metadata, "pipeline_parallel_stages", 1);
  result.disaggregation = json::LookupOrDefault<bool>(metadata, "disaggregation", false);
  result.kv_state_kind = KVStateKindFromString(
      json::LookupOrDefault<std::string>(metadata, "kv_state_kind", "kv_cache"));
  if (result.kv_state_kind != KVStateKind::kNone &&
      result.kv_state_kind != KVStateKind::kRNNState) {
    result.kv_cache_metadata =
        KVCacheMetadata::FromJSON(json::Lookup<tvm::ffi::json::Object>(metadata, "kv_cache"));
  } else {
    result.kv_cache_metadata = {/*num_hidden_layers=*/0,
                                /*head_dim=*/0,
                                /*num_attention_heads=*/0,
                                /*num_key_value_heads=*/0};
  }
  {
    std::vector<ModelMetadata::Param>& params = result.params;
    tvm::ffi::json::Array json_params = json::Lookup<tvm::ffi::json::Array>(metadata, "params");
    params.reserve(json_params.size());
    for (int i = 0, n = json_params.size(); i < n; ++i) {
      params.emplace_back(ModelMetadata::Param::FromJSON(
          json::Lookup<tvm::ffi::json::Object>(json_params, i), model_config));
    }
  }
  {
    std::unordered_map<std::string, int64_t>& memory_usage = result.memory_usage;
    tvm::ffi::json::Object json_memory_usage =
        json::Lookup<tvm::ffi::json::Object>(metadata, "memory_usage");
    memory_usage.reserve(json_memory_usage.size());
    for (const auto& [key, val] : json_memory_usage) {
      std::string func_name = key.cast<tvm::ffi::String>();
      memory_usage[func_name] = json::Lookup<int64_t>(json_memory_usage, func_name);
    }
  }
  return result;
}

ModelMetadata ModelMetadata::FromModule(Module module, const tvm::ffi::json::Object& model_config) {
  std::string json_str = "";
  Optional<Function> pf = module->GetFunction("_metadata");
  CHECK(pf.defined()) << "ValueError: _metadata function not found in module";
  json_str = pf.value()().cast<String>();
  tvm::ffi::json::Object json = json::ParseToJSONObject(json_str);
  try {
    return ModelMetadata::FromJSON(json, model_config);
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to parse metadata:\n" << json_str << "\nerror: " << e.what();
    throw e;
  }
}

}  // namespace llm
}  // namespace mlc
