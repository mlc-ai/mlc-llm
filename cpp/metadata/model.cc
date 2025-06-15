#include "./model.h"

#include <unordered_map>

#include "../support/json_parser.h"

namespace mlc {
namespace llm {

using namespace tvm::runtime;
using tvm::ffi::TypedFunction;

ModelMetadata::Param::Preproc ModelMetadata::Param::Preproc::FromJSON(
    const picojson::object& js, const picojson::object& model_config) {
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

ModelMetadata::Param ModelMetadata::Param::FromJSON(const picojson::object& param,
                                                    const picojson::object& model_config) {
  Param result;
  result.name = json::Lookup<std::string>(param, "name");
  result.dtype = json::Lookup<DataType>(param, "dtype");
  // A shape being `-1` means that it is dynamic
  json::SymShapeTuple sym_shape = json::Lookup<json::SymShapeTuple>(param, "shape");
  result.shape = sym_shape.ToStatic(model_config);
  // - "preproc"
  picojson::array preprocs = json::Lookup<picojson::array>(param, "preprocs");
  result.preprocs.reserve(preprocs.size());
  for (int i = 0; i < preprocs.size(); i++) {
    result.preprocs.emplace_back(ModelMetadata::Param::Preproc::FromJSON(
        json::Lookup<picojson::object>(preprocs, i), model_config));
  }
  // - "pipeline_stages"
  int pipeline_parallel_stages =
      json::LookupOrDefault<int64_t>(model_config, "pipeline_parallel_stages", 1);
  std::optional<picojson::array> opt_pipeline_stages =
      json::LookupOptional<picojson::array>(param, "pipeline_stages");
  if (pipeline_parallel_stages > 1) {
    CHECK(opt_pipeline_stages.has_value())
        << "The pipeline stage is undefined for parameter \"" << result.name
        << "\" when the number of pipeline parallel stages is " << pipeline_parallel_stages;
  }
  if (opt_pipeline_stages.has_value()) {
    result.pipeline_stages.reserve(opt_pipeline_stages.value().size());
    for (const picojson::value& v : opt_pipeline_stages.value()) {
      CHECK(v.is<int64_t>()) << "Pipeline stage is not a integer.";
      result.pipeline_stages.push_back(v.get<int64_t>());
    }
  } else {
    result.pipeline_stages = {0};
  }
  return result;
}

ModelMetadata::KVCacheMetadata ModelMetadata::KVCacheMetadata::FromJSON(
    const picojson::object& json) {
  KVCacheMetadata kv_cache_metadata;
  kv_cache_metadata.num_hidden_layers = json::Lookup<int64_t>(json, "num_hidden_layers");
  kv_cache_metadata.head_dim = json::Lookup<int64_t>(json, "head_dim");
  kv_cache_metadata.num_attention_heads = json::Lookup<int64_t>(json, "num_attention_heads");
  kv_cache_metadata.num_key_value_heads = json::Lookup<int64_t>(json, "num_key_value_heads");
  return kv_cache_metadata;
}

ModelMetadata ModelMetadata::FromJSON(const picojson::object& metadata,
                                      const picojson::object& model_config) {
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
  result.tensor_parallel_shards = json::Lookup<int64_t>(metadata, "tensor_parallel_shards");
  result.pipeline_parallel_stages =
      json::LookupOrDefault<int64_t>(metadata, "pipeline_parallel_stages", 1);
  result.disaggregation = json::LookupOrDefault<bool>(metadata, "disaggregation", false);
  result.kv_state_kind = KVStateKindFromString(
      json::LookupOrDefault<std::string>(metadata, "kv_state_kind", "kv_cache"));
  if (result.kv_state_kind != KVStateKind::kNone &&
      result.kv_state_kind != KVStateKind::kRNNState) {
    result.kv_cache_metadata =
        KVCacheMetadata::FromJSON(json::Lookup<picojson::object>(metadata, "kv_cache"));
  } else {
    result.kv_cache_metadata = {/*num_hidden_layers=*/0,
                                /*head_dim=*/0,
                                /*num_attention_heads=*/0,
                                /*num_key_value_heads=*/0};
  }
  {
    std::vector<ModelMetadata::Param>& params = result.params;
    picojson::array json_params = json::Lookup<picojson::array>(metadata, "params");
    params.reserve(json_params.size());
    for (int i = 0, n = json_params.size(); i < n; ++i) {
      params.emplace_back(ModelMetadata::Param::FromJSON(
          json::Lookup<picojson::object>(json_params, i), model_config));
    }
  }
  {
    std::unordered_map<std::string, int64_t>& memory_usage = result.memory_usage;
    picojson::object json_memory_usage = json::Lookup<picojson::object>(metadata, "memory_usage");
    memory_usage.reserve(json_memory_usage.size());
    for (const auto& [func_name, _] : json_memory_usage) {
      memory_usage[func_name] = json::Lookup<int64_t>(json_memory_usage, func_name);
    }
  }
  return result;
}

ModelMetadata ModelMetadata::FromModule(tvm::runtime::Module module,
                                        const picojson::object& model_config) {
  std::string json_str = "";
  TypedFunction<String()> pf = module.GetFunction("_metadata");
  json_str = pf();
  picojson::object json = json::ParseToJSONObject(json_str);
  try {
    return ModelMetadata::FromJSON(json, model_config);
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to parse metadata:\n" << json_str << "\nerror: " << e.what();
    throw e;
  }
}

}  // namespace llm
}  // namespace mlc
