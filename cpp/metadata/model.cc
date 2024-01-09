#include "./model.h"

#include <tvm/runtime/packed_func.h>

#include <unordered_map>

#include "./json_parser.h"

namespace mlc {
namespace llm {

using namespace tvm::runtime;

ModelMetadata::Param::Preproc ModelMetadata::Param::Preproc::FromJSON(const picojson::object& js) {
  Preproc preproc;
  CHECK_EQ(js.size(), 3) << "ValueError: Invalid preprocessing info in JSON";
  preproc.func_name = json::Lookup<std::string>(js, "func_name");
  preproc.out_shape = json::Lookup<ShapeTuple>(js, "out_shape");
  preproc.out_dtype = json::Lookup<DataType>(js, "out_dtype");
  return preproc;
}

ModelMetadata::Param ModelMetadata::Param::FromJSON(const picojson::object& param) {
  Param result;
  result.name = json::Lookup<std::string>(param, "name");
  result.shape = json::Lookup<ShapeTuple>(param, "shape");
  result.dtype = json::Lookup<DataType>(param, "dtype");
  picojson::array preprocs = json::Lookup<picojson::array>(param, "preprocs");
  result.preprocs.reserve(preprocs.size());
  for (int i = 0; i < preprocs.size(); i++) {
    result.preprocs.emplace_back(
        ModelMetadata::Param::Preproc::FromJSON(json::Lookup<picojson::object>(preprocs, i)));
  }
  return result;
}

ModelMetadata ModelMetadata::FromJSON(const picojson::object& metadata) {
  ModelMetadata result;
  result.model_type = json::Lookup<std::string>(metadata, "model_type");
  result.quantization = json::Lookup<std::string>(metadata, "quantization");
  result.context_window_size = json::Lookup<int64_t>(metadata, "context_window_size");
  result.prefill_chunk_size = json::Lookup<int64_t>(metadata, "prefill_chunk_size");
  if (metadata.count("sliding_window_size"))
    result.sliding_window_size = json::Lookup<int64_t>(metadata, "sliding_window_size");
  if (metadata.count("sliding_window"))  // to be removed after SLM migration
    result.sliding_window_size = json::Lookup<int64_t>(metadata, "sliding_window");
  if (metadata.count("attention_sink_size"))  // remove after sink is decoupled from model lib
    result.attention_sink_size = json::Lookup<int64_t>(metadata, "attention_sink_size");
  result.tensor_parallel_shards = json::Lookup<int64_t>(metadata, "tensor_parallel_shards");
  {
    std::vector<ModelMetadata::Param>& params = result.params;
    picojson::array json_params = json::Lookup<picojson::array>(metadata, "params");
    params.reserve(json_params.size());
    for (int i = 0, n = json_params.size(); i < n; ++i) {
      params.emplace_back(
          ModelMetadata::Param::FromJSON(json::Lookup<picojson::object>(json_params, i)));
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

ModelMetadata ModelMetadata::FromModule(tvm::runtime::Module module) {
  std::string json_str = "";
  try {
    TypedPackedFunc<String()> pf = module.GetFunction("_metadata");
    ICHECK(pf != nullptr) << "Unable to find `_metadata` function.";
    json_str = pf();
  } catch (...) {
    return ModelMetadata();  // TODO: add a warning message about legacy usecases
  }
  picojson::object json = json::ParseToJsonObject(json_str);
  try {
    return ModelMetadata::FromJSON(json);
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to parse metadata:\n" << json_str;
    throw e;
  }
}

}  // namespace llm
}  // namespace mlc
