#define __STDC_FORMAT_MACROS
#include "./model_metadata.h"

#include <tvm/runtime/packed_func.h>

#include "./json_parser.h"

namespace mlc {
namespace llm {

using namespace tvm::runtime;

ModelMetadata::Param ModelMetadata::Param::FromJSON(const picojson::object& param) {
  Param result;
  result.name = json::Lookup<std::string>(param, "name");
  result.shape = json::Lookup<ShapeTuple>(param, "shape");
  result.dtype = json::Lookup<DataType>(param, "dtype");
  return result;
}

ModelMetadata ModelMetadata::FromJSON(const picojson::object& metadata) {
  ModelMetadata result;
  result.model_type = json::Lookup<std::string>(metadata, "model_type");
  result.quantization = json::Lookup<std::string>(metadata, "quantization");
  picojson::array params = json::Lookup<picojson::array>(metadata, "params");
  result.params.reserve(params.size());
  for (const picojson::value& json_param : params) {
    result.params.emplace_back(ModelMetadata::Param::FromJSON(json::AsJSONObject(json_param)));
  }
  return result;
}

ModelMetadata ModelMetadata::FromModule(tvm::runtime::Module module) {
  std::string json_str = "";
  try {
    TypedPackedFunc<String()> pf = module.GetFunction("_metadata");
    ICHECK(pf != nullptr);
    json_str = pf();
  } catch (...) {
    return ModelMetadata();  // TODO: add a warning message about legacy usecases
  }
  picojson::object json = json::ParseObject(json_str);
  try {
    return ModelMetadata::FromJSON(json);
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to parse metadata:\n" << json_str;
    throw e;
  }
}

}  // namespace llm
}  // namespace mlc
