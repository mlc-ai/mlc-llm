#ifndef MLC_LLM_CPP_JSON_PARSER_H_
#define MLC_LLM_CPP_JSON_PARSER_H_

#define PICOJSON_USE_INT64
#define __STDC_FORMAT_MACROS

#include <picojson.h>
#include <tvm/runtime/container/shape_tuple.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>

namespace mlc {
namespace llm {
namespace json {

template <typename ValueType>
inline ValueType Lookup(const picojson::object& json, const std::string& key) {
  auto it = json.find(key);
  CHECK(it != json.end()) << "ValueError: key `" << key << "` not found in the JSON object";
  CHECK(it->second.is<ValueType>()) << "ValueError: key `" << key << "` has unexpected type";
  return it->second.get<ValueType>();
}

template <>
inline tvm::runtime::DataType Lookup(const picojson::object& json, const std::string& key) {
  return tvm::runtime::DataType(tvm::runtime::String2DLDataType(Lookup<std::string>(json, key)));
}

template <>
inline tvm::runtime::ShapeTuple Lookup(const picojson::object& json, const std::string& key) {
  picojson::array shape = Lookup<picojson::array>(json, key);
  std::vector<int64_t> result;
  result.reserve(shape.size());
  for (const picojson::value& dim : shape) {
    CHECK(dim.is<int64_t>()) << "ValueError: key `" << key << "` has unexpected type";
    result.push_back(dim.get<int64_t>());
  }
  return tvm::runtime::ShapeTuple(std::move(result));
}

inline picojson::object ParseObject(const std::string& json_str) {
  picojson::value result;
  std::string err = picojson::parse(result, json_str);
  if (!err.empty()) {
    LOG(FATAL) << "Failed to parse JSON: err. The JSON string is:" << json_str;
  }
  CHECK(result.is<picojson::object>())
      << "ValueError: The given string is not a JSON object: " << json_str;
  return result.get<picojson::object>();
}

inline picojson::object AsJSONObject(const picojson::value& json) {
  CHECK(json.is<picojson::object>()) << "ValueError: The given value is not a JSON object";
  return json.get<picojson::object>();
}

}  // namespace json
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_CPP_JSON_PARSER_H_
