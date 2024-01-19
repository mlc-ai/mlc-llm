/*!
 * \file json_parser.h
 * \brief Helps to parse JSON strings and objects.
 */
#ifndef MLC_LLM_CPP_JSON_PARSER_H_
#define MLC_LLM_CPP_JSON_PARSER_H_

#include <picojson.h>
#include <tvm/runtime/container/shape_tuple.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>

namespace mlc {
namespace llm {
namespace json {

/*!
 * \brief Parse a JSON string to a JSON object.
 * \param json_str The JSON string to parse.
 * \return The parsed JSON object.
 */
picojson::object ParseToJsonObject(const std::string& json_str);
/*!
 * \brief Lookup a JSON object by a key, and convert it to a given type.
 * \param json The JSON object to look up.
 * \param key The key to look up.
 * \tparam ValueType The type to be converted to.
 * \return The converted value.
 */
template <typename ValueType>
ValueType Lookup(const picojson::object& json, const std::string& key);
/*!
 * \brief Lookup a JSON array by an index, and convert it to a given type.
 * \param json The JSON array to look up.
 * \param index The index to look up.
 * \tparam ValueType The type to be converted to.
 * \return The converted value.
 */
template <typename ValueType>
ValueType Lookup(const picojson::array& json, int index);

/*! \brief ShapeTuple extension to incorporate symbolic shapes. */
struct SymShapeTuple {
  tvm::runtime::ShapeTuple shape_values;
  std::vector<std::string> sym_names;

  /*! \brief Convert symbolic shape tuple to static shape tuple with model config. */
  tvm::runtime::ShapeTuple ToStatic(const picojson::object& model_config) {
    std::vector<int64_t> shape;
    shape.reserve(shape_values.size());
    for (int i = 0; i < static_cast<int>(shape_values.size()); ++i) {
      if (shape_values[i] != -1) {
        shape.push_back(shape_values[i]);
      } else {
        CHECK(model_config.at(sym_names[i]).is<int64_t>())
            << "ValueError: model config is expected to contain \"" << sym_names[i]
            << "\" as an integer. However, the given config has unexpected type for \""
            << sym_names[i] << "\".";
        shape.push_back(model_config.at(sym_names[i]).get<int64_t>());
      }
    }
    return tvm::runtime::ShapeTuple(std::move(shape));
  }
};

// Implementation details

namespace details {

inline tvm::runtime::DataType DTypeFromString(const std::string& s) {
  return tvm::runtime::DataType(tvm::runtime::String2DLDataType(s));
}

inline SymShapeTuple SymShapeTupleFromArray(const picojson::array& shape) {
  std::vector<int64_t> result;
  std::vector<std::string> sym_names;
  result.reserve(shape.size());
  sym_names.reserve(shape.size());
  for (int i = 0; i < static_cast<int>(shape.size()); ++i) {
    const picojson::value& dim = shape[i];
    if (dim.is<std::string>()) {
      result.push_back(-1);
      sym_names.push_back(dim.get<std::string>());
    } else {
      CHECK(dim.is<int64_t>()) << "ValueError: shape has unexpected type";
      result.push_back(dim.get<int64_t>());
      sym_names.push_back("");
    }
  }
  return SymShapeTuple{tvm::runtime::ShapeTuple(std::move(result)), sym_names};
}

}  // namespace details

template <typename ValueType>
inline ValueType Lookup(const picojson::object& json, const std::string& key) {
  auto it = json.find(key);
  CHECK(it != json.end()) << "ValueError: key `" << key << "` not found in the JSON object";
  CHECK(it->second.is<ValueType>()) << "ValueError: key `" << key << "` has unexpected type";
  return it->second.get<ValueType>();
}

template <typename ValueType>
inline ValueType Lookup(const picojson::array& json, int index) {
  CHECK(index < json.size()) << "IndexError: json::array index out of range";
  auto value = json.at(index);
  CHECK(value.is<ValueType>()) << "ValueError: value at index `" << index
                               << "` has unexpected type";
  return value.get<ValueType>();
}

template <>
inline tvm::runtime::DataType Lookup(const picojson::object& json, const std::string& key) {
  return details::DTypeFromString(Lookup<std::string>(json, key));
}

template <>
inline tvm::runtime::DataType Lookup(const picojson::array& json, int index) {
  return details::DTypeFromString(Lookup<std::string>(json, index));
}

template <>
inline SymShapeTuple Lookup(const picojson::object& json, const std::string& key) {
  return details::SymShapeTupleFromArray(Lookup<picojson::array>(json, key));
}

template <>
inline SymShapeTuple Lookup(const picojson::array& json, int index) {
  return details::SymShapeTupleFromArray(Lookup<picojson::array>(json, index));
}

inline picojson::object ParseToJsonObject(const std::string& json_str) {
  picojson::value result;
  std::string err = picojson::parse(result, json_str);
  if (!err.empty()) {
    LOG(FATAL) << "Failed to parse JSON: err. The JSON string is:" << json_str;
  }
  CHECK(result.is<picojson::object>())
      << "ValueError: The given string is not a JSON object: " << json_str;
  return result.get<picojson::object>();
}

}  // namespace json
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_CPP_JSON_PARSER_H_
