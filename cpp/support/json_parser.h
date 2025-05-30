/*!
 * \file support/json_parser.h
 * \brief Helps to parse JSON strings and objects.
 */
#ifndef MLC_LLM_SUPPORT_JSON_PARSER_H_
#define MLC_LLM_SUPPORT_JSON_PARSER_H_

#include <picojson.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>

#include <optional>

#include "result.h"

namespace mlc {
namespace llm {
namespace json {

/*!
 * \brief Parse a JSON string to a JSON object.
 * \param json_str The JSON string to parse.
 * \return The parsed JSON object.
 */
inline picojson::object ParseToJSONObject(const std::string& json_str) {
  picojson::value result;
  std::string err = picojson::parse(result, json_str);
  CHECK(err.empty()) << "Failed to parse JSON: err. The JSON string is:" << json_str;
  CHECK(result.is<picojson::object>())
      << "ValueError: The given string is not a JSON object: " << json_str;
  return result.get<picojson::object>();
}
/*!
 * \brief Parse a JSON string to a JSON object.
 * \param json_str The JSON string to parse.
 * \return The parsed JSON object, or the error message.
 */
inline Result<picojson::object> ParseToJSONObjectWithResultReturn(const std::string& json_str) {
  using TResult = Result<picojson::object>;
  picojson::value result;
  std::string err = picojson::parse(result, json_str);
  if (!err.empty()) {
    return TResult::Error("Failed to parse JSON: err. The JSON string is: " + json_str +
                          ". The error is " + err);
  }
  if (!result.is<picojson::object>()) {
    return TResult::Error("ValueError: The given string is not a JSON object: " + json_str);
  }
  return TResult::Ok(result.get<picojson::object>());
}

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
/*!
 * \brief Lookup a JSON object by a key, and convert it to a given type.
 * If the key doesn't exist or has null value, the default value is returned.
 * \param json The JSON object to look up.
 * \param key The key to look up.
 * \tparam ValueType The type to be converted to.
 * \return The converted value, or the default value if the key doesn't exist or has null value.
 */
template <typename ValueType>
inline ValueType LookupOrDefault(const picojson::object& json, const std::string& key,
                                 const ValueType& default_value) {
  auto it = json.find(key);
  if (it == json.end() || it->second.is<picojson::null>()) {
    return default_value;
  }
  CHECK(it->second.is<ValueType>()) << "ValueError: key `" << key << "` has unexpected type";
  return it->second.get<ValueType>();
}
/*!
 * \brief Lookup a JSON object by a key, and convert it to a given type.
 * If the key doesn't exist or has null value, return std::nullopt.
 * \param json The JSON object to look up.
 * \param key The key to look up.
 * \tparam ValueType The type to be converted to.
 * \return The converted value, or std::nullopt if the value doesn't exist or has null value.
 */
template <typename ValueType>
inline std::optional<ValueType> LookupOptional(const picojson::object& json,
                                               const std::string& key) {
  auto it = json.find(key);
  if (it == json.end() || it->second.is<picojson::null>()) {
    return std::nullopt;
  }
  CHECK(it->second.is<ValueType>()) << "ValueError: key `" << key << "` has unexpected type";
  return it->second.get<ValueType>();
}
/*!
 * \brief Lookup a JSON object by a key, and convert it to a given type.
 * \param json The JSON object to look up.
 * \param key The key to look up.
 * \tparam ValueType The type to be converted to.
 * \return The converted value, or the error message.
 */
template <typename ValueType>
inline Result<ValueType> LookupWithResultReturn(const picojson::object& json,
                                                const std::string& key) {
  using TResult = Result<ValueType>;
  auto it = json.find(key);
  if (it == json.end()) {
    return TResult::Error("ValueError: key \"" + key + "\" not found in the JSON object");
  }
  if (!it->second.is<ValueType>()) {
    return TResult::Error("ValueError: key \"" + key + "\" has unexpected value type.");
  }
  return TResult::Ok(it->second.get<ValueType>());
}
/*!
 * \brief Lookup a JSON object by a key, and convert it to a given type.
 * If the key doesn't exist or has null value, the default value is returned.
 * \param json The JSON object to look up.
 * \param key The key to look up.
 * \tparam ValueType The type to be converted to.
 * \return The converted value, or the default value if the key doesn't exist or has null value
 * , or the error message.
 */
template <typename ValueType>
inline Result<ValueType> LookupOrDefaultWithResultReturn(const picojson::object& json,
                                                         const std::string& key,
                                                         const ValueType& default_value) {
  using TResult = Result<ValueType>;
  auto it = json.find(key);
  if (it == json.end() || it->second.is<picojson::null>()) {
    return TResult::Ok(default_value);
  }
  if (!it->second.is<ValueType>()) {
    return TResult::Error("ValueError: key \"" + key + "\" has unexpected value type.");
  }
  return TResult::Ok(it->second.get<ValueType>());
}
/*!
 * \brief Lookup a JSON object by a key, and convert it to a given type.
 * If the key doesn't exist or has null value, return std::nullopt.
 * \param json The JSON object to look up.
 * \param key The key to look up.
 * \tparam ValueType The type to be converted to.
 * \return The converted value, or std::nullopt if the value doesn't exist or has null value,
 * , or the error message.
 */
template <typename ValueType>
inline Result<std::optional<ValueType>> LookupOptionalWithResultReturn(const picojson::object& json,
                                                                       const std::string& key) {
  using TResult = Result<std::optional<ValueType>>;
  auto it = json.find(key);
  if (it == json.end() || it->second.is<picojson::null>()) {
    return TResult::Ok(std::nullopt);
  }
  if (!it->second.is<ValueType>()) {
    return TResult::Error("ValueError: key \"" + key + "\" has unexpected value type.");
  }
  return TResult::Ok(it->second.get<ValueType>());
}

// Implementation details

/*! \brief Shape extension to incorporate symbolic shapes. */
struct SymShapeTuple {
  tvm::ffi::Shape shape_values;
  std::vector<std::string> sym_names;

  /*! \brief Convert symbolic shape tuple to static shape tuple with model config. */
  tvm::ffi::Shape ToStatic(const picojson::object& model_config) {
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
    return tvm::ffi::Shape(std::move(shape));
  }
};

namespace details {

inline tvm::runtime::DataType DTypeFromString(const std::string& s) {
  return tvm::runtime::DataType(tvm::runtime::StringToDLDataType(s));
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
  return SymShapeTuple{tvm::ffi::Shape(std::move(result)), sym_names};
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
inline SymShapeTuple LookupOrDefault(const picojson::object& json, const std::string& key,
                                     const SymShapeTuple& default_value) {
  auto it = json.find(key);
  if (it == json.end() || it->second.is<picojson::null>()) {
    return default_value;
  }
  return details::SymShapeTupleFromArray(Lookup<picojson::array>(json, key));
}

template <>
inline SymShapeTuple Lookup(const picojson::array& json, int index) {
  return details::SymShapeTupleFromArray(Lookup<picojson::array>(json, index));
}

}  // namespace json
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SUPPORT_JSON_PARSER_H_
