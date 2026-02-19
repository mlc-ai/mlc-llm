/*!
 * \file support/json_parser.h
 * \brief Helps to parse JSON strings and objects.
 */
#ifndef MLC_LLM_SUPPORT_JSON_PARSER_H_
#define MLC_LLM_SUPPORT_JSON_PARSER_H_

#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>

#include <optional>

#include "result.h"

namespace mlc {
namespace llm {
namespace json {

using ::tvm::ffi::json::Array;
using ::tvm::ffi::json::Object;
using ::tvm::ffi::json::Value;

/*!
 * \brief Parse a JSON string to a JSON object.
 * \param json_str The JSON string to parse.
 * \return The parsed JSON object.
 */
inline Object ParseToJSONObject(const std::string& json_str) {
  tvm::ffi::String err;
  Value result = ::tvm::ffi::json::Parse(json_str, &err);
  CHECK(err.empty()) << "Failed to parse JSON: err. The JSON string is:" << json_str;
  auto opt = result.try_cast<Object>();
  CHECK(opt.has_value()) << "ValueError: The given string is not a JSON object: " << json_str;
  return *opt;
}
/*!
 * \brief Parse a JSON string to a JSON object.
 * \param json_str The JSON string to parse.
 * \return The parsed JSON object, or the error message.
 */
inline Result<Object> ParseToJSONObjectWithResultReturn(const std::string& json_str) {
  using TResult = Result<Object>;
  tvm::ffi::String err;
  Value result = ::tvm::ffi::json::Parse(json_str, &err);
  if (!err.empty()) {
    return TResult::Error("Failed to parse JSON: err. The JSON string is: " + json_str +
                          ". The error is " + std::string(err));
  }
  auto opt = result.try_cast<Object>();
  if (!opt.has_value()) {
    return TResult::Error("ValueError: The given string is not a JSON object: " + json_str);
  }
  return TResult::Ok(*opt);
}

/*!
 * \brief Lookup a JSON object by a key, and convert it to a given type.
 * \param json The JSON object to look up.
 * \param key The key to look up.
 * \tparam ValueType The type to be converted to.
 * \return The converted value.
 */
template <typename ValueType>
ValueType Lookup(const Object& json, const std::string& key);
/*!
 * \brief Lookup a JSON array by an index, and convert it to a given type.
 * \param json The JSON array to look up.
 * \param index The index to look up.
 * \tparam ValueType The type to be converted to.
 * \return The converted value.
 */
template <typename ValueType>
ValueType Lookup(const Array& json, int index);
/*!
 * \brief Lookup a JSON object by a key, and convert it to a given type.
 * If the key doesn't exist or has null value, the default value is returned.
 * \param json The JSON object to look up.
 * \param key The key to look up.
 * \tparam ValueType The type to be converted to.
 * \return The converted value, or the default value if the key doesn't exist or has null value.
 */
template <typename ValueType>
inline ValueType LookupOrDefault(const Object& json, const std::string& key,
                                 const ValueType& default_value) {
  if (json.count(key) == 0 || json.at(key) == nullptr) {
    return default_value;
  }
  auto opt = json.at(key).try_cast<ValueType>();
  CHECK(opt.has_value()) << "ValueError: key `" << key << "` has unexpected type";
  return *opt;
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
inline std::optional<ValueType> LookupOptional(const Object& json, const std::string& key) {
  if (json.count(key) == 0 || json.at(key) == nullptr) {
    return std::nullopt;
  }
  auto opt = json.at(key).try_cast<ValueType>();
  CHECK(opt.has_value()) << "ValueError: key `" << key << "` has unexpected type";
  return *opt;
}
/*!
 * \brief Lookup a JSON object by a key, and convert it to a given type.
 * \param json The JSON object to look up.
 * \param key The key to look up.
 * \tparam ValueType The type to be converted to.
 * \return The converted value, or the error message.
 */
template <typename ValueType>
inline Result<ValueType> LookupWithResultReturn(const Object& json, const std::string& key) {
  using TResult = Result<ValueType>;
  if (json.count(key) == 0) {
    return TResult::Error("ValueError: key \"" + key + "\" not found in the JSON object");
  }
  auto opt = json.at(key).try_cast<ValueType>();
  if (!opt.has_value()) {
    return TResult::Error("ValueError: key \"" + key + "\" has unexpected value type.");
  }
  return TResult::Ok(*opt);
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
inline Result<ValueType> LookupOrDefaultWithResultReturn(const Object& json, const std::string& key,
                                                         const ValueType& default_value) {
  using TResult = Result<ValueType>;
  if (json.count(key) == 0 || json.at(key) == nullptr) {
    return TResult::Ok(default_value);
  }
  auto opt = json.at(key).try_cast<ValueType>();
  if (!opt.has_value()) {
    return TResult::Error("ValueError: key \"" + key + "\" has unexpected value type.");
  }
  return TResult::Ok(*opt);
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
inline Result<std::optional<ValueType>> LookupOptionalWithResultReturn(const Object& json,
                                                                       const std::string& key) {
  using TResult = Result<std::optional<ValueType>>;
  if (json.count(key) == 0 || json.at(key) == nullptr) {
    return TResult::Ok(std::nullopt);
  }
  auto opt = json.at(key).try_cast<ValueType>();
  if (!opt.has_value()) {
    return TResult::Error("ValueError: key \"" + key + "\" has unexpected value type.");
  }
  return TResult::Ok(*opt);
}

// Implementation details

/*! \brief Shape extension to incorporate symbolic shapes. */
struct SymShapeTuple {
  tvm::ffi::Shape shape_values;
  std::vector<std::string> sym_names;

  /*! \brief Convert symbolic shape tuple to static shape tuple with model config. */
  tvm::ffi::Shape ToStatic(const Object& model_config) {
    std::vector<int64_t> shape;
    shape.reserve(shape_values.size());
    for (int i = 0; i < static_cast<int>(shape_values.size()); ++i) {
      if (shape_values[i] != -1) {
        shape.push_back(shape_values[i]);
      } else {
        auto opt = model_config.at(sym_names[i]).try_cast<int64_t>();
        CHECK(opt.has_value())
            << "ValueError: model config is expected to contain \"" << sym_names[i]
            << "\" as an integer. However, the given config has unexpected type for \""
            << sym_names[i] << "\".";
        shape.push_back(*opt);
      }
    }
    return tvm::ffi::Shape(std::move(shape));
  }
};

namespace details {

inline tvm::runtime::DataType DTypeFromString(const std::string& s) {
  return tvm::runtime::DataType(tvm::runtime::StringToDLDataType(s));
}

inline SymShapeTuple SymShapeTupleFromArray(const Array& shape) {
  std::vector<int64_t> result;
  std::vector<std::string> sym_names;
  result.reserve(shape.size());
  sym_names.reserve(shape.size());
  for (int i = 0; i < static_cast<int>(shape.size()); ++i) {
    const auto& dim = shape[i];
    auto str_opt = dim.try_cast<std::string>();
    if (str_opt.has_value()) {
      result.push_back(-1);
      sym_names.push_back(*str_opt);
    } else {
      auto int_opt = dim.try_cast<int64_t>();
      CHECK(int_opt.has_value()) << "ValueError: shape has unexpected type";
      result.push_back(*int_opt);
      sym_names.push_back("");
    }
  }
  return SymShapeTuple{tvm::ffi::Shape(std::move(result)), sym_names};
}

}  // namespace details

template <typename ValueType>
inline ValueType Lookup(const Object& json, const std::string& key) {
  CHECK(json.count(key) != 0) << "ValueError: key `" << key << "` not found in the JSON object";
  auto opt = json.at(key).try_cast<ValueType>();
  CHECK(opt.has_value()) << "ValueError: key `" << key << "` has unexpected type";
  return *opt;
}

template <typename ValueType>
inline ValueType Lookup(const Array& json, int index) {
  CHECK(index < static_cast<int>(json.size())) << "IndexError: json::array index out of range";
  auto opt = json[index].try_cast<ValueType>();
  CHECK(opt.has_value()) << "ValueError: value at index `" << index << "` has unexpected type";
  return *opt;
}

template <>
inline tvm::runtime::DataType Lookup(const Object& json, const std::string& key) {
  return details::DTypeFromString(Lookup<std::string>(json, key));
}

template <>
inline tvm::runtime::DataType Lookup(const Array& json, int index) {
  return details::DTypeFromString(Lookup<std::string>(json, index));
}

template <>
inline SymShapeTuple Lookup(const Object& json, const std::string& key) {
  return details::SymShapeTupleFromArray(Lookup<Array>(json, key));
}

template <>
inline SymShapeTuple LookupOrDefault(const Object& json, const std::string& key,
                                     const SymShapeTuple& default_value) {
  if (json.count(key) == 0 || json.at(key) == nullptr) {
    return default_value;
  }
  return details::SymShapeTupleFromArray(Lookup<Array>(json, key));
}

template <>
inline SymShapeTuple Lookup(const Array& json, int index) {
  return details::SymShapeTupleFromArray(Lookup<Array>(json, index));
}

}  // namespace json
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SUPPORT_JSON_PARSER_H_
