/*!
 * Copyright (c) 2023-2025 by Contributors
 * \file support/utils.h
 * \brief Utility functions.
 */
#ifndef MLC_LLM_SUPPORT_UTILS_H_
#define MLC_LLM_SUPPORT_UTILS_H_

#include <dmlc/memory_io.h>

#include <sstream>
#include <string>
#include <vector>

#include "../../3rdparty/tvm/src/support/base64.h"

namespace mlc {
namespace llm {

/*! \brief Split the input string by the given delimiter character. */
inline std::vector<std::string> Split(const std::string& str, char delim) {
  std::string item;
  std::istringstream is(str);
  std::vector<std::string> ret;
  while (std::getline(is, item, delim)) {
    ret.push_back(item);
  }
  return ret;
}

/*!
 * \brief Check whether the string starts with a given prefix.
 * \param str The given string.
 * \param prefix The given prefix.
 * \return Whether the prefix matched.
 */
inline bool StartsWith(const std::string& str, const char* prefix) {
  size_t n = str.length();
  for (size_t i = 0; i < n; i++) {
    if (prefix[i] == '\0') return true;
    if (str.data()[i] != prefix[i]) return false;
  }
  // return true if the str is equal to the prefix
  return prefix[n] == '\0';
}

/*!
 * \brief Get the base64 encoded result of a string.
 * \param str The string to encode.
 * \return The base64 encoded string.
 */
inline std::string Base64Encode(std::string str) {
  std::string result;
  dmlc::MemoryStringStream m_stream(&result);
  tvm::support::Base64OutStream b64stream(&m_stream);
  static_cast<dmlc::Stream*>(&b64stream)->Write(str);
  b64stream.Finish();
  return result;
}

/*!
 * \brief Get the base64 decoded result of a string.
 * \param str The string to decode.
 * \return The base64 decoded string.
 */
inline std::string Base64Decode(std::string str) {
  std::string result;
  dmlc::MemoryStringStream m_stream(&str);
  tvm::support::Base64InStream b64stream(&m_stream);
  b64stream.InitPosition();
  static_cast<dmlc::Stream*>(&b64stream)->Read(&result);
  return result;
}

}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SUPPORT_UTILS_H_
