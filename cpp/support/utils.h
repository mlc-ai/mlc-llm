/*!
 * Copyright (c) 2023 by Contributors
 * \file support/utils.h
 * \brief Utility functions.
 */
#ifndef MLC_LLM_SUPPORT_UTILS_H_
#define MLC_LLM_SUPPORT_UTILS_H_

#include <sstream>
#include <string>
#include <vector>

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

}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SUPPORT_UTILS_H_
