/*!
 * Copyright (c) 2023 by Contributors
 * \file utils.h
 * \brief Utility functions.
 */
#include <sstream>
#include <string>
#include <vector>

namespace mlc {
namespace llm {

inline std::vector<std::string> Split(const std::string& str, char delim) {
  std::string item;
  std::istringstream is(str);
  std::vector<std::string> ret;
  while (std::getline(is, item, delim)) {
    ret.push_back(item);
  }
  return ret;
}

}  // namespace llm
}  // namespace mlc
