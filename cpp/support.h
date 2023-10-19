/*!
 *  Copyright (c) 2023 by Contributors
 * \file support.h
 * \brief Header of utilities.
 */

#ifndef MLC_LLM_COMMON_H_
#define MLC_LLM_COMMON_H_

#include <fstream>
#include <string>

namespace mlc {
namespace llm {

inline std::string LoadBytesFromFile(const std::string& path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  ICHECK(!fs.fail()) << "Cannot open " << path;
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_COMMON_H_
