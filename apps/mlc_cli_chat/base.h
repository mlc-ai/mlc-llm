/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file base.h
 */

#ifndef MLC_CLI_CHAT_BASE_H
#define MLC_CLI_CHAT_BASE_H

#include <dlpack/dlpack.h>

#include <string>
#include <unordered_map>

struct Message {
  std::unordered_map<std::string, std::string> content;
};

#endif
