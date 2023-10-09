/*!
 *  Copyright (c) 2023 by Contributors
 * \file llm_chat.cc
 * \brief Implementation of llm chat.
 */
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/module.h>

#include "base.h"

namespace mlc {
namespace llm {

// explicit export via TVM_DLL
MLC_LLM_DLL std::string GetDeltaMessage(std::string curr_message, std::string new_message);

MLC_LLM_DLL tvm::runtime::Module CreateChatModule(DLDevice device);

}  // namespace llm
}  // namespace mlc
