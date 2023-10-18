/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/tokenizer.h
 * \brief The header for runtime module of tokenizer encode/decode functions.
 */

#ifndef MLC_LLM_SERVE_TOKENIZER_H_
#define MLC_LLM_SERVE_TOKENIZER_H_

#include <tvm/runtime/container/string.h>
#include <tvm/runtime/module.h>

#include "../base.h"

namespace mlc {
namespace llm {
namespace serve {

using tvm::Device;
using namespace tvm::runtime;

/*!
 * \brief Create the runtime module for tokenizer encode/decode functions.
 * \param model_path The path to the model weights which also contains the tokenizer.
 * \return The created runtime module.
 */
MLC_LLM_DLL tvm::runtime::Module CreateTokenizerModule(String model_path);

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_TOKENIZER_H_
