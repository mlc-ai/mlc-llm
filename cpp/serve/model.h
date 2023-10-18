/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/model.h
 * \brief The header for runtime module of LLM functions (prefill/decode/etc.)
 */

#ifndef MLC_LLM_SERVE_MODEL_H_
#define MLC_LLM_SERVE_MODEL_H_

#include <tvm/runtime/container/string.h>
#include <tvm/runtime/module.h>

#include "../base.h"

namespace mlc {
namespace llm {
namespace serve {

using tvm::Device;
using namespace tvm::runtime;

/*!
 * \brief Create the runtime module for LLM functions.
 * \param reload_lib The model library. It might be a path to the binary
 * file or an executable module that is pre-loaded.
 * \param model_path The path to the model weight parameters.
 * \param device The device to run the model on.
 * \return The created runtime module.
 */
MLC_LLM_DLL tvm::runtime::Module CreateModelModule(TVMArgValue reload_lib, String model_path,
                                                   DLDevice device);

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_MODEL_H_
