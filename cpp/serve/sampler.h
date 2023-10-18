/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/sampler.h
 * \brief The header for runtime module of sampler functions.
 */

#ifndef MLC_LLM_SERVE_SAMPLER_H_
#define MLC_LLM_SERVE_SAMPLER_H_

#include <tvm/runtime/container/string.h>
#include <tvm/runtime/module.h>

#include "../base.h"

namespace mlc {
namespace llm {
namespace serve {

using tvm::Device;
using namespace tvm::runtime;

/*!
 * \brief Create the runtime module for sampler functions.
 * \param device The device to run the sampling-related functions on.
 * \return The created runtime module.
 */
MLC_LLM_DLL tvm::runtime::Module CreateSamplerModule(DLDevice device);

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_SAMPLER_H_
