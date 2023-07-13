/*!
 *  Copyright (c) 2023 by Contributors
 * \file image_embed.h
 * \brief Implementation of image embedding pipeline.
 */
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/module.h>

#ifndef MLC_LLM_DLL
#ifdef _WIN32
#ifdef MLC_LLM_EXPORTS
#define MLC_LLM_DLL __declspec(dllexport)
#else
#define MLC_LLM_DLL __declspec(dllimport)
#endif
#else
#define MLC_LLM_DLL __attribute__((visibility("default")))
#endif
#endif

namespace mlc {
namespace llm {

// explicit export via TVM_DLL
MLC_LLM_DLL tvm::runtime::Module CreateImageModule(DLDevice device);

}  // namespace llm
}  // namespace mlc
