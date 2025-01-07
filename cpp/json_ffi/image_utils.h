/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file json_ffi/image_utils.h
 * \brief The header of Image utils for JSON FFI Engine in MLC LLM.
 */
#ifndef MLC_LLM_JSON_FFI_IMAGE_UTILS_H_
#define MLC_LLM_JSON_FFI_IMAGE_UTILS_H_

#include <tvm/runtime/ndarray.h>

#include <optional>
#include <string>

#include "../support/result.h"

namespace mlc {
namespace llm {
namespace json_ffi {

/*! \brief Load a base64 encoded image string into a CPU NDArray of shape {height, width, 3} */
Result<tvm::runtime::NDArray> LoadImageFromBase64(const std::string& base64_str);

/*! \brief Preprocess the CPU image for CLIP encoder and return an NDArray on the given device */
tvm::runtime::NDArray ClipPreprocessor(tvm::runtime::NDArray image_data, int target_size,
                                       DLDevice device);

}  // namespace json_ffi
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_JSON_FFI_IMAGE_UTILS_H_
