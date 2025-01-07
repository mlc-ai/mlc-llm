/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file support/vlm_utils.h
 * \brief Tools for debug purposes.
 */
#ifndef MLC_LLM_SUPPORT_VLM_UTILS_H_
#define MLC_LLM_SUPPORT_VLM_UTILS_H_

#include <tvm/runtime/ndarray.h>

#include <string>

namespace mlc {
namespace llm {

/*!
 * \brief Calculate the target height and width for resizing an image based on the input data and
 * model type. \param image_data The input image data as a TVM NDArray. \param model_type The type
 * of the model influencing the resizing parameters (e.g., phi3v). \param target_height Reference to
 * the variable where the calculated target height will be stored. \param target_width Reference to
 * the variable where the calculated target width will be stored.
 */
void CalculateResizeShape(tvm::runtime::NDArray image_data, std::string model_type,
                          int* p_target_height, int* p_target_width);
/*!
 * \brief Calculate the padding height and width for an image based on the input data and model
 * type. \param image_data The input image data as a TVM NDArray. \param model_type The type of the
 * model influencing the padding parameters (e.g., phi3v). \param pad_height Reference to the
 * variable where the calculated padding height will be stored. \param pad_width Reference to the
 * variable where the calculated padding width will be stored.
 */
void CalculatePadShape(tvm::runtime::NDArray image_data, std::string model_type, int* p_pad_height,
                       int* p_pad_width);

/*!
 * \brief Calculate the cropping height and width for an image based on the input data and model
 * type. \param image_data The input image data as a TVM NDArray. \param model_type The type of the
 * model influencing the cropping parameters (e.g., phi3v). \param crop_height Reference to the
 * variable where the calculated cropping height will be stored. \param crop_width Reference to the
 * variable where the calculated cropping width will be stored.
 */
void CalculateCropShape(tvm::runtime::NDArray image_data, std::string model_type,
                        int* p_crop_height, int* p_crop_width);

}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SUPPORT_IMAGE_UTILS_H_
