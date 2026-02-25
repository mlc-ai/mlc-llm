/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file support/image_utils.cc
 */
#include "vlm_utils.h"

#include <cmath>

namespace mlc {
namespace llm {

void CalculateResizeShape(tvm::runtime::Tensor image_data, std::string model_type,
                          int* p_target_height, int* p_target_width) {
  TVM_FFI_ICHECK_EQ(image_data->shape[3], 3) << "Image format must be NHWC";
  int height = image_data->shape[1];
  int width = image_data->shape[2];

  if ("phi3_v" == model_type) {
    const int hd_num = 4;
    double ratio = static_cast<double>(width) / height;
    int scale = 1;
    while (scale * std::ceil(scale / ratio) <= hd_num) {
      scale += 1;
    }
    scale -= 1;
    *p_target_width = static_cast<int>(scale * 336);
    *p_target_height = static_cast<int>(*p_target_width / ratio);
  }
}

void CalculatePadShape(tvm::runtime::Tensor image_data, std::string model_type, int* p_pad_height,
                       int* p_pad_width) {
  TVM_FFI_ICHECK_EQ(image_data->shape[3], 3) << "Image format must be NHWC";
  if ("phi3_v" == model_type) {
    int resized_height = 0, resized_width = 0;
    CalculateResizeShape(image_data, model_type, &resized_height, &resized_width);
    int tar = (int)(ceil(resized_height / 336.0) * 336);
    int top_padding = (int)((tar - resized_height) / 2);
    int bottom_padding = tar - resized_height - top_padding;
    TVM_FFI_ICHECK_EQ(tar, resized_height + top_padding + bottom_padding)
        << "Padding size not equal!";
    *p_pad_height = tar;
    *p_pad_width = resized_width;
  }
}

void CalculateCropShape(tvm::runtime::Tensor image_data, std::string model_type, int* p_crop_height,
                        int* p_crop_width) {
  TVM_FFI_ICHECK_EQ(image_data->shape[3], 3) << "Image format must be NHWC";
  if ("phi3_v" == model_type) {
    int pad_h = 0, pad_w = 0;
    CalculatePadShape(image_data, model_type, &pad_h, &pad_w);
    *p_crop_height = pad_h / 336;
    *p_crop_width = pad_w / 336;
  }
}

}  // namespace llm
}  // namespace mlc
