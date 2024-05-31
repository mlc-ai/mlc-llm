#include "image_utils.h"

#include <dmlc/io.h>

#include "../../3rdparty/tvm/src/support/base64.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace mlc {
namespace llm {
namespace json_ffi {

using namespace tvm::runtime;

class MemoryBufferStream : public dmlc::Stream {
 public:
  MemoryBufferStream(const char* data, size_t size) : data_(data), size_(size), pos_(0) {}

  size_t Read(void* ptr, size_t size) override {
    size_t remaining = size_ - pos_;
    if (size > remaining) {
      size = remaining;
    }
    if (size == 0) {
      return 0;
    }
    std::memcpy(ptr, data_ + pos_, size);
    pos_ += size;
    return size;
  }

  size_t Write(const void* ptr, size_t size) override {
    LOG(FATAL) << "MemoryBufferStream does not support write";
  }

 private:
  const char* data_;
  size_t size_;
  size_t pos_;
};

size_t Base64DecodedSize(const std::string& base64_str) {
  size_t len = base64_str.size();
  size_t padding = 0;
  if (base64_str[len - 1] == '=') {
    padding++;
  }
  if (base64_str[len - 2] == '=') {
    padding++;
  }
  return 3 * len / 4 - padding;
}

Result<NDArray> LoadImageFromBase64(const std::string& base64_str) {
  using TResult = Result<NDArray>;
  MemoryBufferStream stream(base64_str.c_str(), base64_str.size());
  tvm::support::Base64InStream base64_stream(&stream);
  size_t decoded_size = Base64DecodedSize(base64_str);
  std::vector<unsigned char> decoded(decoded_size);
  base64_stream.InitPosition();
  base64_stream.Read((void*)decoded.data(), decoded_size);
  int width, height, num_channels;
  unsigned char* image_data =
      stbi_load_from_memory(decoded.data(), decoded_size, &width, &height, &num_channels, 3);
  if (!image_data) {
    return TResult::Error(stbi_failure_reason());
  }
  auto image_ndarray = NDArray::Empty({height, width, 3}, {kDLUInt, 8, 1}, {kDLCPU, 0});
  image_ndarray.CopyFromBytes((void*)image_data, width * height * 3);
  stbi_image_free(image_data);
  return TResult::Ok(image_ndarray);
}

NDArray ClipPreprocessor(NDArray image_data, int target_size, DLDevice device) {
  int height = image_data->shape[0];
  int width = image_data->shape[1];
  // Resize
  const int short_side = width < height ? width : height;
  const int long_side = width > height ? width : height;
  const int new_short_side = target_size;
  const int new_long_side = (int)(new_short_side * (long_side / (float)short_side));
  const int new_width = width < height ? new_short_side : new_long_side;
  const int new_height = width > height ? new_short_side : new_long_side;

  std::vector<float> processed_image_data(new_width * new_height * 3);

  // Bilinear Interpolation
  for (int y = 0; y < new_height; y++) {
    for (int x = 0; x < new_width; x++) {
      const float x_ratio = float(width - 1) / new_width;
      const float y_ratio = float(height - 1) / new_height;
      const int x1 = int(x_ratio * x);
      const int y1 = int(y_ratio * y);
      const int x2 = x1 + 1;
      const int y2 = y1 + 1;
      const float x_diff = x_ratio * x - x1;
      const float y_diff = y_ratio * y - y1;
      for (int c = 0; c < 3; c++) {
        const uint8_t top_left = ((uint8_t*)image_data->data)[(y1 * width + x1) * 3 + c];
        const uint8_t top_right = ((uint8_t*)image_data->data)[(y1 * width + x2) * 3 + c];
        const uint8_t bottom_left = ((uint8_t*)image_data->data)[(y2 * width + x1) * 3 + c];
        const uint8_t bottom_right = ((uint8_t*)image_data->data)[(y2 * width + x2) * 3 + c];
        processed_image_data[(y * new_width + x) * 3 + c] =
            (float)(int(top_left * (1 - x_diff) * (1 - y_diff) + top_right * x_diff * (1 - y_diff) +
                        bottom_left * y_diff * (1 - x_diff) + bottom_right * x_diff * y_diff));
      }
    }
  }

  // Center crop
  const int crop_x = (new_width - target_size) / 2;
  const int crop_y = (new_height - target_size) / 2;
  std::vector<float> cropped_image_data(target_size * target_size * 3);
  for (int y = 0; y < target_size; y++) {
    for (int x = 0; x < target_size; x++) {
      for (int c = 0; c < 3; c++) {
        cropped_image_data[(y * target_size + x) * 3 + c] =
            processed_image_data[((y + crop_y) * new_width + x + crop_x) * 3 + c];
      }
    }
  }

  // Rescale
  for (int i = 0; i < target_size * target_size * 3; i++) {
    cropped_image_data[i] = cropped_image_data[i] / 255.0f;
  }

  // Normalize
  const float IMAGE_MEAN[] = {0.48145466f, 0.4578275f, 0.40821073f};
  const float IMAGE_STD[] = {0.26862954f, 0.26130258f, 0.27577711f};
  for (int i = 0; i < target_size * target_size * 3; i++) {
    const int c = i % 3;
    cropped_image_data[i] = (cropped_image_data[i] - IMAGE_MEAN[c]) / IMAGE_STD[c];
  }

  std::vector<float> image_data_channel_first(target_size * target_size * 3);
  for (int y = 0; y < target_size; y++) {
    for (int x = 0; x < target_size; x++) {
      for (int c = 0; c < 3; c++) {
        image_data_channel_first[c * target_size * target_size + y * target_size + x] =
            cropped_image_data[(y * target_size + x) * 3 + c];
      }
    }
  }

  // Create NDArray
  auto image_ndarray = NDArray::Empty({1, 3, target_size, target_size}, {kDLFloat, 32, 1}, device);
  image_ndarray.CopyFromBytes((void*)image_data_channel_first.data(),
                              target_size * target_size * 3 * sizeof(float));

  return image_ndarray;
}

}  // namespace json_ffi
}  // namespace llm
}  // namespace mlc
