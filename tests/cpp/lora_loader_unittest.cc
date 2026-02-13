#include <gtest/gtest.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/device_api.h>

#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <vector>

#include "3rdparty/cnpy/cnpy.h"
#include "serve/lora_manager.h"

using namespace mlc::serve;

namespace {

// Helper: write a .npy header + data for a small FP32 array (C-order).
std::vector<char> BuildNpy(const std::vector<float>& data, const std::vector<size_t>& shape) {
  std::ostringstream oss(std::ios::binary);
  // Magic string + version 1.0
  const char magic[] = "\x93NUMPY";
  oss.write(magic, 6);
  uint8_t ver[2] = {1, 0};
  oss.write(reinterpret_cast<char*>(ver), 2);
  // Header dict
  std::ostringstream hdr;
  hdr << "{'descr': '<f4', 'fortran_order': False, 'shape': (";
  for (size_t i = 0; i < shape.size(); ++i) {
    hdr << shape[i];
    if (i + 1 != shape.size()) hdr << ", ";
  }
  if (shape.size() == 1) hdr << ",";  // numpy tuple syntax
  hdr << "), }";
  // Pad header to 64-byte alignment
  std::string hdr_str = hdr.str();
  size_t header_len = hdr_str.size() + 1;      // include newline
  size_t pad = 64 - ((10 + header_len) % 64);  // 10 = magic+ver+len
  hdr_str.append(pad, ' ');
  hdr_str.push_back('\n');
  uint16_t hlen16 = static_cast<uint16_t>(hdr_str.size());
  oss.write(reinterpret_cast<char*>(&hlen16), 2);
  oss.write(hdr_str.data(), hdr_str.size());
  // Write raw data
  oss.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
  std::string result = oss.str();
  return std::vector<char>(result.begin(), result.end());
}

// Write a minimal uncompressed .npz containing one member "delta.w".
void WriteMinimalNpz(const std::filesystem::path& path, const std::vector<char>& npy_bytes,
                     const std::string& member_name) {
  std::ofstream ofs(path, std::ios::binary);
  // Local file header (no compression)
  uint32_t sig = 0x04034b50;
  uint16_t version = 20;
  uint16_t flags = 0;
  uint16_t method = 0;  // stored
  uint16_t mtime = 0, mdate = 0;
  uint32_t crc32 = 0;  // not checked by loader
  uint32_t comp_size = static_cast<uint32_t>(npy_bytes.size());
  uint32_t uncomp_size = comp_size;
  uint16_t fname_len = static_cast<uint16_t>(member_name.size());
  uint16_t extra_len = 0;
  ofs.write(reinterpret_cast<char*>(&sig), 4);
  ofs.write(reinterpret_cast<char*>(&version), 2);
  ofs.write(reinterpret_cast<char*>(&flags), 2);
  ofs.write(reinterpret_cast<char*>(&method), 2);
  ofs.write(reinterpret_cast<char*>(&mtime), 2);
  ofs.write(reinterpret_cast<char*>(&mdate), 2);
  ofs.write(reinterpret_cast<char*>(&crc32), 4);
  ofs.write(reinterpret_cast<char*>(&comp_size), 4);
  ofs.write(reinterpret_cast<char*>(&uncomp_size), 4);
  ofs.write(reinterpret_cast<char*>(&fname_len), 2);
  ofs.write(reinterpret_cast<char*>(&extra_len), 2);
  ofs.write(member_name.data(), member_name.size());
  ofs.write(npy_bytes.data(), npy_bytes.size());
  // No central directory required for our reader.
}

TEST(LoraLoaderTest, LoadAndFetchDelta) {
  // Prepare temporary dir
  auto temp_dir = std::filesystem::temp_directory_path() / "mlc_lora_test";
  std::filesystem::create_directories(temp_dir);
  auto npz_path = temp_dir / "adapter.npz";

  // Data 2x2
  std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  std::vector<size_t> shape = {2, 2};
  auto npy_bytes = BuildNpy(data, shape);
  WriteMinimalNpz(npz_path, npy_bytes, "delta.w.npy");

  // Manifest scaling (alpha=2.0) – simple JSON
  std::ofstream(temp_dir / "adapter.npz.json") << "{\"delta.w.npy\": 2.0}";

  // Set runtime device to CPU using direct LoraManager call
  LoraManager::Global()->SetDevice(kDLCPU, 0);

  // Upload adapter
  LoraManager::Global()->UploadAdapter(npz_path.string(), /*alpha=*/1.0f);

  // Fetch directly through LoraManager
  tvm::runtime::NDArray arr = LoraManager::Global()->Lookup("delta.w.npy");
  ASSERT_TRUE(arr.defined());
  EXPECT_EQ(arr->dtype.bits, 32);
  EXPECT_EQ(arr->shape[0], 2);
  EXPECT_EQ(arr->shape[1], 2);
  EXPECT_EQ(arr->device.device_type, kDLCPU);
  // Check values (scaled by 2.0)
  float* ptr = static_cast<float*>(arr->data);
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_FLOAT_EQ(ptr[i], data[i] * 2.0f);
  }

  // Clean up
  std::filesystem::remove_all(temp_dir);
}

}  // namespace
