#include "serve/lora_manager.h"

#include <mutex>
#include <fstream>
#include "3rdparty/cnpy/cnpy.h"

#include <regex>

namespace mlc::serve {

namespace {
// Mutex to guard singleton construction (call-once).
std::once_flag g_once;
LoraManager* g_inst{nullptr};
}

LoraManager* LoraManager::Global() {
  std::call_once(g_once, []() { g_inst = new LoraManager(); });
  return g_inst;
}

void LoraManager::UploadAdapter(const std::string& adapter_npz_path, float alpha) {
  // Load manifest JSON (same dir, same base + .json) to grab layer names if present.
  std::string manifest_path = adapter_npz_path + ".json";
  std::unordered_map<std::string, float> scaling_map;  // full_param_name -> scaling
  if (std::ifstream mf(manifest_path); mf.good()) {
    std::string text((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
    // Very small regex-based parser assuming {"key": 1.0, "k2": 0.5}
    std::regex kv_re("\"([^\"]+)\"\s*:\s*([0-9.+-eE]+)");
    auto begin = std::sregex_iterator(text.begin(), text.end(), kv_re);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
      std::string k = (*it)[1].str();
      float v = std::stof((*it)[2].str());
      scaling_map[k] = v;
    }
  }

  // Load every array in the .npz file via cnpy.
  std::map<std::string, cnpy::NpyArray> arrays = cnpy::npz_load(adapter_npz_path);
  tvm::Device cpu_dev{kDLCPU, 0};
  for (const auto& kv : arrays) {
    const std::string& name = kv.first;  // e.g., "decoder.layers.0.mlp.w1.delta"
    const cnpy::NpyArray& arr = kv.second;

    bool promote_to_fp32 = (arr.word_size == 2);
    DLDataType dtype;
    dtype.code = kDLFloat;
    dtype.lanes = 1;
    dtype.bits = promote_to_fp32 ? 32 : (arr.word_size == 4 ? 32 : 64);

    // Shape tuple
    std::vector<int64_t> shape_vec;
    for (auto d : arr.shape) shape_vec.push_back(static_cast<int64_t>(d));
    tvm::runtime::Shape shape(shape_vec);
    size_t numel = 1;
    for (auto d : arr.shape) numel *= d;

    tvm::Device target_dev = runtime_device_;
    tvm::runtime::NDArray nd;
    bool alloc_failed = false;
    try {
      nd = tvm::runtime::NDArray::Empty(shape, dtype, target_dev);
    } catch (const std::exception&) {
      alloc_failed = true;
    }
    if (alloc_failed) {
      target_dev = cpu_dev;
      nd = tvm::runtime::NDArray::Empty(shape, dtype, cpu_dev);
    }

    if (promote_to_fp32) {
      // Convert each half precision value to float32.
      const uint16_t* src = reinterpret_cast<const uint16_t*>(arr.data_holder->data());
      float* dst = static_cast<float*>(nd->data);
      for (size_t i = 0; i < numel; ++i) {
        uint16_t h = src[i];
        // IEEE 754 half to float conversion (reference implementation)
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t exp = (h & 0x7C00) >> 10;
        uint32_t mant = (h & 0x03FF);
        uint32_t f;
        if (exp == 0) {
          if (mant == 0) {
            f = sign;  // zero
          } else {
            // subnormal
            exp = 1;
            while ((mant & 0x0400) == 0) {
              mant <<= 1;
              exp -= 1;
            }
            mant &= 0x03FF;
            exp += 127 - 15;
            mant <<= 13;
            f = sign | (exp << 23) | mant;
          }
        } else if (exp == 0x1F) {
          // Inf or NaN
          f = sign | 0x7F800000 | (mant << 13);
        } else {
          // Normalised
          exp = exp + (127 - 15);
          f = sign | (exp << 23) | (mant << 13);
        }
        dst[i] = *reinterpret_cast<float*>(&f);
      }
    } else {
      nd.CopyFromBytes(arr.data_holder->data(), arr.data_holder->size());
    }

    // Apply alpha scaling if provided
    auto it_scale = scaling_map.find(name);
    if (it_scale != scaling_map.end()) {
      float scale = it_scale->second * alpha;
      if (dtype.bits == 32) {
        float* p = static_cast<float*>(nd->data);
        for (size_t i = 0; i < numel; ++i) p[i] *= scale;
      }
    }

    // If we allocated on CPU but runtime device is GPU, copy now.
    if (target_dev.device_type != runtime_device_.device_type || target_dev.device_id != runtime_device_.device_id) {
      nd = nd.CopyTo(runtime_device_);
    }

    delta_map_[name] = nd;

    // Keep the backing buffer alive for the lifetime of the manager.  This is
    // only necessary if we ever move to zero-copy NDArray creation, but is
    // safe to do now.
    owned_buffers_.push_back(arr.data_holder);
  }
}

tvm::runtime::NDArray LoraManager::Lookup(const std::string& param_name) const {
  auto it = delta_map_.find(param_name);
  if (it != delta_map_.end()) {
    return it->second;
  }
  return tvm::runtime::NDArray();  // undefined if not present.
}

}  // namespace mlc::serve 