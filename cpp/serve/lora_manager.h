#pragma once

#include <tvm/runtime/ndarray.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlc::serve {

// Lightweight singleton that maps parameter names to LoRA delta tensors that
// live on the *runtime device* (CPU or GPU).  The first iteration keeps the
// implementation minimal so CI can compile on CPU-only runners; actual .npz
// loading and GPU transfer will be filled in later.
class LoraManager {
 public:
  /*!\brief Return global singleton. */
  static LoraManager* Global();

  /*!\brief Upload a LoRA adapter given an on-disk artefact path.
   *
   *  For now we accept the path but load nothing; this keeps the build green
   *  while Python-level tests monkey-patch the upload path.  In a follow-up we
   *  will parse the associated manifest, mmap the .npz file and copy tensors
   *  to the active device.
   */
  void UploadAdapter(const std::string& adapter_npz_path, float alpha);

  /*!\brief Look up delta tensor for a parameter.  Returns an undefined NDArray
   *  if not present.
   */
  tvm::runtime::NDArray Lookup(const std::string& param_name) const;

  /*!\brief Record the runtime device (set once by Python engine). */
  void SetDevice(int device_type, int device_id) {
    runtime_device_ = {static_cast<DLDeviceType>(device_type), device_id};
  }

  tvm::Device runtime_device() const { return runtime_device_; }

 private:
  LoraManager() = default;
  std::unordered_map<std::string, tvm::runtime::NDArray> delta_map_;
  // Hold shared ownership of raw buffers backing the NDArrays to guarantee
  // they stay alive as long as the manager lives.
  std::vector<std::shared_ptr<std::vector<char>>> owned_buffers_;

  tvm::Device runtime_device_{kDLCPU, 0};
};

}  // namespace mlc::serve
