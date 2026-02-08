#include <tvm/ffi/function.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/tensor.h>

#include <iostream>
#include <string>

#include "lora_manager.h"

namespace mlc::serve {

using namespace tvm;
using namespace tvm::runtime;

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("mlc.get_lora_delta",
           [](const String& param_name) -> Tensor {
             std::cout << "REAL TVM FFI: get_lora_delta called for: " << param_name << std::endl;

             // Get the actual LoRA delta from the manager.
             auto delta_tensor = LoraManager::Global()->Lookup(param_name);

             if (delta_tensor.defined()) {
               std::cout << "REAL TVM FFI: Found delta tensor with shape: [";
               for (int i = 0; i < delta_tensor->ndim; ++i) {
                 std::cout << delta_tensor->shape[i];
                 if (i < delta_tensor->ndim - 1) std::cout << ", ";
               }
               std::cout << "]" << std::endl;
               return delta_tensor;
             }

             std::cout << "REAL TVM FFI: No delta found, creating zero tensor" << std::endl;
             Device device{kDLCPU, 0};
             auto zero_tensor = Tensor::Empty({1, 1}, DataType::Float(32), device);
             float* data = static_cast<float*>(zero_tensor->data);
             data[0] = 0.0f;
             return zero_tensor;
           })
      .def("mlc.set_active_device",
           [](int dev_type, int dev_id) {
             std::cout << "REAL TVM FFI: set_active_device called: " << dev_type << ", " << dev_id
                       << std::endl;
             LoraManager::Global()->SetDevice(dev_type, dev_id);
           })
      .def("mlc.serve.UploadLora", [](const String& adapter_path) {
        std::cout << "REAL TVM FFI: UploadLora called with: " << adapter_path << std::endl;
        LoraManager::Global()->UploadAdapter(adapter_path, 1.0f);
      });
}

// Keep the namespace functions for direct C++ access
void UploadLora(const std::string& adapter_path) {
  LoraManager::Global()->UploadAdapter(adapter_path, 1.0f);
}

std::string GetLoraDelta(const std::string& param_name) {
  auto result = LoraManager::Global()->Lookup(param_name);
  return result.defined() ? "tensor_found" : "tensor_not_found";
}

void SetActiveDevice(int dev_type, int dev_id) {
  LoraManager::Global()->SetDevice(dev_type, dev_id);
}

}  // namespace mlc::serve
