#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <string>
#include "serve/lora_manager.h"

namespace mlc::serve {

static void UploadLora(const std::string& adapter_npz) {
  // Alpha to be plumbed in later via manifest – use 1.0 for now.
  mlc::serve::LoraManager::Global()->UploadAdapter(adapter_npz, /*alpha=*/1.0f);
}

}  // namespace mlc::serve

// Expose a getter so Python (and other frontends) can retrieve the materialised
// delta tensor for a given full parameter name.  The returned NDArray may be
// undefined if the key is missing.
TVM_REGISTER_GLOBAL("mlc.get_lora_delta").set_body_typed([](const std::string& param_name) {
  return mlc::serve::LoraManager::Global()->Lookup(param_name);
});

// Called once by Python side to tell C++ what device the runtime operates on.
TVM_REGISTER_GLOBAL("mlc.set_active_device").set_body_typed([](int dev_type, int dev_id) {
  mlc::serve::LoraManager::Global()->SetDevice(dev_type, dev_id);
});

// Register with TVM's FFI so that python can call this symbol via
// `tvm.get_global_func("mlc.serve.UploadLora")`.
TVM_REGISTER_GLOBAL("mlc.serve.UploadLora")
    .set_body_typed([](const std::string& adapter_path) {
      mlc::serve::UploadLora(adapter_path);
    }); 