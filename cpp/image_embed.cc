/*!
 *  Copyright (c) 2023 by Contributors
 * \file image_embed.cc
 * \brief Implementation of image embedding module in support of multimodality in LLM.
 */
#include "image_embed.h"

#include <picojson.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <list>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_set>

namespace mlc {
namespace llm {

using tvm::Device;
using namespace tvm::runtime;

//------------------------------
// Image embedding module
//------------------------------
class LLMImageModule;

/*!
 * \brief Implements the image embedding module wrapper
 */
class LLMImage {
  friend class LLMImageModule;

 public:
  explicit LLMImage(DLDevice device) : device_(device) {}

  /*!
   * \brief Reload the image model from the specified model path.
   * \param executable The module to reload.
   * \param model_path The path to search for models.
   */
  void Reload(tvm::runtime::Module executable, String model_path) {
    // Step 1. Initialize vm, we use the packed function mechanism
    // so there is no explicit abi dependency on these extra
    // classes other than basic tvm runtime.
    auto fload_exec = executable->GetFunction("vm_load_executable");
    ICHECK(fload_exec.defined()) << "TVM runtime cannot find vm_load_executable";
    vm_ = fload_exec();
    vm_->GetFunction("vm_initialization")(static_cast<int>(device_.device_type), device_.device_id,
                                          static_cast<int>(memory::AllocatorType::kPooled),
                                          static_cast<int>(kDLCPU), 0,
                                          static_cast<int>(memory::AllocatorType::kPooled));

    embed_func_ = vm_->GetFunction("embed");

    // Step 2. Load params in nd-array cache.
    const PackedFunc* fload_cache = tvm::runtime::Registry::Get("vm.builtin.ndarray_cache.load");
    ICHECK(fload_cache) << "TVM runtime cannot find vm.builtin.ndarray_cache.load";
    (*fload_cache)(model_path, static_cast<int32_t>(device_.device_type), device_.device_id);

    const PackedFunc* fload_params =
        tvm::runtime::Registry::Get("vm.builtin.param_array_from_cache");
    ICHECK(fload_params) << "Cannot find env function vm.builtin.param_array_from_cache";
    params_ = (*fload_params)("param", -1);

    // after we get params, it is safe to simply clear the cached version
    // as these params are referenced by params_
    const PackedFunc* fclear_ndarray_cache =
        tvm::runtime::Registry::Get("vm.builtin.ndarray_cache.clear");
    ICHECK(fclear_ndarray_cache) << "Cannot find env function vm.builtin.ndarray_cache.clear";
    (*fclear_ndarray_cache)();

    this->Reset();
  }

  void Reset() { this->ResetRuntimeStats(); }

  /*! \brief reset the runtime stats. */
  void ResetRuntimeStats() { this->embed_total_time = 0; }

  /*!
   * \brief Given the input image, generate the embedding of the image.
   * \param image The input image in type DLTensor*.
   * \return The embedding of the input image.
   */
  NDArray EmbedStep(NDArray image) {
    CHECK(embed_func_.defined());
    auto tstart = std::chrono::high_resolution_clock::now();

    NDArray embedding = embed_func_(image, params_);

    auto tend = std::chrono::high_resolution_clock::now();
    this->embed_total_time += static_cast<double>((tend - tstart).count()) / 1e9;

    return embedding;
  }

  /*!
   * \return Text describing runtime stats.
   */
  std::string RuntimeStatsText() {
    std::ostringstream os;
    os << "image embed: " << std::setprecision(1) << std::fixed << this->embed_total_time << " s";
    return os.str();
  }

  //----------------------------
  // Statistics
  //----------------------------
  double embed_total_time = 0;
  //----------------------------
  // TVM related states
  //----------------------------
  // runtime device
  Device device_;
  // The vm module
  Module vm_;
  // embedding function
  PackedFunc embed_func_;
  // local params
  Array<NDArray> params_;
};

/*!
 * \brief An image module implementation that exposes
 *  the functions as tvm::runtime::Module.
 *
 * We do it so that the module is accessible to any image module in LLM
 * that tvm runtime can access.
 */
class LLMImageModule : public ModuleNode {
 public:
  // overrides
  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "reload") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        image_mod_ = nullptr;
        // we do not call ClearGlobalMemoryManager() here, please make sure to call reload image
        // model after reload LLM, since ClearGlobalMemoryManager() will be called there
        image_mod_ = std::make_unique<LLMImage>(LLMImage(device_));
        ICHECK_EQ(args.size(), 2);
        image_mod_->Reload(args[0], args[1]);
      });
    } else if (name == "unload") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        // we do not call ClearGlobalMemoryManager() here, please make sure to call unload image
        // model before unload LLM, since ClearGlobalMemoryManager() will be called there
        image_mod_ = nullptr;
      });
    } else if (name == "embed") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.size(), 1);
        *rv = GetImageModule()->EmbedStep(args[0]);
      });
    } else if (name == "reset") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.size(), 0);
        GetImageModule()->Reset();
      });
    } else if (name == "runtime_stats_text") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        *rv = GetImageModule()->RuntimeStatsText();
      });
    } else if (name == "reset_runtime_stats") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        GetImageModule()->ResetRuntimeStats();
      });
    } else {
      return PackedFunc(nullptr);
    }
  }

  void Init(DLDevice device) { device_ = device; }

  LLMImage* GetImageModule() {
    ICHECK(image_mod_ != nullptr) << "Image embedding module is not initialized via reload";
    return image_mod_.get();
  }

  const char* type_key() const final { return "mlc.image_embed"; }

 private:
  std::unique_ptr<LLMImage> image_mod_ = nullptr;
  DLDevice device_;
};

tvm::runtime::Module CreateImageModule(DLDevice device) {
  ObjectPtr<LLMImageModule> n = make_object<LLMImageModule>();
  n->Init(device);
  return Module(n);
}

// register as a system function that can be queried
TVM_REGISTER_GLOBAL("mlc.llm_image_module_create")
    .set_body_typed([](int device_type, int device_id) {
      return CreateImageModule(DLDevice{static_cast<DLDeviceType>(device_type), device_id});
    });

}  // namespace llm
}  // namespace mlc
