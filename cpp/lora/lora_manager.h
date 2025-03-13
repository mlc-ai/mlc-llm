/*!
 * \file lora_manager.h
 * \brief lora manager for loras
 */

#ifndef MLC_LLM_CPP_LORA_MANAGER_H__
#define MLC_LLM_CPP_LORA_MANAGER_H__

#include <picojson.h>
#include <tvm/runtime/container/shape_tuple.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/ir/expr.h>

#include <unordered_map>
#include "../metadata/model.h"

namespace mlc {
namespace llm {

using namespace tvm::runtime;

class LoraManager {
    public:
    LoraManager() {}
    LoraManager(const ModelMetadata* model_metadata, String model_path, DLDevice device)
    : model_metadata_(model_metadata), model_path_(model_path), device_(device) {}

    void LoadLora(const std::string& uid, int32_t index);

    std::vector<int64_t> GetLoraWeightIndices(Array<Optional<String>> lora_uids);

    Array<tvm::runtime::NDArray> LoadBaseParamsAndAllocateLoraBuffers();

    void UploadLora(const Array<String>& lora_ids, const Array<tvm::Integer>& indices);

    private:
    String model_path_;
    DLDevice device_;
    std::unordered_set<std::string> active_uids_;
    std::unordered_map<std::string, int64_t> buffer_id_;
    Array<tvm::runtime::NDArray> params_;
    const ModelMetadata* model_metadata_;
};

} // namespace llm
} // namespace mlc

#endif // MLC_LLM_CPP_LORA_MANAGER_H_