#include "lora_manager.h"

namespace mlc {
namespace llm {

using namespace tvm::runtime;

void LoraManager::LoadLora(const std::string& uid, int32_t index) {
    const static PackedFunc* fload_cache = tvm::runtime::Registry::Get("vm.builtin.ndarray_cache.load");
    ICHECK(fload_cache) << "TVM runtime cannot find vm.builtin.ndarray_cache.load";
    (*fload_cache)(model_path_, uid + "-ndarray-cache.json", static_cast<int32_t>(device_.device_type), device_.device_id);
    const char* name_loader = "vm.builtin.param_array_from_cache_by_name";
    const static PackedFunc* fload_params = tvm::runtime::Registry::Get(name_loader);
    ICHECK(fload_params) << "Cannot find env function: " << name_loader;
    for (size_t i = 0; i < model_metadata_->params.size(); ++i) {
        if (std::string(model_metadata_->params[i].name).find("lora_") != std::string::npos) {
            Array<tvm::runtime::NDArray> new_params = (*fload_params)(Array<String>({model_metadata_->params[i].name}));
            size_t bytes_offset = GetDataSize(*new_params[0].operator->()) * index;
            NDArray stacked_param = params_[i];
            NDArray view = stacked_param.CreateView(new_params[0].Shape(), new_params[0].DataType(), bytes_offset);
            view.CopyFrom(new_params[0]);
        }
    }

    // after we get params, it is safe to simply clear the cached version
    // as these params are referenced by params.
    const static PackedFunc* fclear_ndarray_cache =
        tvm::runtime::Registry::Get("vm.builtin.ndarray_cache.clear");
    ICHECK(fclear_ndarray_cache) << "Cannot find env function vm.builtin.ndarray_cache.clear";
    (*fclear_ndarray_cache)();
    // update cache
    active_uids_.insert(uid);
    buffer_id_[uid] = index;
}

std::vector<int64_t> LoraManager::GetLoraWeightIndices(Array<Optional<String>> lora_uids) {
    std::unordered_set<std::string> valid_lora_uids;
    for (const auto& lora_id : lora_uids) {
        if (lora_id.defined()) {
            valid_lora_uids.insert(lora_id.value());
        }
    }

    std::vector<int64_t> weight_indices(lora_uids.size(), -1);
    if (valid_lora_uids.empty()) { return weight_indices; }
    ICHECK(valid_lora_uids.size() <= model_metadata_->max_loras_per_batch) << "Lora uid size " << valid_lora_uids.size() << " is bigger than max_loras_per_batch " << model_metadata_->max_loras_per_batch;
    size_t i = 0;
    size_t j = active_uids_.size();
    int64_t index = j;
    std::vector<std::string> evictable_uids(active_uids_.begin(), active_uids_.end());
    for (size_t n = 0; n < lora_uids.size(); ++n) {
        const auto& uid = lora_uids[n];
        if (!uid.defined()) continue;
        if (active_uids_.count(uid.value())) {
            weight_indices[n] = buffer_id_[uid.value()];
            continue;
        }
        if (j < model_metadata_->max_loras_per_batch) {
            index = j;
        } else {
            while (i < evictable_uids.size() && valid_lora_uids.count(evictable_uids[i])) {
                i++;
            }
            ICHECK(i < evictable_uids.size());
            active_uids_.erase(evictable_uids[i]);
            buffer_id_.erase(evictable_uids[i]);
            index = i;
            i++;
        }
        LoadLora(uid.value(), index);
        weight_indices[n] = index;
    }

    return weight_indices;
}

Array<tvm::runtime::NDArray> LoraManager::LoadBaseParamsAndAllocateLoraBuffers() {
    const char* name_loader = "vm.builtin.param_array_from_cache_by_name";
    const PackedFunc* load_params = tvm::runtime::Registry::Get(name_loader);
    ICHECK(load_params) << "Cannot find env function: " << name_loader;
    params_.reserve(model_metadata_->params.size());
    for (const auto& param : model_metadata_->params) {
        if (std::string(param.name).find("lora_") == std::string::npos) {
            Array<NDArray> params = (*load_params)(Array<String>({param.name}));
            params_.push_back(params[0]);
        } else {
            params_.push_back(NDArray::Empty(param.shape, param.dtype, device_));
        }
    }
    return params_;
}

void LoraManager::UploadLora(const Array<String>& lora_ids, const Array<tvm::Integer>& indices) {
    ICHECK(lora_ids.size() == indices.size());
    for (size_t i = 0; i < lora_ids.size(); ++i) {
        LoadLora(lora_ids[i], indices[i].IntValue());
    }
}

} // namespace llm
} // namespace mlc