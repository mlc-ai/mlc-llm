/*!
 * \file multi_gpu_loader.cc
 * \brief Implementation of a multi-GPU loader with loading-time sharding.
 */
#include <picojson.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/disco/builtin.h>
#include <tvm/runtime/disco/disco_worker.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/relax_vm/ndarray_cache_support.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../metadata/model.h"
#include "../support/progress_bar.h"

namespace mlc {
namespace llm {
namespace loader {

using tvm::Device;
using tvm::runtime::relax_vm::NDArrayCacheMetadata;
using namespace tvm::runtime;
using DurationType = std::chrono::microseconds;

class RangeTimer {
 public:
  explicit RangeTimer(DurationType* result)
      : start(std::chrono::high_resolution_clock::now()), result(result) {}

  ~RangeTimer() {
    std::chrono::time_point<std::chrono::high_resolution_clock> end =
        std::chrono::high_resolution_clock::now();  //
    auto duration = end - start;
    (*result) += std::chrono::duration_cast<DurationType>(end - start);
  }

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  DurationType* result;
};

class PreprocessorPool {
 public:
  explicit PreprocessorPool(const ModelMetadata& model_metadata, Module relax_vm_module) {
    for (const ModelMetadata::Param& param : model_metadata.params) {
      for (const ModelMetadata::Param::Preproc& preproc : param.preprocs) {
        const std::string& func_name = preproc.func_name;
        if (PackedFunc f =
                relax_vm_module.defined() ? relax_vm_module->GetFunction(func_name, true) : nullptr;
            f != nullptr) {
          preproc_funcs[func_name] = f;
        } else if (const PackedFunc* f = tvm::runtime::Registry::Get(func_name)) {
          preproc_funcs[func_name] = *f;
        } else {
          LOG(FATAL) << "ValueError: Undefined function: " << func_name;
        }
      }
    }
  }

  NDArray Apply(NDArray param, const ModelMetadata::Param& param_info) const {
    for (const ModelMetadata::Param::Preproc& preproc : param_info.preprocs) {
      const std::string& func_name = preproc.func_name;
      NDArray param_in = param;
      param = NDArray::Empty(preproc.out_shape, preproc.out_dtype, param->device);
      ICHECK(preproc_funcs.count(func_name));
      preproc_funcs.at(func_name)(const_cast<DLTensor*>(param_in.operator->()),
                                  const_cast<DLTensor*>(param.operator->()));
    }
    return param;
  }

 private:
  std::unordered_map<std::string, TypedPackedFunc<void(DLTensor*, DLTensor*)>> preproc_funcs;
};

struct ParamInfo {
  const NDArrayCacheMetadata::FileRecord* file;
  const NDArrayCacheMetadata::FileRecord::ParamRecord* param;
};

NDArray BroadcastOrShardAndScatter(NDArray param, const ModelMetadata::Param& param_info,
                                   int num_shards, const PreprocessorPool& preprocs) {
  bool needs_sharding = !param_info.preprocs.empty();
  if (!needs_sharding) {
    BroadcastFromWorker0(param, param);
    return param;
  }
  Device device = param->device;
  ShapeTuple shape = param_info.preprocs.back().out_shape;
  DataType dtype = param_info.preprocs.back().out_dtype;
  ICHECK(shape.size() >= 1 && shape[0] == num_shards)
      << "ValueError: The first dimension of the "
      << "output shape must be equal to the "
      << "number of shards, but got: " << shape << " and num_shards = " << num_shards;
  param = preprocs.Apply(param, param_info);
  NDArray result = NDArray::Empty(ShapeTuple(shape.begin() + 1, shape.end()), dtype, device);
  ScatterFromWorker0(param, result);
  return result;
}

NDArray ReceiveBroadcastedOrSharded(Device device, const ModelMetadata::Param& param_info,
                                    int num_shards) {
  bool needs_sharding = !param_info.preprocs.empty();
  NDArray result;
  if (needs_sharding) {
    ShapeTuple shape = param_info.preprocs.back().out_shape;
    DataType dtype = param_info.preprocs.back().out_dtype;
    result = NDArray::Empty(ShapeTuple(shape.begin() + 1, shape.end()), dtype, device);
    ScatterFromWorker0(tvm::NullOpt, result);
  } else {
    result = NDArray::Empty(param_info.shape, param_info.dtype, device);
    BroadcastFromWorker0(result, result);
  }
  return result;
}

Array<NDArray> LoadMultiGPU(const std::string& model_path, Module relax_vm_module,
                            const std::string& model_config_str) {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  Device device = worker->default_device;
  int worker_id = worker->worker_id;
  int num_shards = worker->num_workers;
  LOG(INFO) << "[Worker #" << worker_id << "] Loading model to device: " << device;
  // Step 0. Initialize metadata and paths
  NDArrayCacheMetadata ndarray_cache_metadata = NDArrayCacheMetadata::Load(model_path);
  picojson::value model_config;
  picojson::parse(model_config, model_config_str);
  ModelMetadata model_metadata =
      ModelMetadata::FromModule(relax_vm_module, model_config.get<picojson::object>());
  CHECK_EQ(model_metadata.tensor_parallel_shards, num_shards)
      << "ValueError: The model is compiled using `--tensor-parallel-shards="
      << model_metadata.tensor_parallel_shards << "`, but ChatModule is configured to use "
      << num_shards << " GPUs. "
      << "Please use `ChatConfig(tensor_parallel_shards=" << model_metadata.tensor_parallel_shards
      << ", ...)` to initialize ChatModule.";
  // Step 1. Extract auxiliary information
  PreprocessorPool preprocs(model_metadata, relax_vm_module);
  std::unordered_map<std::string, ModelMetadata::Param> param_name2info;
  for (const ModelMetadata::Param& param : model_metadata.params) {
    param_name2info[param.name] = param;
  }
  // Step 2. Load, preprocess and shard all the parameters
  Map<String, NDArray> sharded_params;
  if (worker_id == 0) {
    DurationType time_loading(0);
    DurationType time_preproc(0);
    ProgressBar progress_bar(model_metadata.params.size());
    LOG(INFO) << "Loading parameters...";
    for (const NDArrayCacheMetadata::FileRecord& record : ndarray_cache_metadata.records) {
      Array<NDArray> loaded_params;
      {
        RangeTimer _(&time_loading);
        std::string raw_data_buffer;
        loaded_params = record.Load(device, model_path, &raw_data_buffer);
        TVMSynchronize(device.device_type, device.device_id, nullptr);
      }
      // For each parameter in the shard file, preprocess and shard it
      for (size_t i = 0; i < record.records.size(); ++i, progress_bar.Progress()) {
        RangeTimer _(&time_preproc);
        const std::string& param_name = record.records[i].name;
        const ModelMetadata::Param& param_info = param_name2info.at(param_name);
        sharded_params.Set(param_name, BroadcastOrShardAndScatter(loaded_params[i], param_info,
                                                                  num_shards, preprocs));
        TVMSynchronize(device.device_type, device.device_id, nullptr);
      }
    }
    auto f_convert = [](DurationType time) { return static_cast<double>(time.count()) / 1e6; };
    LOG(INFO) << "Loading done. Time used:" << std::fixed << std::setprecision(3)  //
              << " Loading " << f_convert(time_loading) << " s;"
              << " Preprocessing " << f_convert(time_preproc) << " s.";
  } else {
    for (const NDArrayCacheMetadata::FileRecord& record : ndarray_cache_metadata.records) {
      for (size_t i = 0; i < record.records.size(); ++i) {
        const std::string& param_name = record.records[i].name;
        const ModelMetadata::Param& param_info = param_name2info.at(param_name);
        sharded_params.Set(param_name, ReceiveBroadcastedOrSharded(device, param_info, num_shards));
      }
    }
  }

  // Step 3. Reorder the sharded parameters according to the order in model_metadata
  Array<NDArray> shards;
  shards.reserve(model_metadata.params.size());
  for (const ModelMetadata::Param& param : model_metadata.params) {
    std::string param_name = param.name;
    ICHECK(sharded_params.count(param_name))
        << "ValueError: Parameter " << param_name << " not found in loaded parameters.";
    shards.push_back(sharded_params.at(param_name));
  }
  return shards;
}

Array<NDArray> LoadMultiGPUPresharded(const std::string& model_path, Module relax_vm_module,
                                      const std::string& model_config_str) {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  Device device = worker->default_device;
  int worker_id = worker->worker_id;
  int num_shards = worker->num_workers;
  LOG(INFO) << "[Worker #" << worker_id << "] Loading model to device: " << device;
  // Step 0. Initialize metadata and paths
  NDArrayCacheMetadata ndarray_cache_metadata = NDArrayCacheMetadata::Load(model_path);
  picojson::value model_config;
  picojson::parse(model_config, model_config_str);
  ModelMetadata model_metadata =
      ModelMetadata::FromModule(relax_vm_module, model_config.get<picojson::object>());

  std::unordered_map<std::string, ParamInfo> param_info_map;
  for (const NDArrayCacheMetadata::FileRecord& file_record : ndarray_cache_metadata.records) {
    for (const NDArrayCacheMetadata::FileRecord::ParamRecord& param_record : file_record.records) {
      const std::string& param_name = param_record.name;
      param_info_map[param_name] = ParamInfo{&file_record, &param_record};
    }
  }

  Array<NDArray> params;
  const NDArrayCacheMetadata::FileRecord* current_file_;
  std::string current_file_stream_;
  params.reserve(model_metadata.params.size());
  for (const ModelMetadata::Param& param : model_metadata.params) {
    bool needs_sharding = !param.preprocs.empty();
    std::string param_name = needs_sharding
                                 ? static_cast<const std::stringstream&>(
                                       std::stringstream() << param.name << "_shard-" << worker_id)
                                       .str()
                                 : std::string(param.name);
    const ParamInfo& param_info = param_info_map.at(param_name);
    const NDArrayCacheMetadata::FileRecord::ParamRecord* param_record = param_info.param;
    const NDArrayCacheMetadata::FileRecord* file_record = param_info.file;

    if (file_record != current_file_) {
      current_file_ = file_record;
      file_record->Load(device, model_path, &current_file_stream_);
    }

    params.push_back(param_record->Load(device, &current_file_stream_));
  }
  return params;
}

TVM_REGISTER_GLOBAL("mlc.loader.LoadMultiGPU").set_body_typed(LoadMultiGPU);
TVM_REGISTER_GLOBAL("mlc.loader.LoadMultiGPUPresharded").set_body_typed(LoadMultiGPUPresharded);

}  // namespace loader
}  // namespace llm
}  // namespace mlc
