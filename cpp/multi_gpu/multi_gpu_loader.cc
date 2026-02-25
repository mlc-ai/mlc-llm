/*!
 * \file multi_gpu_loader.cc
 * \brief Implementation of a multi-GPU loader with loading-time sharding.
 */
#ifndef MLC_SINGLE_GPU_ONLY
#include <picojson.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/disco/builtin.h>
#include <tvm/runtime/disco/disco_worker.h>
#include <tvm/runtime/vm/tensor_cache_support.h>

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
namespace multi_gpu {

using tvm::Device;
using tvm::runtime::vm::TensorCacheMetadata;
using namespace tvm::runtime;
using tvm::ffi::Array;
using tvm::ffi::Function;
using tvm::ffi::Optional;
using tvm::ffi::TypedFunction;
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
  explicit PreprocessorPool(const ModelMetadata& model_metadata, Module vm_module) {
    for (const ModelMetadata::Param& param : model_metadata.params) {
      for (const ModelMetadata::Param::Preproc& preproc : param.preprocs) {
        const std::string& func_name = preproc.func_name;
        if (Function f = vm_module.defined()
                             ? vm_module->GetFunction(func_name, true).value_or(Function(nullptr))
                             : nullptr;
            f != nullptr) {
          preproc_funcs[func_name] = f;
        } else if (const auto f = Function::GetGlobal(func_name); f.has_value()) {
          preproc_funcs[func_name] = *f;
        } else {
          LOG(FATAL) << "ValueError: Undefined function: " << func_name;
        }
      }
    }
  }

  Tensor Apply(Tensor param, const ModelMetadata::Param& param_info) const {
    for (const ModelMetadata::Param::Preproc& preproc : param_info.preprocs) {
      const std::string& func_name = preproc.func_name;
      Tensor param_in = param;
      param = Tensor::Empty(preproc.out_shape, preproc.out_dtype, param->device);
      TVM_FFI_ICHECK(preproc_funcs.count(func_name));
      DLTensor dl_param_in = *param_in.operator->();
      DLTensor dl_param = *param.operator->();
      preproc_funcs.at(func_name)(&dl_param_in, &dl_param);
    }
    return param;
  }

 private:
  std::unordered_map<std::string, TypedFunction<void(DLTensor*, DLTensor*)>> preproc_funcs;
};

struct ParamInfo {
  const TensorCacheMetadata::FileRecord* file;
  const TensorCacheMetadata::FileRecord::ParamRecord* param;
};

Tensor RecvFromGlobalWorker0(Device device, const ModelMetadata::Param& param_info) {
  Shape shape = param_info.preprocs.empty() ? param_info.shape : param_info.preprocs[0].in_shape;
  Tensor result = Tensor::Empty(shape, param_info.dtype, device);
  RecvFromWorker0(result);
  return result;
}

Tensor BroadcastOrShardAndScatter(Tensor param, const ModelMetadata::Param& param_info,
                                  int num_shards, const PreprocessorPool& preprocs) {
  bool needs_sharding = !param_info.preprocs.empty();
  if (!needs_sharding) {
    BroadcastFromWorker0(param, /*in_group=*/true, param);
    return param;
  }
  Device device = param->device;
  Shape shape = param_info.preprocs.back().out_shape;
  DataType dtype = param_info.preprocs.back().out_dtype;
  TVM_FFI_ICHECK(shape.size() >= 1 && shape[0] == num_shards)
      << "ValueError: The first dimension of the output shape must be equal to the "
      << "number of shards, but got: " << shape << " and num_shards = " << num_shards;
  param = preprocs.Apply(param, param_info);
  Tensor result = Tensor::Empty(Shape(shape.begin() + 1, shape.end()), dtype, device);
  ScatterFromWorker0(param, /*in_group=*/true, result);
  return result;
}

Tensor ReceiveBroadcastedOrSharded(Device device, const ModelMetadata::Param& param_info,
                                   int num_shards) {
  bool needs_sharding = !param_info.preprocs.empty();
  Tensor result;
  if (needs_sharding) {
    Shape shape = param_info.preprocs.back().out_shape;
    DataType dtype = param_info.preprocs.back().out_dtype;
    result = Tensor::Empty(Shape(shape.begin() + 1, shape.end()), dtype, device);
    ScatterFromWorker0(std::nullopt, /*in_group=*/true, result);
  } else {
    result = Tensor::Empty(param_info.shape, param_info.dtype, device);
    BroadcastFromWorker0(result, /*in_group=*/true, result);
  }
  return result;
}

std::string FormatDuration(DurationType duration) {
  std::ostringstream os;
  auto float_seconds = std::chrono::duration_cast<std::chrono::duration<float>>(duration).count();
  os << std::fixed << std::setprecision(3) << float_seconds << " s";
  return os.str();
}

Array<Optional<Tensor>> LoadMultiGPU(const std::string& model_path, Module vm_module,
                                     const std::string& model_config_str) {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  Device device = worker->default_device;
  int worker_id = worker->worker_id;
  int group_size = worker->num_workers / worker->num_groups;
  int num_shards = group_size;
  int group_id = worker_id / group_size;
  LOG(INFO) << "[Worker #" << worker_id << "] Loading model to device: " << device;
  // Step 0. Initialize metadata and paths
  TensorCacheMetadata tensor_cache_metadata = TensorCacheMetadata::Load(model_path);
  picojson::value model_config;
  picojson::parse(model_config, model_config_str);
  ModelMetadata model_metadata =
      ModelMetadata::FromModule(vm_module, model_config.get<picojson::object>());
  CHECK_EQ(model_metadata.tensor_parallel_shards, num_shards)
      << "ValueError: The model is compiled using `--tensor-parallel-shards="
      << model_metadata.tensor_parallel_shards
      << "`, but mlc-chat-config.json is configured to use " << num_shards << " GPUs. "
      << "Please set \"tensor_parallel_shards\" in mlc-chat-config.json to "
      << model_metadata.tensor_parallel_shards;
  // Step 1. Extract auxiliary information
  PreprocessorPool preprocs(model_metadata, vm_module);
  std::unordered_map<std::string, ModelMetadata::Param> param_name2info;
  for (const ModelMetadata::Param& param : model_metadata.params) {
    param_name2info[param.name] = param;
  }
  // Step 2. Load, preprocess and shard all the parameters
  std::unordered_map<std::string, Tensor> sharded_params;
  if (worker_id == 0) {
    DurationType time_loading(0);
    DurationType time_preproc(0);
    ProgressBar progress_bar(model_metadata.params.size());
    LOG(INFO) << "Loading parameters...";
    for (const TensorCacheMetadata::FileRecord& record : tensor_cache_metadata.records) {
      Array<Tensor> loaded_params;
      {
        RangeTimer _(&time_loading);
        std::string raw_data_buffer;
        loaded_params = record.Load(device, model_path, &raw_data_buffer);
        DeviceAPI::Get(device)->StreamSync(device, nullptr);
      }
      // For each parameter in the shard file, preprocess and shard it
      for (size_t i = 0; i < record.records.size(); ++i, progress_bar.Progress()) {
        RangeTimer _(&time_preproc);
        const std::string& param_name = record.records[i].name;
        const ModelMetadata::Param& param_info = param_name2info.at(param_name);
        for (int group_id : param_info.pipeline_stages) {
          if (group_id == 0) {
            // Broadcast or shard-scatter this parameter to all workers in worker group 0.
            sharded_params[param_name] =
                BroadcastOrShardAndScatter(loaded_params[i], param_info, num_shards, preprocs);
          } else {
            // Send this parameter to the first worker of the worker group of "group_id",
            // and let that first worker to process this parameter.
            SendToWorker(loaded_params[i], /*receiver_id=*/group_id * group_size);
          }
        }
        DeviceAPI::Get(device)->StreamSync(device, nullptr);
      }
    }
    LOG(INFO) << "Loading done. Time used:" << std::fixed << std::setprecision(3)  //
              << " Loading " << FormatDuration(time_loading) << " Preprocessing "
              << FormatDuration(time_preproc) << ".";
  } else {
    for (const TensorCacheMetadata::FileRecord& record : tensor_cache_metadata.records) {
      for (size_t i = 0; i < record.records.size(); ++i) {
        const std::string& param_name = record.records[i].name;
        const ModelMetadata::Param& param_info = param_name2info.at(param_name);
        if (std::find(param_info.pipeline_stages.begin(), param_info.pipeline_stages.end(),
                      group_id) == param_info.pipeline_stages.end()) {
          // This worker group doesn't need to hold a copy of this parameter.
          continue;
        }

        if (worker_id % group_size == 0) {
          // The worker is the first worker of its worker group (while not the first worker group).
          // Receive the full parameter from the global worker 0.
          Tensor full_param = RecvFromGlobalWorker0(device, param_info);
          // Broadcast or shard-scatter this parameter to all workers in its worker group.
          sharded_params[param_name] =
              BroadcastOrShardAndScatter(full_param, param_info, num_shards, preprocs);
        } else {
          // The worker is not the first worker of its worker group.
          // Receive from the first worker in the its worker group.
          sharded_params[param_name] = ReceiveBroadcastedOrSharded(device, param_info, num_shards);
        }
      }
    }
  }

  // Step 3. Reorder the sharded parameters according to the order in model_metadata
  Array<Optional<Tensor>> shards;
  shards.reserve(model_metadata.params.size());
  for (const ModelMetadata::Param& param : model_metadata.params) {
    const auto& it = sharded_params.find(param.name);
    shards.push_back(it == sharded_params.end() ? Optional<Tensor>() : it->second);
  }
  return shards;
}

Array<Optional<Tensor>> LoadMultiGPUPresharded(const std::string& model_path, Module vm_module,
                                               const std::string& model_config_str) {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  Device device = worker->default_device;
  int worker_id = worker->worker_id;
  int group_size = worker->num_workers / worker->num_groups;
  int num_shards = group_size;
  int group_id = worker_id / group_size;
  int local_worker_id = worker_id % group_size;
  LOG(INFO) << "[Worker #" << worker_id << "] Loading model to device: " << device;
  // Step 0. Initialize metadata and paths
  TensorCacheMetadata tensor_cache_metadata = TensorCacheMetadata::Load(model_path);
  picojson::value model_config;
  picojson::parse(model_config, model_config_str);
  ModelMetadata model_metadata =
      ModelMetadata::FromModule(vm_module, model_config.get<picojson::object>());

  std::unordered_map<std::string, ParamInfo> param_info_map;
  for (const TensorCacheMetadata::FileRecord& file_record : tensor_cache_metadata.records) {
    for (const TensorCacheMetadata::FileRecord::ParamRecord& param_record : file_record.records) {
      const std::string& param_name = param_record.name;
      param_info_map[param_name] = ParamInfo{&file_record, &param_record};
    }
  }

  Array<Optional<Tensor>> params;
  const TensorCacheMetadata::FileRecord* current_file_;
  std::string current_file_stream_;
  params.reserve(model_metadata.params.size());
  DurationType time_loading(0);
  for (const ModelMetadata::Param& param : model_metadata.params) {
    RangeTimer _(&time_loading);
    if (std::find(param.pipeline_stages.begin(), param.pipeline_stages.end(), group_id) ==
        param.pipeline_stages.end()) {
      // This worker group doesn't need to hold a copy of this parameter.
      params.push_back(Optional<Tensor>());
      continue;
    }
    bool needs_sharding = !param.preprocs.empty();
    std::string param_name =
        needs_sharding ? static_cast<const std::stringstream&>(
                             std::stringstream() << param.name << "_shard-" << local_worker_id)
                             .str()
                       : std::string(param.name);
    auto it = param_info_map.find(param_name);
    CHECK(it != param_info_map.end()) << "ValueError: Cannot find parameter: " << param_name;
    const ParamInfo& param_info = (*it).second;
    const TensorCacheMetadata::FileRecord::ParamRecord* param_record = param_info.param;
    const TensorCacheMetadata::FileRecord* file_record = param_info.file;

    if (file_record != current_file_) {
      current_file_ = file_record;
      file_record->Load(device, model_path, &current_file_stream_);
    }

    params.push_back(param_record->Load(device, &current_file_stream_));
  }
  SyncWorker();
  if (worker_id == 0) {
    LOG(INFO) << "Loading done. Time used: " << FormatDuration(time_loading) << ".";
  }
  return params;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("mlc.multi_gpu.LoadMultiGPU", LoadMultiGPU)
      .def("mlc.multi_gpu.LoadMultiGPUPresharded", LoadMultiGPUPresharded);
}

}  // namespace multi_gpu
}  // namespace llm
}  // namespace mlc

#endif  // MLC_SINGLE_GPU_ONLY
