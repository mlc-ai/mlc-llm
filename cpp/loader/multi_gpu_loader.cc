/*!
 * \file multi_gpu_loader.cc
 * \brief Implementation of a multi-GPU loader with loading-time sharding.
 */
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

class ProgressBar {
 public:
  explicit ProgressBar(int width, int total) : width(width), total(total), cur(0) {}

  void Progress() {
    if (cur < total) {
      ++cur;
    }
    int bar_width = width - 2;  // Adjust for borders
    int completed = static_cast<int>(static_cast<float>(cur) / total * bar_width);
    int remaining = bar_width - completed;
    std::cout << "["                          //
              << std::string(completed, '=')  //
              << ">"                          //
              << std::string(remaining, ' ')  //
              << "] "                         //
              << " [" << cur << "/" << total << "]";
    if (cur < total) {
      std::cout << "\r";
      std::cout.flush();
    } else {
      std::cout << std::endl;  // Move to the next line after the progress bar is complete
    }
  }

 private:
  int width;
  int total;
  int cur;
};

Array<NDArray> LoadMultiGPU(const std::string& model_path, Module relax_vm_module) {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  Device device = worker->default_device;
  int worker_id = worker->worker_id;
  int num_shards = worker->num_workers;
  LOG(INFO) << "[Worker #" << worker_id << "] Loading model to device: " << device;
  // Step 0. Initialize metadata and paths
  NDArrayCacheMetadata ndarray_cache_metadata = NDArrayCacheMetadata::Load(model_path);
  ModelMetadata model_metadata = ModelMetadata::FromModule(relax_vm_module);
  // Step 1. Extract auxiliary information
  // - preproc_funcs: map from function name to function that does weight preprocessing
  // - param_name2info: map from each parameter name to parameter's metadata
  std::unordered_map<std::string, TypedPackedFunc<void(DLTensor*, DLTensor*)>> preproc_funcs;
  std::unordered_map<std::string, ModelMetadata::Param> param_name2info;
  for (const ModelMetadata::Param& param : model_metadata.params) {
    param_name2info[param.name] = param;
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
  // Step 2. Load, preprocess and shard all the parameters
  Map<String, NDArray> sharded_params;

  if (worker_id == 0) {
    DurationType time_loading(0);
    DurationType time_preproc(0);
    DurationType time_distribute(0);
    ProgressBar progress_bar(100, model_metadata.params.size());
    LOG(INFO) << "Loading parameters...";
    for (const NDArrayCacheMetadata::FileRecord& record : ndarray_cache_metadata.records) {
      int num_params = record.records.size();
      Array<NDArray> loaded_params;
      {
        RangeTimer _(&time_loading);
        std::string raw_data_buffer;
        loaded_params = record.Load(device, model_path, &raw_data_buffer);
        ICHECK_EQ(loaded_params.size(), num_params);
        TVMSynchronize(device.device_type, device.device_id, nullptr);
      }
      // For each parameter in the shard file, preprocess and shard it
      for (int i = 0; i < num_params; ++i) {
        const std::string& param_name = record.records[i].name;
        const ModelMetadata::Param& param_info = param_name2info.at(param_name);
        bool needs_sharding = !param_info.preprocs.empty();
        NDArray recv;
        if (needs_sharding) {
          ShapeTuple shape = param_info.preprocs.back().out_shape;
          DataType dtype = param_info.preprocs.back().out_dtype;
          ICHECK(shape.size() >= 1 && shape[0] == num_shards)
              << "ValueError: The first dimension of the "
              << "output shape must be equal to the "
              << "number of shards, but got: " << shape << " and num_shards = " << num_shards;
          NDArray param = loaded_params[i];
          {
            RangeTimer _(&time_preproc);
            for (const ModelMetadata::Param::Preproc& preproc : param_info.preprocs) {
              const std::string& func_name = preproc.func_name;
              NDArray param_in = param;
              param = NDArray::Empty(preproc.out_shape, preproc.out_dtype, device);
              ICHECK(preproc_funcs.count(func_name));
              preproc_funcs.at(func_name)(const_cast<DLTensor*>(param_in.operator->()),
                                          const_cast<DLTensor*>(param.operator->()));
            }
            TVMSynchronize(device.device_type, device.device_id, nullptr);
          }
          {
            RangeTimer _(&time_distribute);
            recv = NDArray::Empty(ShapeTuple(shape.begin() + 1, shape.end()), dtype, device);
            ScatterFromWorker0(param, recv);
            TVMSynchronize(device.device_type, device.device_id, nullptr);
          }
        } else {
          recv = loaded_params[i];
          {
            RangeTimer _(&time_distribute);
            BroadcastFromWorker0(recv, recv);
            TVMSynchronize(device.device_type, device.device_id, nullptr);
          }
        }
        sharded_params.Set(param_name, recv);
        progress_bar.Progress();
      }
    }
    auto f_convert = [](DurationType time) { return static_cast<double>(time.count()) / 1e6; };
    LOG(INFO) << "Loading done. Time used:" << std::fixed << std::setprecision(3)  //
              << " Loading " << f_convert(time_loading) << " s;"
              << " Preprocessing " << f_convert(time_preproc) << " s;"
              << " Distributing " << f_convert(time_distribute) << " s.";
  } else {
    for (const NDArrayCacheMetadata::FileRecord& record : ndarray_cache_metadata.records) {
      int num_params = record.records.size();
      // For each parameter in the shard file, preprocess and shard it
      for (int i = 0; i < num_params; ++i) {
        const std::string& param_name = record.records[i].name;
        const ModelMetadata::Param& param_info = param_name2info.at(param_name);
        bool needs_sharding = !param_info.preprocs.empty();
        NDArray recv;
        if (needs_sharding) {
          ShapeTuple shape = param_info.preprocs.back().out_shape;
          DataType dtype = param_info.preprocs.back().out_dtype;
          recv = NDArray::Empty(ShapeTuple(shape.begin() + 1, shape.end()), dtype, device);
          ScatterFromWorker0(tvm::NullOpt, recv);
        } else {
          recv = NDArray::Empty(param_info.shape, param_info.dtype, device);
          BroadcastFromWorker0(recv, recv);
        }
        sharded_params.Set(param_name, recv);
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

TVM_REGISTER_GLOBAL("mlc.loader.LoadMultiGPU").set_body_typed(LoadMultiGPU);

}  // namespace loader
}  // namespace llm
}  // namespace mlc
