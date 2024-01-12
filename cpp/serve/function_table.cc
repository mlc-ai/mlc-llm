/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/function_table.cc
 * \brief The implementation of function table in serving for distributed inference.
 */

#include "function_table.h"

#include <tvm/runtime/disco/session.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <filesystem>
#include <string>
#include <vector>

#include "../support/load_bytes_from_file.h"

namespace mlc {
namespace llm {
namespace serve {

PackedFunc FunctionTable::SessionFuncAsPackedFunc(Session sess, DRef sess_func, String name) {
  return PackedFunc([sess, func = std::move(sess_func), name = std::move(name)](
                        TVMArgs args, TVMRetValue* rv) -> void {
    std::vector<TVMValue> tvm_values(args.num_args + 3);
    std::vector<int> tvm_type_codes(args.num_args + 3);
    TVMArgsSetter setter(tvm_values.data(), tvm_type_codes.data());
    setter(0, static_cast<int>(DiscoAction::kCallPacked));
    setter(1, 0);
    setter(2, func);
    for (int i = 0; i < args.num_args; ++i) {
      tvm_values[i + 3] = args.values[i];
      tvm_type_codes[i + 3] = args.type_codes[i];
    }
    *rv =
        sess->CallWithPacked(TVMArgs(tvm_values.data(), tvm_type_codes.data(), args.num_args + 3));
  });
}

void FunctionTable::Init(TVMArgValue reload_lib, Device device, int num_shards) {
  Device null_device{DLDeviceType(0), 0};
  if (num_shards > 1) {
    String lib_path{nullptr};
    try {
      lib_path = reload_lib.operator String();
    } catch (...) {
      LOG(FATAL)
          << "ValueError: In multi-GPU inference, we expect the first argument to Reload to be a "
             "string path to the model library (.so on Linux or .dll on Windows), but got: "
          << ArgTypeCode2Str(reload_lib.type_code());
    }
    constexpr const char* f_create_process_pool = "runtime.disco.create_process_pool";
    if (Registry::Get(f_create_process_pool) == nullptr) {
      LOG(FATAL) << "Cannot find process launcher `" << f_create_process_pool << "`. "
                 << "Multi-GPU inference depends on MLC LLM Python API to launch process.";
    }
    std::string ccl;
    if (device.device_type == kDLCUDA) {
      ccl = "nccl";
    } else if (device.device_type == kDLROCM) {
      ccl = "rccl";
    } else {
      LOG(FATAL) << "ValueError: Multi-GPU on device " << DLDeviceType2Str(device.device_type)
                 << " is not supported. Currently, only NCCL and RCCL are integrated.";
    }
    std::vector<int64_t> device_ids(num_shards);
    for (int i = 0; i < num_shards; ++i) {
      device_ids[i] = i;
    }
    this->use_disco = true;
    this->sess = Session::ProcessSession(num_shards, f_create_process_pool, "mlc_chat.cli.worker");
    this->sess->InitCCL(ccl, ShapeTuple(device_ids));
    this->disco_mod = sess->CallPacked(sess->GetGlobalFunc("runtime.disco.load_vm_module"),
                                       lib_path, null_device);
    this->mod_get_func = [this,
                          fmodule_get_function = sess->GetGlobalFunc("runtime.ModuleGetFunction")](
                             const std::string& name) -> PackedFunc {
      DRef func = sess->CallPacked(fmodule_get_function, this->disco_mod, name, false);
      bool exists = (func->DebugGetFromRemote(0).operator PackedFunc()) != nullptr;
      if (!exists) {
        return PackedFunc(nullptr);
      }
      return SessionFuncAsPackedFunc(sess, func, name);
    };
    this->get_global_func = [this](const std::string& name) -> PackedFunc {
      return SessionFuncAsPackedFunc(sess, sess->GetGlobalFunc(name), name);
    };
    this->_InitFunctions();
    {
      Module mod = this->disco_mod->DebugGetFromRemote(0);
      this->softmax_func_ = mod->GetFunction("softmax_with_temperature");
    }
  } else {
    Module executable{nullptr};
    if (reload_lib.type_code() == kTVMModuleHandle) {
      executable = reload_lib.operator Module();
    } else {
      String lib_path = reload_lib.operator String();
      executable = tvm::runtime::Module::LoadFromFile(lib_path);
    }
    this->use_disco = false;
    auto fload_exec = executable->GetFunction("vm_load_executable");
    ICHECK(fload_exec.defined()) << "TVM runtime cannot find vm_load_executable";
    this->local_vm = fload_exec();
    this->local_vm->GetFunction("vm_initialization")(
        static_cast<int>(device.device_type), device.device_id,
        static_cast<int>(tvm::runtime::memory::AllocatorType::kPooled), static_cast<int>(kDLCPU), 0,
        static_cast<int>(tvm::runtime::memory::AllocatorType::kPooled));
    this->mod_get_func = [this](const std::string& name) -> PackedFunc {
      return this->local_vm->GetFunction(name, false);
    };
    this->get_global_func = [](const std::string& name) -> PackedFunc {
      const auto* f = tvm::runtime::Registry::Get(name);
      CHECK(f != nullptr) << "ValueError: Cannot find function " << name;
      return *f;
    };
    this->_InitFunctions();
  }
}

ObjectRef FunctionTable::LoadParams(const std::string& model_path, Device device) {
  if (this->use_disco) {
    std::filesystem::path fs_model_path = model_path;
    std::string metadata_path = (fs_model_path / "ndarray-cache.json").string();
    std::string ndarray_cache_metadata = LoadBytesFromFile(metadata_path);
    PackedFunc loader_create = this->get_global_func("runtime.disco.ShardLoader");
    PackedFunc loader_load_all = this->get_global_func("runtime.disco.ShardLoaderLoadAll");
    CHECK(loader_create != nullptr);
    CHECK(loader_load_all != nullptr);
    DRef loader = loader_create(metadata_path, ndarray_cache_metadata, "", this->disco_mod);
    DRef params = loader_load_all(loader);
    return params;
  } else {
    const PackedFunc* fload_cache = tvm::runtime::Registry::Get("vm.builtin.ndarray_cache.load");
    ICHECK(fload_cache) << "TVM runtime cannot find vm.builtin.ndarray_cache.load";
    (*fload_cache)(model_path, static_cast<int32_t>(device.device_type), device.device_id);
    const PackedFunc* fload_params =
        tvm::runtime::Registry::Get("vm.builtin.param_array_from_cache");
    ICHECK(fload_params) << "Cannot find env function vm.builtin.param_array_from_cache";
    Array<NDArray> params = (*fload_params)("param", -1);
    // after we get params, it is safe to simply clear the cached version
    // as these params are referenced by params_
    const PackedFunc* fclear_ndarray_cache =
        tvm::runtime::Registry::Get("vm.builtin.ndarray_cache.clear");
    ICHECK(fclear_ndarray_cache) << "Cannot find env function vm.builtin.ndarray_cache.clear";
    (*fclear_ndarray_cache)();
    return params;
  }
}

void FunctionTable::_InitFunctions() {
  this->embed_func_ = mod_get_func("embed");
  this->prefill_func_ = mod_get_func("prefill_with_embed");
  this->decode_func_ = mod_get_func("decode_with_embed");
  this->softmax_func_ = mod_get_func("softmax_with_temperature");
  this->create_kv_cache_func_ = mod_get_func("create_kv_cache");
  this->reset_kv_cache_func_ = get_global_func("vm.builtin.paged_attention_kv_cache_clear");
  this->kv_cache_add_sequence_func_ =
      get_global_func("vm.builtin.paged_attention_kv_cache_add_sequence");
  this->kv_cache_remove_sequence_func_ =
      get_global_func("vm.builtin.paged_attention_kv_cache_remove_sequence");
  this->kv_cache_begin_forward_func_ =
      get_global_func("vm.builtin.paged_attention_kv_cache_begin_forward");
  this->kv_cache_end_forward_func_ =
      get_global_func("vm.builtin.paged_attention_kv_cache_end_forward");
  this->kv_cache_attention_func_ = get_global_func("vm.builtin.paged_attention_kv_cache_attention");
  this->kv_cache_popn_func_ = get_global_func("vm.builtin.paged_attention_kv_cache_popn");
  this->kv_cache_get_num_available_pages_func_ =
      get_global_func("vm.builtin.paged_attention_kv_cache_get_num_available_pages");
  support_backtracking_kv_ = true;
}

ObjectRef FunctionTable::Empty(ShapeTuple shape, DataType dtype, Device device) const {
  Device null_device{DLDeviceType(0), 0};
  if (this->use_disco) {
    DRef empty_func = sess->GetGlobalFunc("runtime.disco.empty");
    return sess->CallPacked(empty_func, shape, dtype, null_device);
  } else {
    return NDArray::Empty(shape, dtype, device);
  }
}

ObjectRef FunctionTable::CopyToWorker0(const NDArray& host_array) {
  Device null_device{DLDeviceType(0), 0};
  if (this->use_disco) {
    DRef array =
        Downcast<DRef>(this->Empty(host_array.Shape(), host_array.DataType(), null_device));
    sess->CopyToWorker0(host_array, array);
    return array;
  } else {
    return host_array;
  }
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
