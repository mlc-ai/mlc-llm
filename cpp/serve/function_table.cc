/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/function_table.cc
 * \brief The implementation of function table in serving for distributed inference.
 */

#include "function_table.h"

#include <tvm/ffi/function.h>
#include <tvm/runtime/disco/session.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/tensor.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "../support/load_bytes_from_file.h"
#include "../support/utils.h"
#include "sampler/sampler.h"

namespace mlc {
namespace llm {
namespace serve {

Optional<IntTuple> GetDiscoWorkerCPUBinding(int num_workers) {
  const char* raw_cpu_binding = std::getenv("MLC_DISCO_WORKER_CPU_BINDING");
  if (raw_cpu_binding == nullptr) {
    return std::nullopt;
  }

  std::string cpu_binding_str(raw_cpu_binding);
  std::vector<std::string> cpu_ids_str = Split(cpu_binding_str, ',');
  std::vector<int64_t> cpu_ids;
  for (const std::string& cpu_id_str : cpu_ids_str) {
    try {
      cpu_ids.push_back(std::stol(cpu_id_str));
    } catch (std::invalid_argument const& ex) {
      LOG(FATAL) << "Invalid MLC_DISCO_WORKER_CPU_BINDING \"" << cpu_binding_str << "\"";
    }
  }
  if (static_cast<int>(cpu_ids.size()) < num_workers) {
    LOG(FATAL) << "Insufficient number of specified CPU workers in MLC_DISCO_WORKER_CPU_BINDING, "
                  "expecting at least "
               << num_workers << "CPU ids but only " << cpu_ids.size() << " are given.";
  }

  return IntTuple{cpu_ids};
}

Function FunctionTable::SessionFuncAsPackedFunc(Session sess, DRef sess_func, String name) {
  return Function([sess, func = std::move(sess_func), name = std::move(name)](
                      ffi::PackedArgs args, ffi::Any* rv) -> void {
    std::vector<AnyView> packed_args(args.size() + 3);
    packed_args[0] = static_cast<int>(DiscoAction::kCallPacked);
    packed_args[1] = 0;
    packed_args[2] = func;
    for (int i = 0; i < args.size(); ++i) {
      packed_args[i + 3] = args[i];
    }
    *rv = sess->CallWithPacked(tvm::ffi::PackedArgs(packed_args.data(), packed_args.size()));
  });
}

void FunctionTable::Init(String reload_lib_path, Device device, picojson::object model_config,
                         Optional<Session> session, int num_shards, int num_stages) {
  local_gpu_device = device;
  this->model_config = model_config;
  this->cached_buffers = Map<String, ObjectRef>();

  int num_workers = num_shards * num_stages;
  if (num_workers > 1) {
    TVM_FFI_ICHECK(session.defined());
    this->sess = session.value();
    this->use_disco = true;
    this->disco_mod = sess->CallPacked(sess->GetGlobalFunc("runtime.disco.load_vm_module"),
                                       reload_lib_path, Optional<Device>(std::nullopt));
    this->mod_get_func = [this, fmodule_get_function = sess->GetGlobalFunc(
                                    "ffi.ModuleGetFunction")](const std::string& name) -> Function {
      DRef func = sess->CallPacked(fmodule_get_function, this->disco_mod, name, true);
      bool exists = (func->DebugGetFromRemote(0).as<Function>()) != nullptr;
      if (!exists) {
        return Function(nullptr);
      }
      return SessionFuncAsPackedFunc(sess, func, name);
    };
    if (num_stages == 1) {
      if (Optional<IntTuple> cpu_ids = GetDiscoWorkerCPUBinding(/*num_workers=*/num_shards)) {
        IntTuple cpu_ids_value = cpu_ids.value();
        sess->CallPacked(sess->GetGlobalFunc("runtime.disco.bind_worker_to_cpu_core"),
                         cpu_ids_value);
      }
    }
    this->get_global_func = [this](const std::string& name) -> Function {
      return SessionFuncAsPackedFunc(sess, sess->GetGlobalFunc(name), name);
    };
    this->model_metadata_ = ModelMetadata::FromModule(
        this->disco_mod.value()->DebugGetFromRemote(0).cast<Module>(), std::move(model_config));
    this->_InitFunctions();
  } else {
    TVM_FFI_ICHECK(!session.defined());
    Optional<Module> executable = std::nullopt;
    Optional<Function> fload_exec;
    if (StartsWith(reload_lib_path, "system://")) {
      static Function f_load_system_lib = Function::GetGlobalRequired("ffi.SystemLib");
      std::string system_lib_prefix = std::string(reload_lib_path).substr(9);
      std::replace(system_lib_prefix.begin(), system_lib_prefix.end(), /*old=*/'-', /*new=*/'_');
      executable = f_load_system_lib(system_lib_prefix + "_").cast<Module>();
      fload_exec = executable.value()->GetFunction("vm_load_executable");
      TVM_FFI_ICHECK(fload_exec.defined())
          << "Cannot find system lib with " << system_lib_prefix
          << ", please make sure you set model_lib field consistently with the compilation ";
    } else {
      executable = tvm::ffi::Module::LoadFromFile(reload_lib_path);
      fload_exec = executable.value()->GetFunction("vm_load_executable");
      /* precompile opencl kernel programs */
      if (device.device_type == kDLOpenCL) {
        auto f_get = executable.value()->GetFunction("opencl.GetPreCompiledPrograms", true);
        CHECK(f_get.defined()) << "Cannot find opencl.GetPreCompiledPrograms";
        tvm::ffi::String bytes = f_get.value()().cast<String>();
        auto f_set = executable.value()->GetFunction("opencl.SetPreCompiledPrograms", true);
        CHECK(f_set.defined()) << "Cannot find opencl.SetPreCompiledPrograms";
        f_set.value()(tvm::ffi::String(bytes));
      }
      TVM_FFI_ICHECK(fload_exec.defined()) << "TVM runtime cannot find vm_load_executable";
    }
    this->use_disco = false;
    this->local_vm = fload_exec.value()().cast<Module>();
    this->local_vm.value()
        ->GetFunction("vm_initialization")
        .value()(static_cast<int>(device.device_type), device.device_id,
                 static_cast<int>(tvm::runtime::memory::AllocatorType::kPooled),
                 static_cast<int>(kDLCPU), 0,
                 static_cast<int>(tvm::runtime::memory::AllocatorType::kPooled));
    this->mod_get_func = [this](const std::string& name) -> Function {
      return this->local_vm.value()->GetFunction(name, true).value_or(Function(nullptr));
    };
    this->get_global_func = [](const std::string& name) -> Function {
      return Function::GetGlobalRequired(name);
    };
    this->model_metadata_ =
        ModelMetadata::FromModule(this->local_vm.value(), std::move(model_config));
    this->_InitFunctions();
  }
  TVM_FFI_ICHECK_EQ(this->model_metadata_.tensor_parallel_shards, num_shards);
  TVM_FFI_ICHECK_EQ(this->model_metadata_.pipeline_parallel_stages, num_stages);
  // Invoke the CUDA graph allocation init function if it is defined.
  if (cuda_graph_alloc_init_func_.defined()) {
    this->cuda_graph_alloc_init_func_();
  }
}

ObjectRef FunctionTable::LoadParams(const std::string& model_path, Device device) {
  if (this->use_disco) {
    Optional<DRef> params = std::nullopt;
    if (this->model_metadata_.params.empty()) {
      std::filesystem::path fs_model_path = model_path;
      std::string metadata_path = (fs_model_path / "tensor-cache.json").string();
      std::string tensor_cache_metadata = LoadBytesFromFile(metadata_path);
      Function loader_create = this->get_global_func("runtime.disco.ShardLoader");

      auto load_all_func_name = "runtime.disco.ShardLoaderLoadAll";
      Function loader_load_all = this->get_global_func(load_all_func_name);
      CHECK(loader_create != nullptr);
      CHECK(loader_load_all != nullptr);
      DRef loader =
          loader_create(metadata_path, tensor_cache_metadata, "", this->disco_mod).cast<DRef>();
      params = loader_load_all(loader).cast<DRef>();
    } else {
      auto load_func_name = getenv("MLC_INTERNAL_PRESHARD_NUM") == nullptr
                                ? "mlc.multi_gpu.LoadMultiGPU"
                                : "mlc.multi_gpu.LoadMultiGPUPresharded";
      Function loader = this->get_global_func(load_func_name);
      params = loader(model_path, this->disco_mod, picojson::value(this->model_config).serialize())
                   .cast<DRef>();
    }
    return params.value();
  } else {
    static Function fload_cache = Function::GetGlobalRequired("vm.builtin.tensor_cache.load");
    fload_cache(model_path, static_cast<int32_t>(device.device_type), device.device_id);
    Array<Tensor> params;
    if (this->model_metadata_.params.empty()) {
      constexpr const char* name_loader = "vm.builtin.param_array_from_cache";
      static Function fload_params = Function::GetGlobalRequired(name_loader);
      params = fload_params("param", -1).cast<Array<Tensor>>();
    } else {
      constexpr const char* name_loader = "vm.builtin.param_array_from_cache_by_name";
      static Function fload_params = Function::GetGlobalRequired(name_loader);
      Array<String> param_names;
      param_names.reserve(this->model_metadata_.params.size());
      for (const auto& param : this->model_metadata_.params) {
        param_names.push_back(param.name);
      }
      params = fload_params(param_names).cast<Array<Tensor>>();
    }
    // after we get params, it is safe to simply clear the cached version
    // as these params are referenced by params_
    static Function fclear_tensor_cache =
        Function::GetGlobalRequired("vm.builtin.tensor_cache.clear");
    fclear_tensor_cache();
    return params;
  }
}

void FunctionTable::_InitFunctions() {
  this->embed_func_ = mod_get_func("embed");
  this->image_embed_func_ = mod_get_func("image_embed");
  this->single_batch_prefill_func_ = mod_get_func("prefill");
  this->single_batch_decode_func_ = mod_get_func("decode");
  this->single_batch_extend_func_ = mod_get_func("extend");
  this->prefill_func_ = mod_get_func("batch_prefill");
  this->decode_func_ = mod_get_func("batch_decode");
  this->extend_func_ = mod_get_func("batch_extend");
  this->verify_func_ = mod_get_func("batch_verify");
  this->single_batch_prefill_to_last_hidden_func_ = mod_get_func("prefill_to_last_hidden_states");
  this->single_batch_decode_to_last_hidden_func_ = mod_get_func("decode_to_last_hidden_states");
  this->prefill_to_last_hidden_func_ = mod_get_func("batch_prefill_to_last_hidden_states");
  this->decode_to_last_hidden_func_ = mod_get_func("batch_decode_to_last_hidden_states");
  this->verify_to_last_hidden_func_ = mod_get_func("batch_verify_to_last_hidden_states");
  this->fuse_embed_hidden_func_ = mod_get_func("fuse_embed_hidden_states");
  Module mod = this->use_disco ? this->disco_mod.value()->DebugGetFromRemote(0).cast<Module>()
                               : this->local_vm.value();
  this->get_logits_func_ = mod_get_func("get_logits");
  this->batch_get_logits_func_ = mod_get_func("batch_get_logits");
  this->batch_select_last_hidden_func_ = mod_get_func("batch_select_last_hidden_states");
  this->softmax_func_ =
      mod->GetFunction("softmax_with_temperature", true).value_or(Function(nullptr));
  this->apply_logit_bias_func_ =
      mod->GetFunction("apply_logit_bias_inplace", true).value_or(Function(nullptr));
  this->apply_penalty_func_ =
      mod->GetFunction("apply_penalty_inplace", true).value_or(Function(nullptr));
  this->apply_bitmask_func_ =
      mod->GetFunction("apply_bitmask_inplace", true).value_or(Function(nullptr));
  this->alloc_embedding_tensor_func_ = mod_get_func("alloc_embedding_tensor");
  this->cuda_graph_alloc_init_func_ = mod_get_func("cuda_graph_alloc_init");
  this->create_kv_cache_func_ = mod_get_func("create_flashinfer_paged_kv_cache");
  if (this->model_metadata_.sliding_window_size != -1 || !this->create_kv_cache_func_.defined()) {
    Function f_create_rnn_state = mod_get_func("create_rnn_state");
    if (f_create_rnn_state.defined()) {
      this->create_kv_cache_func_ = f_create_rnn_state;
    } else {
      this->create_kv_cache_func_ = mod_get_func("create_tir_paged_kv_cache");
    }
  }
  this->reset_kv_cache_func_ = get_global_func("vm.builtin.kv_state_clear");
  this->kv_cache_add_sequence_func_ = get_global_func("vm.builtin.kv_state_add_sequence");
  this->kv_cache_fork_sequence_func_ = get_global_func("vm.builtin.kv_state_fork_sequence");
  this->kv_cache_enable_sliding_window_for_seq_ =
      get_global_func("vm.builtin.attention_kv_cache_enable_sliding_window_for_seq");
  this->kv_cache_remove_sequence_func_ = get_global_func("vm.builtin.kv_state_remove_sequence");
  this->kv_cache_begin_forward_func_ = get_global_func("vm.builtin.kv_state_begin_forward");
  this->kv_cache_end_forward_func_ = get_global_func("vm.builtin.kv_state_end_forward");
  this->kv_cache_disagg_prepare_recv_func_ =
      get_global_func("vm.builtin.kv_cache_disagg_prepare_recv");
  this->kv_cache_disagg_mark_send_func_ = get_global_func("vm.builtin.kv_cache_disagg_mark_send");
  this->kv_cache_popn_func_ = get_global_func("vm.builtin.kv_state_popn");
  this->kv_cache_commit_accepted_token_tree_nodes_func_ =
      get_global_func("vm.builtin.attention_kv_cache_commit_accepted_token_tree_nodes");
  this->kv_cache_get_num_available_pages_func_ =
      Function::GetGlobalRequired("vm.builtin.attention_kv_cache_get_num_available_pages");
  this->kv_cache_get_total_sequence_length_func_ =
      Function::GetGlobalRequired("vm.builtin.attention_kv_cache_get_total_sequence_length");
  if (Sampler::SupportGPUSampler(local_gpu_device)) {
    gpu_multinomial_from_uniform_func_ =
        mod->GetFunction("multinomial_from_uniform", true).value_or(Function(nullptr));
    gpu_argsort_probs_func_ = mod->GetFunction("argsort_probs", true).value_or(Function(nullptr));
    gpu_sample_with_top_p_func_ =
        mod->GetFunction("sample_with_top_p", true).value_or(Function(nullptr));
    gpu_sampler_take_probs_func_ =
        mod->GetFunction("sampler_take_probs", true).value_or(Function(nullptr));
    gpu_verify_draft_tokens_func_ =
        mod->GetFunction("sampler_verify_draft_tokens", true).value_or(Function(nullptr));
    gpu_renormalize_by_top_p_func_ =
        mod->GetFunction("renormalize_by_top_p", true).value_or(Function(nullptr));
  }
  this->nd_view_func_ = get_global_func("vm.builtin.reshape");
  this->nd_get_shape_func_ = get_global_func("vm.builtin.shape_of");
  this->nd_copy_embedding_to_offset_func_ = get_global_func("mlc.copy_embedding_to_offset");
  support_backtracking_kv_ = true;
  this->tuple_getitem_func_ = get_global_func("vm.builtin.tuple_getitem");
  if (use_disco) {
    this->last_group_send_to_worker_0_ =
        get_global_func("mlc.multi_gpu.SendFromLastGroupToWorker0");
  }

  this->gather_probs_func_ = mod->GetFunction("gather_probs", true).value_or(Function(nullptr));
  this->scatter_probs_func_ = mod->GetFunction("scatter_probs", true).value_or(Function(nullptr));
  this->gather_hidden_states_func_ = mod_get_func("gather_hidden_states");
  this->scatter_hidden_states_func_ = mod_get_func("scatter_hidden_states");
}

ObjectRef FunctionTable::Empty(Shape shape, DataType dtype, Device device,
                               bool worker0_only) const {
  if (this->use_disco) {
    DRef empty_func = sess->GetGlobalFunc("runtime.disco.empty");
    return sess->CallPacked(empty_func, shape, dtype, Optional<Device>(std::nullopt), worker0_only,
                            /*in_group=*/false);
  } else {
    return Tensor::Empty(shape, dtype, device);
  }
}

ObjectRef FunctionTable::CopyToWorker0(const Tensor& host_array, String buffer_cache_key,
                                       Shape max_reserved_shape, bool local_only) {
  Map<String, ObjectRef> cached_buffers = this->cached_buffers.value();
  if (this->use_disco && !local_only) {
    Device null_device{DLDeviceType(0), 0};
    Optional<DRef> buffer = std::nullopt;
    auto it = cached_buffers.find(buffer_cache_key);
    if (it != cached_buffers.end()) {
      buffer = Downcast<DRef>((*it).second);
    } else {
      buffer = Downcast<DRef>(this->Empty(max_reserved_shape, host_array.DataType(), null_device,
                                          /*worker0_only=*/false));
      cached_buffers.Set(buffer_cache_key, buffer.value());
    }
    Shape real_shape = host_array.Shape();
    DRef buffer_view = nd_view_func_(buffer.value(), real_shape).cast<DRef>();
    sess->CopyToWorker0(host_array, buffer_view);
    return buffer_view;
  } else {
    auto it = cached_buffers.find(buffer_cache_key);
    Tensor buffer{nullptr};
    if (it != cached_buffers.end()) {
      buffer = Downcast<Tensor>((*it).second);
      if (buffer_cache_key == "image") {
        if (runtime::GetDataSize(*buffer.operator->()) <
            runtime::GetDataSize(*host_array.operator->())) {
          buffer = Tensor::Empty(max_reserved_shape, host_array->dtype, local_gpu_device);
          cached_buffers.Set(buffer_cache_key, buffer);
        }
      }
    } else {
      buffer = Tensor::Empty(max_reserved_shape, host_array->dtype, local_gpu_device);
      cached_buffers.Set(buffer_cache_key, buffer);
    }
    buffer = buffer.CreateView(host_array.Shape(), host_array->dtype);
    DLTensor copy_dst = *(buffer.operator->());
    Tensor::CopyFromTo(host_array.operator->(), &copy_dst);
    return buffer;
  }
}

void FunctionTable::DebugCallFuncOnAllAllWorker(const String& func_name,
                                                Optional<String> func_args) const {
  if (func_args) {
    std::string args = func_args.value();
    if (this->use_disco) {
      sess->CallPacked(sess->GetGlobalFunc(func_name), args);
    } else {
      static Function func = Function::GetGlobalRequired(func_name);
      func(args);
    }
  } else {
    if (this->use_disco) {
      sess->CallPacked(sess->GetGlobalFunc(func_name));
    } else {
      static Function func = Function::GetGlobalRequired(func_name);
      func();
    }
  }
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
