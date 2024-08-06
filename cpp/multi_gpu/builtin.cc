/*!
 * \file builtin.cc
 * \brief Multi-GPU builtin functions in MLC LLM.
 */
#ifndef MLC_SINGLE_GPU_ONLY

#include <tvm/runtime/disco/builtin.h>
#include <tvm/runtime/disco/disco_worker.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/relax_vm/vm.h>

namespace mlc {
namespace llm {
namespace multi_gpu {

using namespace tvm::runtime;

ObjectRef DispatchFunctionByGroup(TVMArgValue vm_arg, Array<Array<ObjectRef>> funcs_and_args) {
  using namespace relax_vm;
  VirtualMachine* vm = VirtualMachine::GetContextPtr(vm_arg);
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  int world_size = worker->num_workers;
  int group_size = worker->num_workers / worker->num_groups;
  int num_group = world_size / group_size;
  CHECK_EQ(funcs_and_args.size(), num_group)
      << "Number of groups mismatches. There are " << num_group
      << " groups while the function/arg array has " << funcs_and_args.size() << " elements.";

  int group_id = worker->worker_id / group_size;
  CHECK(!funcs_and_args[group_id].empty()) << "No function is provided for group " << group_id;
  VMClosure func = Downcast<VMClosure>(funcs_and_args[group_id][0]);

  int num_args = static_cast<int>(funcs_and_args[group_id].size()) - 1;
  std::vector<TVMValue> values;
  std::vector<int> type_codes;
  values.resize(num_args);
  type_codes.resize(num_args);
  TVMArgsSetter setter(values.data(), type_codes.data());
  for (int i = 0; i < num_args; ++i) {
    // NOTE: Need explicily define `arg` so that the argument does not
    // have type code kTVMObjectRValueRefArg.
    ObjectRef arg = funcs_and_args[group_id][1 + i];
    setter(i, arg);
  }

  TVMRetValue rv;
  vm->InvokeClosurePacked(Downcast<VMClosure>(funcs_and_args[group_id][0]),
                          TVMArgs(values.data(), type_codes.data(), num_args), &rv);
  return rv;
}

ObjectRef SendFromLastGroupToWorker0(NDArray send, Optional<NDArray> recv, ShapeTuple shape,
                                     DataType dtype) {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  int worker_id = worker->worker_id;
  int world_size = worker->num_workers;
  int group_size = worker->num_workers / worker->num_groups;
  CHECK_NE(world_size, group_size) << "Cannot perform when there is only one group.";
  int sender_id = world_size - group_size;
  if (worker_id == 0) {
    CHECK(recv.defined()) << "The receive NDArray is undefined for worker 0.";
    NDArray recv_arr = recv.value().CreateView(shape, dtype);
    RecvFromWorker(recv_arr, sender_id);
    return recv_arr;
  } else if (worker_id == sender_id) {
    CHECK_EQ(DataType(send->dtype), dtype)
        << "The src NDArray has mismatched dtype than the expected dtype.";
    CHECK_EQ(send->ndim, shape.size())
        << "The src NDArray has mismatched shape than the expected shape.";
    for (int i = 0; i < send->ndim; ++i) {
      CHECK_EQ(send->shape[i], shape[i])
          << "The src NDArray has mismatched shape than the expected shape.";
    }
    SendToWorker(send, /*receiver_id=*/0);
    return recv;
  }

  // We only process for worker 0 and the first worker of the last group.
  // For other workers, we return the input object.
  return recv;
}

TVM_REGISTER_GLOBAL("mlc.multi_gpu.DispatchFunctionByGroup")
    .set_body_typed(DispatchFunctionByGroup);
TVM_REGISTER_GLOBAL("mlc.multi_gpu.SendFromLastGroupToWorker0")
    .set_body_typed(SendFromLastGroupToWorker0);

}  // namespace multi_gpu
}  // namespace llm
}  // namespace mlc

#endif  // MLC_SINGLE_GPU_ONLY
