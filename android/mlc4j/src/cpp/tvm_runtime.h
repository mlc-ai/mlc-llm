#define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>
#define TVM_USE_LIBBACKTRACE 0

#include <android/log.h>
#include <dlfcn.h>
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>

#include <ffi/backtrace.cc>
#include <ffi/container.cc>
#include <ffi/dtype.cc>
#include <ffi/error.cc>
#include <ffi/extra/env_c_api.cc>
#include <ffi/extra/env_context.cc>
#include <ffi/extra/library_module.cc>
#include <ffi/extra/library_module_dynamic_lib.cc>
#include <ffi/extra/library_module_system_lib.cc>
#include <ffi/extra/module.cc>
#include <ffi/function.cc>
#include <ffi/object.cc>
#include <runtime/cpu_device_api.cc>
#include <runtime/device_api.cc>
#include <runtime/file_utils.cc>
#include <runtime/logging.cc>
#include <runtime/memory/memory_manager.cc>
#include <runtime/module.cc>
#include <runtime/nvtx.cc>
#include <runtime/opencl/opencl_device_api.cc>
#include <runtime/opencl/opencl_module.cc>
#include <runtime/opencl/opencl_wrapper/opencl_wrapper.cc>
#include <runtime/profiling.cc>
#include <runtime/source_utils.cc>
#include <runtime/tensor.cc>
#include <runtime/thread_pool.cc>
#include <runtime/threading_backend.cc>
#include <runtime/vm/attn_backend.cc>
#include <runtime/vm/builtin.cc>
#include <runtime/vm/bytecode.cc>
#include <runtime/vm/executable.cc>
#include <runtime/vm/kv_state.cc>
#include <runtime/vm/paged_kv_cache.cc>
#include <runtime/vm/rnn_state.cc>
#include <runtime/vm/tensor_cache_support.cc>
#include <runtime/vm/vm.cc>
#include <runtime/workspace_pool.cc>

static_assert(TVM_LOG_CUSTOMIZE == 1, "TVM_LOG_CUSTOMIZE must be 1");

namespace tvm {
namespace runtime {
namespace detail {
// Override logging mechanism
[[noreturn]] void LogFatalImpl(const std::string& file, int lineno, const std::string& message) {
  std::string m = file + ":" + std::to_string(lineno) + ": " + message;
  __android_log_write(ANDROID_LOG_FATAL, "TVM_RUNTIME", m.c_str());
  throw InternalError(file, lineno, message);
}
void LogMessageImpl(const std::string& file, int lineno, int level, const std::string& message) {
  std::string m = file + ":" + std::to_string(lineno) + ": " + message;
  __android_log_write(ANDROID_LOG_DEBUG + level, "TVM_RUNTIME", m.c_str());
}

}  // namespace detail
}  // namespace runtime
}  // namespace tvm
