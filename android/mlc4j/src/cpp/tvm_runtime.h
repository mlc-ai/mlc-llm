#define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>
#define TVM_USE_LIBBACKTRACE 0

#include <android/log.h>
#include <dlfcn.h>
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>

#define STRINGIFY_MACRO(x) STR(x)
#define STR(x) #x
#define EXPAND(x) x
#define CONCAT(n1, n2) STRINGIFY_MACRO(EXPAND(n1) EXPAND(n2))

// clang-format off
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/c_runtime_api.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/container.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/cpu_device_api.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/file_utils.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/library_module.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/logging.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/module.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/ndarray.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/object.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/opencl/opencl_device_api.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/opencl/opencl_module.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/opencl/opencl_wrapper/opencl_wrapper.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/opencl/texture_pool.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/profiling.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/registry.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/source_utils.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/system_library.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/thread_pool.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/threading_backend.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/workspace_pool.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/memory/memory_manager.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/nvtx.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/relax_vm/builtin.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/relax_vm/bytecode.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/relax_vm/executable.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/relax_vm/kv_state.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/relax_vm/ndarray_cache_support.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/relax_vm/paged_kv_cache.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/relax_vm/rnn_state.cc)
#include CONCAT(TVM_SOURCE_DIR,/src/runtime/relax_vm/vm.cc)
// clang-format on

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
