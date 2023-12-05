#define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>
#define TVM_USE_LIBBACKTRACE 0

#include <android/log.h>
#include <dlfcn.h>
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <tvm/runtime/c_runtime_api.h>

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
