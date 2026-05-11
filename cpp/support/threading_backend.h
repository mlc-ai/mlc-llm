/*!
 * \file support/threading_backend.h
 * \brief Compatibility shim providing the threading helpers that used to live
 *        at <tvm/runtime/threading_backend.h>. The public TVM header was
 *        removed in a runtime refactor, but the underlying symbols
 *        (TVMBackendParallelLaunch, threading::MaxConcurrency,
 *        threading::SetMaxConcurrency) are still exported from libtvm_runtime.
 *        We re-declare the inline parallel-for template here so existing
 *        mlc-llm call sites keep compiling.
 */
#ifndef MLC_LLM_SUPPORT_THREADING_BACKEND_H_
#define MLC_LLM_SUPPORT_THREADING_BACKEND_H_

#include <tvm/runtime/base.h>
#include <tvm/runtime/c_backend_api.h>

#include <algorithm>
#include <cstdint>

namespace tvm {
namespace runtime {
namespace threading {

/*! \return the maximum number of effective workers for this system. */
int MaxConcurrency();
/*! \brief Setting the maximum number of available cores. */
void SetMaxConcurrency(int value);

}  // namespace threading

namespace detail {

template <typename T>
struct ParallelForWithThreadingBackendLambdaInvoker {
  static int TVMParallelLambdaInvoke(int task_id, TVMParallelGroupEnv* penv, void* cdata) {
    int num_task = penv->num_task;
    T* lambda_ptr = static_cast<T*>(cdata);
    (*lambda_ptr)(task_id, num_task);
    return 0;
  }
};

template <typename T>
inline void parallel_launch_with_threading_backend(T flambda) {
  void* cdata = &flambda;
  TVMBackendParallelLaunch(ParallelForWithThreadingBackendLambdaInvoker<T>::TVMParallelLambdaInvoke,
                           cdata, /*num_task=*/0);
}

}  // namespace detail

template <typename T>
inline void parallel_for_with_threading_backend(T flambda, int64_t begin, int64_t end) {
  if (end - begin == 1) {
    flambda(begin);
    return;
  }
  auto flaunch = [begin, end, flambda](int task_id, int num_task) {
    int64_t total_len = end - begin;
    int64_t step = (total_len + num_task - 1) / num_task;
    int64_t local_begin = std::min(begin + step * task_id, end);
    int64_t local_end = std::min(local_begin + step, end);
    for (int64_t i = local_begin; i < local_end; ++i) {
      flambda(i);
    }
  };
  detail::parallel_launch_with_threading_backend(flaunch);
}

}  // namespace runtime
}  // namespace tvm

#endif  // MLC_LLM_SUPPORT_THREADING_BACKEND_H_
