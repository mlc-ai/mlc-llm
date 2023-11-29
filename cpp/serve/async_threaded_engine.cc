/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/async_threaded_engine.cc
 * \brief The implementation for asynchronous threaded serving engine in MLC LLM.
 */
#include "async_threaded_engine.h"

#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <atomic>
#include <condition_variable>
#include <mutex>

#include "engine.h"
#include "request.h"

namespace mlc {
namespace llm {
namespace serve {

using tvm::Device;
using namespace tvm::runtime;

/*! \brief The implementation of AsyncThreadedEngine. */
class AsyncThreadedEngineImpl : public AsyncThreadedEngine, public ModuleNode {
 public:
  TVM_MODULE_VTABLE_BEGIN("mlc.serve.async_threaded_engine");
  TVM_MODULE_VTABLE_ENTRY("add_request", &AsyncThreadedEngineImpl::AddRequest);
  TVM_MODULE_VTABLE_ENTRY("abort_request", &AsyncThreadedEngineImpl::AbortRequest);
  TVM_MODULE_VTABLE_ENTRY("run_background_loop", &AsyncThreadedEngineImpl::RunBackgroundLoop);
  TVM_MODULE_VTABLE_ENTRY("exit_background_loop", &AsyncThreadedEngineImpl::ExitBackgroundLoop);
  TVM_MODULE_VTABLE_END();

  explicit AsyncThreadedEngineImpl(std::unique_ptr<Engine> background_engine)
      : background_engine_(std::move(background_engine)) {}

  static tvm::runtime::Module Create(std::unique_ptr<Engine> background_engine) {
    return Module(make_object<AsyncThreadedEngineImpl>(std::move(background_engine)));
  }

  void AddRequest(Request request) final {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      requests_to_add_.push_back(request);
      ++pending_operation_cnt_;
    }
    cv_.notify_one();
  }

  void AbortRequest(const String& request_id) final {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      requests_to_abort_.push_back(request_id);
      ++pending_operation_cnt_;
    }
    cv_.notify_one();
  }

  void RunBackgroundLoop() final {
    // The local vectors that load the requests in critical regions.
    std::vector<Request> local_requests_to_add;
    std::vector<String> local_requests_to_abort;

    while (!exit_now_.load(std::memory_order_relaxed)) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] {
          return !background_engine_->Empty() || pending_operation_cnt_.load() > 0 ||
                 exit_now_.load(std::memory_order_relaxed);
        });

        local_requests_to_add = requests_to_add_;
        local_requests_to_abort = requests_to_abort_;
        requests_to_add_.clear();
        requests_to_abort_.clear();
        pending_operation_cnt_ = 0;
      }
      for (Request request : local_requests_to_add) {
        background_engine_->AddRequest(request);
      }
      for (String request_id : local_requests_to_abort) {
        background_engine_->AbortRequest(request_id);
      }
      background_engine_->Step();
    }
  }

  void ExitBackgroundLoop() final {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      exit_now_.store(true);
    }
    cv_.notify_one();
  }

 private:
  /*! \brief The background normal engine for request processing. */
  std::unique_ptr<Engine> background_engine_;

  /*! \brief The mutex ensuring only one thread can access critical regions. */
  std::mutex mutex_;
  /*! \brief The condition variable preventing threaded engine from spinning. */
  std::condition_variable cv_;
  /*! \brief A boolean flag denoting if the engine needs to exit background loop. */
  std::atomic<bool> exit_now_ = false;

  /************** Critical Regions **************/
  /*!
   * \brief The requests to add into the background engine.
   * Elements are sended from other threads and consumed by
   * the threaded engine in the background loop.
   */
  std::vector<Request> requests_to_add_;
  /*!
   * \brief The requests to abort from the background engine.
   * Elements are sended from other threads and consumed by
   * the threaded engine in the background loop.
   */
  std::vector<String> requests_to_abort_;
  /*!
   * \brief Number of pending operations, should be the size of
   * `requests_to_add_` and `requests_to_abort_`.
   */
  std::atomic<int> pending_operation_cnt_ = 0;
};

std::unique_ptr<AsyncThreadedEngine> AsyncThreadedEngine::Create(std::unique_ptr<Engine> engine) {
  return std::make_unique<AsyncThreadedEngineImpl>(std::move(engine));
}

TVM_REGISTER_GLOBAL("mlc.serve.create_threaded_engine").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = AsyncThreadedEngineImpl::Create(CreateEnginePacked(args));
});

}  // namespace serve
}  // namespace llm
}  // namespace mlc
