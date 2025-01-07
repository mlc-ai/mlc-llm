/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/threaded_engine.cc
 * \brief The implementation for threaded serving engine in MLC LLM.
 */
#include "threaded_engine.h"

#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>

#include "../support/json_parser.h"
#include "../support/result.h"
#include "engine.h"
#include "request.h"

namespace mlc {
namespace llm {
namespace serve {

using tvm::Device;
using namespace tvm::runtime;

/*! \brief The threaded engine instruction kind. */
enum class InstructionKind : int {
  kAddRequest = 0,
  kAbortRequest = 1,
  kUnloadEngine = 2,
  kReloadEngine = 3,
  kResetEngine = 4,
  kDebugCallFuncOnAllAllWorker = 5,
};

/*! \brief The implementation of ThreadedEngine. */
class ThreadedEngineImpl : public ThreadedEngine {
 public:
  void InitThreadedEngine(Device device, Optional<PackedFunc> request_stream_callback,
                          Optional<EventTraceRecorder> trace_recorder) final {
    device_ = device;
    CHECK(request_stream_callback.defined())
        << "ThreadedEngine requires request stream callback function, but it is not given.";
    request_stream_callback_ = request_stream_callback.value();
    trace_recorder_ = trace_recorder;
  }

  void Reload(String engine_config_json_str) final {
    // NOTE: important to set this before, we send out
    // reload instruction to the other threads
    // otherwise there can be deadlocks
    reload_finished_ = false;
    bool need_notify = false;
    {
      std::lock_guard<std::mutex> lock(background_loop_mutex_);
      instruction_queue_.emplace_back(InstructionKind::kReloadEngine,
                                      std::move(engine_config_json_str));
      ++pending_request_operation_cnt_;
      need_notify = engine_waiting_;
    }
    if (need_notify) {
      background_loop_cv_.notify_one();
    }
    {
      std::unique_lock<std::mutex> lock(reload_unload_mutex_);
      reload_unload_cv_.wait(lock, [this] { return reload_finished_; });
    }
  }

  void Unload() final {
    // NOTE: important to set this before, we send out
    // reload instruction to the other threads
    // otherwise there can be deadlocks
    // e.g. the other thread finish unload job and set the flag to true
    // then we set it back to false
    unload_finished_ = false;
    bool need_notify = false;
    {
      std::lock_guard<std::mutex> lock(background_loop_mutex_);
      instruction_queue_.emplace_back(InstructionKind::kUnloadEngine, ObjectRef(nullptr));
      ++pending_request_operation_cnt_;
      need_notify = engine_waiting_;
    }
    if (need_notify) {
      background_loop_cv_.notify_one();
    }
    {
      std::unique_lock<std::mutex> lock(reload_unload_mutex_);
      reload_unload_cv_.wait(lock, [this] { return unload_finished_; });
    }
  }

  void Reset() final {
    bool need_notify = false;
    {
      std::lock_guard<std::mutex> lock(background_loop_mutex_);
      instruction_queue_.emplace_back(InstructionKind::kResetEngine, ObjectRef(nullptr));
      ++pending_request_operation_cnt_;
      need_notify = engine_waiting_;
    }
    if (need_notify) {
      background_loop_cv_.notify_one();
    }
  }

  void AddRequest(Request request) final {
    bool need_notify = false;
    {
      std::lock_guard<std::mutex> lock(background_loop_mutex_);
      instruction_queue_.emplace_back(InstructionKind::kAddRequest, request);
      ++pending_request_operation_cnt_;
      need_notify = engine_waiting_;
    }
    if (need_notify) {
      background_loop_cv_.notify_one();
    }
  }

  void AbortRequest(const String& request_id) final {
    bool need_notify = false;
    {
      std::lock_guard<std::mutex> lock(background_loop_mutex_);
      instruction_queue_.emplace_back(InstructionKind::kAbortRequest, request_id);
      ++pending_request_operation_cnt_;
      need_notify = engine_waiting_;
    }
    if (need_notify) {
      background_loop_cv_.notify_one();
    }
  }

  void RunBackgroundLoop() final {
    // The local vectors that load the requests from critical regions.
    std::vector<std::pair<InstructionKind, ObjectRef>> local_instruction_queue;

    while (!exit_now_.load(std::memory_order_relaxed)) {
      {
        std::unique_lock<std::mutex> lock(background_loop_mutex_);
        engine_waiting_ = true;
        background_loop_cv_.wait(lock, [this] {
          return (background_engine_ != nullptr && !background_engine_->Empty()) ||
                 pending_request_operation_cnt_.load() > 0 ||
                 exit_now_.load(std::memory_order_relaxed);
        });
        engine_waiting_ = false;
        local_instruction_queue = instruction_queue_;
        instruction_queue_.clear();
        pending_request_operation_cnt_ = 0;
      }
      for (const auto& [kind, arg] : local_instruction_queue) {
        if (kind == InstructionKind::kAddRequest) {
          CHECK(background_engine_ != nullptr) << "Background engine is not loaded.";
          background_engine_->AddRequest(Downcast<Request>(arg));
        } else if (kind == InstructionKind::kAbortRequest) {
          // in a rare case, abort request can happen after unloading
          // aka background engine is nullptr
          // this happens when the on going generation was interrupted
          // the engine get unloaded, and then abort was called.
          // it is safe to ignore these abort in such case
          if (background_engine_ != nullptr) {
            background_engine_->AbortRequest(Downcast<String>(arg));
          }
        } else if (kind == InstructionKind::kUnloadEngine) {
          EngineUnloadImpl();
        } else if (kind == InstructionKind::kReloadEngine) {
          EngineUnloadImpl();
          EngineReloadImpl(Downcast<String>(arg));
        } else if (kind == InstructionKind::kResetEngine) {
          if (background_engine_ != nullptr) {
            background_engine_->Reset();
          }
        } else if (kind == InstructionKind::kDebugCallFuncOnAllAllWorker) {
          CHECK(background_engine_ != nullptr) << "Background engine is not loaded.";
          Array<ObjectRef> packed_args = Downcast<Array<ObjectRef>>(arg);
          background_engine_->DebugCallFuncOnAllAllWorker(
              Downcast<String>(packed_args[0]), Downcast<Optional<String>>(packed_args[1]));
        } else {
          LOG(FATAL) << "Cannot reach here";
        }
      }
      if (background_engine_ != nullptr) {
        background_engine_->Step();
      }
    }
  }

  void RunBackgroundStreamBackLoop() final {
    // The local vectors that load the request stream callback inputs from critical regions.
    std::vector<Array<RequestStreamOutput>> local_request_stream_callback_inputs;
    std::vector<RequestStreamOutput> flattened_callback_inputs;

    while (!exit_now_.load(std::memory_order_relaxed)) {
      {
        std::unique_lock<std::mutex> lock(request_stream_callback_mutex_);
        stream_callback_waiting_ = true;
        request_stream_callback_cv_.wait(lock, [this] {
          return pending_request_stream_callback_cnt_.load() > 0 ||
                 exit_now_.load(std::memory_order_relaxed);
        });
        stream_callback_waiting_ = false;

        local_request_stream_callback_inputs = request_stream_callback_inputs_;
        request_stream_callback_inputs_.clear();
        pending_request_stream_callback_cnt_ = 0;
      }
      for (const Array<RequestStreamOutput>& callback_inputs :
           local_request_stream_callback_inputs) {
        for (const RequestStreamOutput& callback_input : callback_inputs) {
          flattened_callback_inputs.push_back(callback_input);
        }
      }
      if (!flattened_callback_inputs.empty()) {
        request_stream_callback_(Array<RequestStreamOutput>(flattened_callback_inputs));
      }
      flattened_callback_inputs.clear();
    }
  }

  void ExitBackgroundLoop() final {
    {
      std::lock_guard<std::mutex> lock(background_loop_mutex_);
      exit_now_.store(true);
    }
    background_loop_cv_.notify_one();
    request_stream_callback_cv_.notify_one();
  }

  /************** Query/Profile/Debug **************/

  GenerationConfig GetDefaultGenerationConfig() const final {
    CHECK(default_generation_config_.defined())
        << "The default generation config has not been set.";
    return default_generation_config_.value();
  }

  Request CreateRequest(String id, Array<Data> inputs, String generation_cfg_json_str) const {
    picojson::object config = json::ParseToJSONObject(generation_cfg_json_str);
    auto gen_config = GenerationConfig::FromJSON(config, GetDefaultGenerationConfig());
    CHECK(gen_config.IsOk()) << gen_config.UnwrapErr();
    return Request(std::move(id), std::move(inputs), gen_config.Unwrap());
  }

  EngineConfig GetCompleteEngineConfig() const final {
    CHECK(complete_engine_config_.defined()) << "The engine config has not been set.";
    return complete_engine_config_.value();
  }

  String GetCompleteEngineConfigJSONString() const {
    return GetCompleteEngineConfig()->AsJSONString();
  }

  void DebugCallFuncOnAllAllWorker(const String& func_name, Optional<String> func_args) final {
    bool need_notify = false;
    {
      std::lock_guard<std::mutex> lock(background_loop_mutex_);
      instruction_queue_.emplace_back(InstructionKind::kDebugCallFuncOnAllAllWorker,
                                      Array<ObjectRef>{func_name, func_args});
      ++pending_request_operation_cnt_;
      need_notify = engine_waiting_;
    }
    if (need_notify) {
      background_loop_cv_.notify_one();
    }
  }

 private:
  void EngineReloadImpl(const std::string& engine_config_json_str) {
    auto frequest_stream_callback_wrapper = [this](Array<RequestStreamOutput> delta_outputs) {
      bool need_notify = false;
      {
        std::lock_guard<std::mutex> lock(request_stream_callback_mutex_);
        request_stream_callback_inputs_.push_back(std::move(delta_outputs));
        ++pending_request_stream_callback_cnt_;
        need_notify = stream_callback_waiting_;
      }
      if (need_notify) {
        request_stream_callback_cv_.notify_one();
      }
    };

    FRequestStreamCallback request_stream_callback(frequest_stream_callback_wrapper);
    Result<EngineCreationOutput> output_res =
        Engine::Create(engine_config_json_str, device_, request_stream_callback, trace_recorder_);
    CHECK(output_res.IsOk()) << output_res.UnwrapErr();
    EngineCreationOutput output = output_res.Unwrap();
    background_engine_ = std::move(output.reloaded_engine);
    default_generation_config_ = output.default_generation_cfg;
    complete_engine_config_ = output.completed_engine_config;
    {
      // Wake up the thread waiting for reload finish.
      std::lock_guard<std::mutex> lock(reload_unload_mutex_);
      reload_finished_ = true;
    }
    reload_unload_cv_.notify_one();
  }

  void EngineUnloadImpl() {
    if (background_engine_ != nullptr) {
      background_engine_->AbortAllRequests();
      background_engine_ = nullptr;
      // Clear the allocated memory in cached memory pool.
      const PackedFunc* fclear_memory_manager =
          tvm::runtime::Registry::Get("vm.builtin.memory_manager.clear");
      ICHECK(fclear_memory_manager) << "Cannot find env function vm.builtin.memory_manager.clear";
      (*fclear_memory_manager)();
      default_generation_config_ = NullOpt;
      complete_engine_config_ = NullOpt;
    }
    {
      // Wake up the thread waiting for unload finish.
      std::lock_guard<std::mutex> lock(reload_unload_mutex_);
      unload_finished_ = true;
    }
    reload_unload_cv_.notify_one();
  }

  /*! \brief The device to run models on. */
  Device device_;
  /*! \brief The background normal engine for request processing. */
  std::unique_ptr<Engine> background_engine_;
  /*! \brief The request stream callback. */
  PackedFunc request_stream_callback_;
  /*! \brief Event trace recorder. */
  Optional<EventTraceRecorder> trace_recorder_;

  /*! \brief complete engine config. */
  Optional<EngineConfig> complete_engine_config_;
  /*! \brief The default generation config. */
  Optional<GenerationConfig> default_generation_config_;

  /*! \brief The mutex ensuring only one thread can access critical regions. */
  std::mutex background_loop_mutex_;
  std::mutex request_stream_callback_mutex_;
  std::mutex reload_unload_mutex_;
  /*! \brief The condition variable preventing threaded engine from spinning. */
  std::condition_variable background_loop_cv_;
  std::condition_variable request_stream_callback_cv_;
  std::condition_variable reload_unload_cv_;
  /*! \brief A boolean flag denoting if the engine needs to exit background loop. */
  std::atomic<bool> exit_now_ = false;

  /************** Critical Regions **************/
  /*!
   * \brief The instruction queue for the threaded engine.
   * The instructions include:
   *  - requests to add into the background engine,
   *  - requests to abort from the background engine,
   *  - engine unload/reload,
   *  - and other debugging instructions.
   * Elements are sended from other threads and consumed by
   * the threaded engine in the background loop.
   */
  std::vector<std::pair<InstructionKind, ObjectRef>> instruction_queue_;
  /*!
   * \brief The delta outputs to pass through callback.
   * Elements are sended from the background loop thread and
   * consumed by the foreground thread.
   */
  std::vector<Array<RequestStreamOutput>> request_stream_callback_inputs_;
  /*!
   * \brief Number of pending request operations, should be the size of
   * `requests_to_add_` and `requests_to_abort_`.
   */
  std::atomic<int> pending_request_operation_cnt_ = 0;
  /*!
   * \brief Number of pending request stream callback invocations.
   * It should be the size of `request_stream_callback_inputs_`.
   */
  std::atomic<int> pending_request_stream_callback_cnt_ = 0;
  /*! \brief A boolean flag indicating if the engine is waiting for new requests/aborts. */
  bool engine_waiting_ = false;
  /*! \brief A boolean flag indicating if the stream callback loop is waiting. */
  bool stream_callback_waiting_ = false;
  /*! \brief A boolean indicating if the engine reload has finished. */
  bool reload_finished_ = false;
  /*! \brief A boolean indicating if the engine unload has finished. */
  bool unload_finished_ = false;
};

/*! \brief The implementation of ThreadedEngine. */
class ThreadedEngineModule : public ThreadedEngineImpl, public ModuleNode {
 public:
  TVM_MODULE_VTABLE_BEGIN("mlc.serve.async_threaded_engine");
  TVM_MODULE_VTABLE_ENTRY("init_threaded_engine", &ThreadedEngineImpl::InitThreadedEngine);
  TVM_MODULE_VTABLE_ENTRY("reload", &ThreadedEngineImpl::Reload);
  TVM_MODULE_VTABLE_ENTRY("add_request", &ThreadedEngineImpl::AddRequest);
  TVM_MODULE_VTABLE_ENTRY("create_request", &ThreadedEngineImpl::CreateRequest);
  TVM_MODULE_VTABLE_ENTRY("abort_request", &ThreadedEngineImpl::AbortRequest);
  TVM_MODULE_VTABLE_ENTRY("run_background_loop", &ThreadedEngineImpl::RunBackgroundLoop);
  TVM_MODULE_VTABLE_ENTRY("run_background_stream_back_loop",
                          &ThreadedEngineImpl::RunBackgroundStreamBackLoop);
  TVM_MODULE_VTABLE_ENTRY("exit_background_loop", &ThreadedEngineImpl::ExitBackgroundLoop);
  TVM_MODULE_VTABLE_ENTRY("get_complete_engine_config",
                          &ThreadedEngineImpl::GetCompleteEngineConfigJSONString);
  TVM_MODULE_VTABLE_ENTRY("reset", &ThreadedEngineImpl::Reset);
  TVM_MODULE_VTABLE_ENTRY("debug_call_func_on_all_worker",
                          &ThreadedEngineImpl::DebugCallFuncOnAllAllWorker);
  TVM_MODULE_VTABLE_END();
};

TVM_REGISTER_GLOBAL("mlc.serve.create_threaded_engine").set_body_typed([]() {
  return Module(make_object<ThreadedEngineModule>());
});

std::unique_ptr<ThreadedEngine> ThreadedEngine::Create() {
  std::unique_ptr<ThreadedEngineImpl> threaded_engine = std::make_unique<ThreadedEngineImpl>();
  return std::move(threaded_engine);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
