//
//  LLMEngine.mm
//  LLMEngine
//
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#include <os/proc.h>

#include "LLMEngine.h"

#define TVM_USE_LIBBACKTRACE 0
#define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/module.h>

using namespace tvm::runtime;
using tvm::ffi::Function;
using tvm::ffi::Module;
using tvm::ffi::Optional;
using tvm::ffi::String;
using tvm::ffi::TypedFunction;

@implementation JSONFFIEngine {
  // Internal c++ classes
  // internal module backed by JSON FFI
  Optional<Module> json_ffi_engine_;
  // member functions
  Function init_background_engine_func_;
  Function unload_func_;
  Function reload_func_;
  Function reset_func_;
  Function chat_completion_func_;
  Function abort_func_;
  Function run_background_loop_func_;
  Function run_background_stream_back_loop_func_;
  Function exit_background_loop_func_;
}

- (instancetype)init {
  if (self = [super init]) {
    // load chat module
    Function f_json_ffi_create = Function::GetGlobalRequired("mlc.json_ffi.CreateJSONFFIEngine");
    json_ffi_engine_ = f_json_ffi_create().cast<Module>();
    init_background_engine_func_ =
        json_ffi_engine_.value()->GetFunction("init_background_engine").value_or(Function(nullptr));
    reload_func_ = json_ffi_engine_.value()->GetFunction("reload").value_or(Function(nullptr));
    unload_func_ = json_ffi_engine_.value()->GetFunction("unload").value_or(Function(nullptr));
    reset_func_ = json_ffi_engine_.value()->GetFunction("reset").value_or(Function(nullptr));
    chat_completion_func_ =
        json_ffi_engine_.value()->GetFunction("chat_completion").value_or(Function(nullptr));
    abort_func_ = json_ffi_engine_.value()->GetFunction("abort").value_or(Function(nullptr));
    run_background_loop_func_ =
        json_ffi_engine_.value()->GetFunction("run_background_loop").value_or(Function(nullptr));
    run_background_stream_back_loop_func_ = json_ffi_engine_.value()
                                                ->GetFunction("run_background_stream_back_loop")
                                                .value_or(Function(nullptr));
    exit_background_loop_func_ =
        json_ffi_engine_.value()->GetFunction("exit_background_loop").value_or(Function(nullptr));

    ICHECK(init_background_engine_func_ != nullptr);
    ICHECK(reload_func_ != nullptr);
    ICHECK(unload_func_ != nullptr);
    ICHECK(reset_func_ != nullptr);
    ICHECK(chat_completion_func_ != nullptr);
    ICHECK(abort_func_ != nullptr);
    ICHECK(run_background_loop_func_ != nullptr);
    ICHECK(run_background_stream_back_loop_func_ != nullptr);
    ICHECK(exit_background_loop_func_ != nullptr);
  }
  return self;
}

- (void)initBackgroundEngine:(void (^)(NSString*))streamCallback {
  TypedFunction<void(String)> internal_stream_callback([streamCallback](String value) {
    streamCallback([NSString stringWithUTF8String:value.c_str()]);
  });
  int device_type = kDLMetal;
  int device_id = 0;
  init_background_engine_func_(device_type, device_id, internal_stream_callback);
}

- (void)reload:(NSString*)engineConfigJson {
  std::string engine_config = engineConfigJson.UTF8String;
  reload_func_(engine_config);
}

- (void)unload {
  unload_func_();
}

- (void)reset {
  reset_func_();
}

- (void)chatCompletion:(NSString*)requestJSON requestID:(NSString*)requestID {
  std::string request_json = requestJSON.UTF8String;
  std::string request_id = requestID.UTF8String;
  chat_completion_func_(request_json, request_id);
}

- (void)abort:(NSString*)requestID {
  std::string request_id = requestID.UTF8String;
  abort_func_(request_id);
}

- (void)runBackgroundLoop {
  run_background_loop_func_();
}

- (void)runBackgroundStreamBackLoop {
  run_background_stream_back_loop_func_();
}

- (void)exitBackgroundLoop {
  exit_background_loop_func_();
}

@end
