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

#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

using namespace tvm::runtime;

@implementation JSONFFIEngine {
  // Internal c++ classes
  // internal module backed by JSON FFI
  Module json_ffi_engine_;
  // member functions
  PackedFunc init_background_engine_func_;
  PackedFunc unload_func_;
  PackedFunc reload_func_;
  PackedFunc reset_func_;
  PackedFunc chat_completion_func_;
  PackedFunc abort_func_;
  PackedFunc run_background_loop_func_;
  PackedFunc run_background_stream_back_loop_func_;
  PackedFunc exit_background_loop_func_;
}

- (instancetype)init {
  if (self = [super init]) {
    // load chat module
    const PackedFunc* f_json_ffi_create = Registry::Get("mlc.json_ffi.CreateJSONFFIEngine");
    ICHECK(f_json_ffi_create) << "Cannot find mlc.json_ffi.CreateJSONFFIEngine";
    json_ffi_engine_ = (*f_json_ffi_create)();
    init_background_engine_func_ = json_ffi_engine_->GetFunction("init_background_engine");
    reload_func_ = json_ffi_engine_->GetFunction("reload");
    unload_func_ = json_ffi_engine_->GetFunction("unload");
    reset_func_ = json_ffi_engine_->GetFunction("reset");
    chat_completion_func_ = json_ffi_engine_->GetFunction("chat_completion");
    abort_func_ = json_ffi_engine_->GetFunction("abort");
    run_background_loop_func_ = json_ffi_engine_->GetFunction("run_background_loop");
    run_background_stream_back_loop_func_ =
        json_ffi_engine_->GetFunction("run_background_stream_back_loop");
    exit_background_loop_func_ = json_ffi_engine_->GetFunction("exit_background_loop");

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
  TypedPackedFunc<void(String)> internal_stream_callback([streamCallback](String value) {
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
