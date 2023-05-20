//
//  LLMChat.mm
//  LLMChat
//
#import <Foundation/Foundation.h>
#include <os/proc.h>

#include "MLCChat-Bridging-Header.h"

#define TVM_USE_LIBBACKTRACE 0
#define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>

#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

using namespace tvm::runtime;

@implementation ChatModule {
  // Internal c++ classes
  Module llm_chat_;
  PackedFunc unload_func_;
  PackedFunc reload_func_;
  PackedFunc prefill_func_;
  PackedFunc decode_func_;
  PackedFunc get_message_;
  PackedFunc stopped_func_;
  PackedFunc reset_chat_func_;
  PackedFunc runtime_stats_text_func_;
}

- (instancetype)init {
  if (self = [super init]) {
    // load module
    const PackedFunc* fcreate = tvm::runtime::Registry::Get("mlc.llm_chat_create");
    ICHECK(fcreate) << "Cannot find mlc.llm_chat_create";

    llm_chat_ = (*fcreate)(static_cast<int>(kDLMetal), 0);

    reload_func_ = llm_chat_->GetFunction("reload");
    unload_func_ = llm_chat_->GetFunction("unload");
    prefill_func_ = llm_chat_->GetFunction("prefill");
    decode_func_ = llm_chat_->GetFunction("decode");
    get_message_ = llm_chat_->GetFunction("get_message");
    stopped_func_ = llm_chat_->GetFunction("stopped");
    reset_chat_func_ = llm_chat_->GetFunction("reset_chat");
    runtime_stats_text_func_ = llm_chat_->GetFunction("runtime_stats_text");

    ICHECK(reload_func_ != nullptr);
    ICHECK(unload_func_ != nullptr);
    ICHECK(prefill_func_ != nullptr);
    ICHECK(decode_func_ != nullptr);
    ICHECK(get_message_ != nullptr);
    ICHECK(stopped_func_ != nullptr);
    ICHECK(reset_chat_func_ != nullptr);
    ICHECK(runtime_stats_text_func_ != nullptr);
  }
  return self;
}

- (void)unload {
  unload_func_();
}

- (void)reload:(NSString*)modelLib modelPath:(NSString*)modelPath {
  std::string lib_prefix = modelLib.UTF8String;
  std::string model_path = modelPath.UTF8String;
  std::replace(lib_prefix.begin(), lib_prefix.end(), '-', '_');
  lib_prefix += '_';
  tvm::runtime::Module lib = (*tvm::runtime::Registry::Get("runtime.SystemLib"))(lib_prefix);
  reload_func_(lib, model_path);
}

- (void)resetChat {
  reset_chat_func_();
}

- (void)prefill:(NSString*)input {
  std::string prompt = input.UTF8String;
  prefill_func_(prompt);
}

- (void)decode {
  decode_func_();
}

- (NSString*)getMessage {
  std::string ret = get_message_();
  return [NSString stringWithUTF8String:ret.c_str()];
}

- (bool)stopped {
  return stopped_func_().operator bool();
}

- (NSString*)runtimeStatsText {
  std::string ret = runtime_stats_text_func_();
  return [NSString stringWithUTF8String:ret.c_str()];
}

- (void)evaluate {
  LOG(INFO) << "Total-mem-budget=" << os_proc_available_memory() / (1 << 20) << "MB";
  llm_chat_->GetFunction("evaluate")();
  LOG(INFO) << "Left-mem-budget=" << os_proc_available_memory() / (1 << 20) << "MB";
}

@end
