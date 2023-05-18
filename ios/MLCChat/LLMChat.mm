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

// use a global singleton to implement llm chat
class LLMChatModuleWrapper {
 public:
  LLMChatModuleWrapper() {
    // load module
    const PackedFunc* fcreate = tvm::runtime::Registry::Get("mlc.llm_chat_create");
    ICHECK(fcreate) << "Cannot find mlc.llm_chat_create";

    llm_chat_ = (*fcreate)(static_cast<int>(kDLMetal), 0);

    reload_func_ = llm_chat_->GetFunction("reload");
    unload_func_ = llm_chat_->GetFunction("unload");

    encode_func_ = llm_chat_->GetFunction("encode");
    decode_func_ = llm_chat_->GetFunction("decode");
    get_message_ = llm_chat_->GetFunction("get_message");
    stopped_func_ = llm_chat_->GetFunction("stopped");
    reset_chat_func_ = llm_chat_->GetFunction("reset_chat");
    runtime_stats_text_func_ = llm_chat_->GetFunction("runtime_stats_text");

    ICHECK(reload_func_ != nullptr);
    ICHECK(unload_func_ != nullptr);
    ICHECK(encode_func_ != nullptr);
    ICHECK(decode_func_ != nullptr);
    ICHECK(get_message_ != nullptr);
    ICHECK(stopped_func_ != nullptr);
    ICHECK(reset_chat_func_ != nullptr);
    ICHECK(runtime_stats_text_func_ != nullptr);
  }

  void Unload() { unload_func_(); }

  void Reload(const std::string& model_lib, const std::string& model_path) {
    std::string lib_prefix = model_lib;
    std::replace(lib_prefix.begin(), lib_prefix.end(), '-', '_');
    lib_prefix += '_';
    tvm::runtime::Module lib = (*tvm::runtime::Registry::Get("runtime.SystemLib"))(lib_prefix);
    reload_func_(lib, model_path);
  }

  void Evaluate() {
    LOG(INFO) << "Total-mem-budget=" << os_proc_available_memory() / (1 << 20) << "MB";
    llm_chat_->GetFunction("evaluate")();
    LOG(INFO) << "Left-mem-budget=" << os_proc_available_memory() / (1 << 20) << "MB";
  }

  std::string GetMessage() {
    return get_message_();
  }

  void Encode(std::string prompt) { encode_func_(prompt); }

  bool Stopped() { return stopped_func_(); }

  void Decode() { decode_func_(); }

  std::string RuntimeStatsText() { return runtime_stats_text_func_(); }

  void ResetChat() { reset_chat_func_(); }

  static LLMChatModuleWrapper* Global() {
    static LLMChatModuleWrapper* inst = new LLMChatModuleWrapper();
    return inst;
  }

 private:
  Module llm_chat_;
  PackedFunc unload_func_;
  PackedFunc reload_func_;
  PackedFunc encode_func_;
  PackedFunc decode_func_;
  PackedFunc get_message_;
  PackedFunc stopped_func_;
  PackedFunc reset_chat_func_;
  PackedFunc runtime_stats_text_func_;
};

@implementation LLMChatInstance

- (void)initialize {
  LLMChatModuleWrapper::Global();
}

- (void)unload {
  LLMChatModuleWrapper::Global()->Unload();
}

- (void)reload:(NSString*)model_lib modelPath:(NSString*)modelPath {
  LLMChatModuleWrapper::Global()->Reload(model_lib.UTF8String, modelPath.UTF8String);
}

- (void)evaluate {
  LLMChatModuleWrapper::Global()->Evaluate();
}

- (void)encode:(NSString*)prompt {
  LLMChatModuleWrapper::Global()->Encode(prompt.UTF8String);
}

- (void)decode {
  LLMChatModuleWrapper::Global()->Decode();
}

- (NSString*)getMessage {
  std::string ret = LLMChatModuleWrapper::Global()->GetMessage();
  return [NSString stringWithUTF8String:ret.c_str()];
}

- (bool)stopped {
  return LLMChatModuleWrapper::Global()->Stopped();
}

- (void)reset {
  return LLMChatModuleWrapper::Global()->ResetChat();
}
- (NSString*)runtimeStatsText {
  std::string ret = LLMChatModuleWrapper::Global()->RuntimeStatsText();
  return [NSString stringWithUTF8String:ret.c_str()];
}
@end
