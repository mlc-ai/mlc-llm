//
//  LLMChat.mm
//  LLMChat
//
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#include <os/proc.h>

#include "LLMChat.h"

#define TVM_USE_LIBBACKTRACE 0
#define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>

#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

using namespace tvm::runtime;

enum PlaceInPrompt : int {
  // The input message should have role names and corresponding seperators appended both
  // prior to it and after it, making it a complete prompt.
  kAll,
  // The input message is only the beginning part of a prompt, no role name and separator should be
  // appended after the message since there will be future messages appended after the message.
  kBegin,
  // The input message is in the middle of a prompt, nothing should be appended before or after the message.
  kMiddle,
  // The input message is the ending part of a prompt, no role name and separator should be appended
  // prior to it since the message is concatenated to some prior messages.
  kEnd,
};

@implementation ChatModule {
  // Internal c++ classes
  // chat-related module and functions
  Module llm_chat_;
  PackedFunc unload_func_;
  PackedFunc reload_func_;
  PackedFunc prefill_func_;
  PackedFunc embed_func_;
  PackedFunc prefill_with_embed_func_;
  PackedFunc decode_func_;
  PackedFunc get_message_;
  PackedFunc stopped_func_;
  PackedFunc reset_chat_func_;
  PackedFunc runtime_stats_text_func_;
  PackedFunc process_system_prompts_func_;
  // image-related module and functions
  Module llm_image_mod_;
  PackedFunc image_mod_unload_func_;
  PackedFunc image_mod_reload_func_;
  PackedFunc image_mod_embed_func_;
  PackedFunc image_mod_reset_func_;
  PackedFunc image_mod_runtime_stats_text_func_;
  // flag for first text input after image uploading
  bool first_input_after_image;
}

- (instancetype)init {
  if (self = [super init]) {
    // load chat module
    const PackedFunc* f_chat_create = Registry::Get("mlc.llm_chat_create");
    ICHECK(f_chat_create) << "Cannot find mlc.llm_chat_create";
    llm_chat_ = (*f_chat_create)(static_cast<int>(kDLMetal), 0);
    // load image module
    const PackedFunc* f_image_mod_create = Registry::Get("mlc.llm_image_mod_create");
    ICHECK(f_image_mod_create) << "Cannot find mlc.llm_image_mod_create";
    llm_image_mod_ = (*f_image_mod_create)(static_cast<int>(kDLMetal), 0);

    // chat-related functions
    reload_func_ = llm_chat_->GetFunction("reload");
    unload_func_ = llm_chat_->GetFunction("unload");
    prefill_func_ = llm_chat_->GetFunction("prefill");
    embed_func_ = llm_chat_->GetFunction("embed");
    prefill_with_embed_func_ = llm_chat_->GetFunction("prefill_with_embed");
    decode_func_ = llm_chat_->GetFunction("decode");
    get_message_ = llm_chat_->GetFunction("get_message");
    stopped_func_ = llm_chat_->GetFunction("stopped");
    reset_chat_func_ = llm_chat_->GetFunction("reset_chat");
    runtime_stats_text_func_ = llm_chat_->GetFunction("runtime_stats_text");
    process_system_prompts_func_ = llm_chat_->GetFunction("process_system_prompts");
    // image-module-related functions
    image_mod_reload_func_ = llm_image_mod_->GetFunction("reload");
    image_mod_unload_func_ = llm_image_mod_->GetFunction("unload");
    image_mod_embed_func_ = llm_image_mod_->GetFunction("embed");
    image_mod_reset_func_ = llm_image_mod_->GetFunction("reset_image_mod");
    image_mod_runtime_stats_text_func_ = llm_image_mod_->GetFunction("runtime_stats_text");
    first_input_after_image = false;

    ICHECK(reload_func_ != nullptr);
    ICHECK(unload_func_ != nullptr);
    ICHECK(prefill_func_ != nullptr);
    ICHECK(embed_func_ != nullptr);
    ICHECK(prefill_with_embed_func_ != nullptr);
    ICHECK(decode_func_ != nullptr);
    ICHECK(get_message_ != nullptr);
    ICHECK(stopped_func_ != nullptr);
    ICHECK(reset_chat_func_ != nullptr);
    ICHECK(runtime_stats_text_func_ != nullptr);
    ICHECK(process_system_prompts_func_ != nullptr);
    ICHECK(image_mod_unload_func_ != nullptr);
    ICHECK(image_mod_reload_func_ != nullptr);
    ICHECK(image_mod_embed_func_ != nullptr);
    ICHECK(image_mod_reset_func_ != nullptr);
    ICHECK(image_mod_runtime_stats_text_func_ != nullptr);
  }
  return self;
}

- (void)unload {
  unload_func_();
}

- (void)reload:(NSString*)modelLib modelPath:(NSString*)modelPath appConfigJson:(NSString*)appConfigJson {
  std::string lib_prefix = modelLib.UTF8String;
  std::string model_path = modelPath.UTF8String;
  std::string app_config_json = appConfigJson.UTF8String;
  std::replace(lib_prefix.begin(), lib_prefix.end(), '-', '_');
  lib_prefix += '_';
  Module lib = (*Registry::Get("runtime.SystemLib"))(lib_prefix);
  reload_func_(lib, model_path, app_config_json);
}

- (void)resetChat {
  reset_chat_func_();
}

- (void)prefill:(NSString*)input {
  std::string prompt = input.UTF8String;
  if (first_input_after_image) {
    prefill_func_(prompt, true, (int)PlaceInPrompt::kEnd);
    first_input_after_image = false;
  } else {
    prefill_func_(prompt);
  }
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

- (void)processSystemPrompts {
  process_system_prompts_func_();
}

- (void)evaluate {
  LOG(INFO) << "Total-mem-budget=" << os_proc_available_memory() / (1 << 20) << "MB";
  llm_chat_->GetFunction("evaluate")();
  LOG(INFO) << "Left-mem-budget=" << os_proc_available_memory() / (1 << 20) << "MB";
}

- (void)unloadImageMod {
  image_mod_unload_func_();
  first_input_after_image = false;
}

- (void)reloadImageMod:(NSString*)modelLib modelPath:(NSString*)modelPath {
  std::string lib_prefix = modelLib.UTF8String;
  std::string model_path = modelPath.UTF8String;
  std::replace(lib_prefix.begin(), lib_prefix.end(), '-', '_');
  lib_prefix += '_';
  Module lib = (*Registry::Get("runtime.SystemLib"))(lib_prefix);
  image_mod_reload_func_(lib, model_path);
  first_input_after_image = false;
}

- (void)resetImageMod {
  image_mod_reset_func_();
  first_input_after_image = false;
}

- (NSString*)runtimeStatsTextImageMod {
  std::string ret = image_mod_runtime_stats_text_func_();
  return [NSString stringWithUTF8String:ret.c_str()];
}

- (void)prefillImage:(UIImage*)image prevPlaceholder:(NSString*)prevPlaceholder postPlaceholder:(NSString*)postPlaceholder {
  // prefill the previous placeholder string
  std::string prev_placeholder = prevPlaceholder.UTF8String;
  prefill_func_(prev_placeholder, false, (int)PlaceInPrompt::kBegin);

  // prefill with image embedding
  // step 1. get image rawdata: credit from https://stackoverflow.com/a/1262893
  CGImageRef imageRef = [image CGImage];
  NSUInteger width = CGImageGetWidth(imageRef);
  NSUInteger height = CGImageGetHeight(imageRef);
  CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
  unsigned char *rawData = (unsigned char*) calloc(height * width * 4, sizeof(unsigned char));
  NSUInteger bytesPerPixel = 4;
  NSUInteger bytesPerRow = bytesPerPixel * width;
  NSUInteger bitsPerComponent = 8;
  CGContextRef context = CGBitmapContextCreate(rawData, width, height,
                          bitsPerComponent, bytesPerRow, colorSpace,
                          kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
  CGColorSpaceRelease(colorSpace);
  CGContextDrawImage(context, CGRectMake(0, 0, width, height), imageRef);
  CGContextRelease(context);
  // step 2. create rgb array from rgba
  uint8_t *image_data = (uint8_t*) calloc(height * width * 3, sizeof(uint8_t));
  int count = 0;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      NSUInteger byteIndex = bytesPerRow * i + bytesPerPixel * j;
      image_data[count] = (uint8_t)rawData[byteIndex];
      image_data[count + 1] = (uint8_t)rawData[byteIndex + 1];
      image_data[count + 2] = (uint8_t)rawData[byteIndex + 2];
      count += 3;
    }
  }
  // step 2. create tvm NDArray
  ShapeTuple shape = {1, 224, 224, 3};
  DLDataType dtype = DataType::UInt(8);
  DLDevice device = DLDevice{kDLMetal, 0};
  size_t nbytes = size_t(dtype.bits / 8);
  for (auto s : shape) {
    nbytes *= (size_t)s;
  }
  NDArray input_image = NDArray::Empty(shape, dtype, device);
  input_image.CopyFromBytes(image_data, nbytes);
  // step 3. prefill with image embedding
  NDArray embedding = image_mod_embed_func_(input_image);
  prefill_with_embed_func_(embedding, false);
  // step 4. free memory
  free(image_data);
  free(rawData);

  // prefill the post placeholder string
  std::string post_placeholder = postPlaceholder.UTF8String;
  prefill_func_(post_placeholder, false, (int)PlaceInPrompt::kMiddle);

  // update the flag
  first_input_after_image = true;
}

@end
