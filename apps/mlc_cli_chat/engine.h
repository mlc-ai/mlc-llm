/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file engine.h
 */

#ifndef MLC_CLI_CHAT_ENGINE_H
#define MLC_CLI_CHAT_ENGINE_H

#include <json_ffi/json_ffi_engine.h>
#include <picojson.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <condition_variable>
#include <fstream>
#include <functional>
#include <memory>
#include <queue>
#include <thread>
#include <vector>

#include "base.h"

class EngineStateCli {
 public:
  std::queue<std::string> sync_queue;
  bool last_chunk_arrived = false;
  std::string output;
  std::string finish_reason;
  std::shared_ptr<std::mutex> queue_mutex;
  std::shared_ptr<std::condition_variable> queue_cv;
  double decode_tokens_per_s;
  double prefill_tokens_per_s;
  double prompt_tokens;
  double completion_tokens;

  EngineStateCli();
  std::function<void(const std::string&)> get_request_stream_callback();
  std::string handle_chat_completion(tvm::runtime::Module mod, const std::string& request_json,
                                     bool include_usage, const std::string& request_id);
  void getStats();
};

class Completions {
 public:
  std::shared_ptr<EngineStateCli> engine_state;
  tvm::runtime::Module __mod;

  Completions();

  Completions(std::shared_ptr<EngineStateCli> engine_state, tvm::runtime::Module mod);

  inline std::string GenerateUUID(size_t length);

  std::string create(std::vector<Message>& messages);
};

class Chat {
 public:
  Completions completions;

  Chat();

  Chat(std::shared_ptr<EngineStateCli> engine_state, tvm::runtime::Module mod);
};

class BackgroundLoops {
 private:
  // Default threads
  std::thread background_loop_thread;
  std::thread background_stream_back_loop_thread;
  bool terminated = false;
  tvm::runtime::Module __mod;

 public:
  // Default Constructor
  BackgroundLoops();

  // Parametrized constructor
  BackgroundLoops(tvm::runtime::Module mod);
  ~BackgroundLoops();

  void terminate();
};

class JSONFFIEngineWrapper {
 public:
  Chat chat;
  std::shared_ptr<EngineConfig> engine_config;
  tvm::runtime::Module mod;
  std::shared_ptr<EngineStateCli> engine_state;
  std::shared_ptr<BackgroundLoops> background_loops;

  JSONFFIEngineWrapper();

  JSONFFIEngineWrapper(std::string model_path, std::string model_lib_path, std::string mode,
                       std::string device, int device_id);
};

#endif
