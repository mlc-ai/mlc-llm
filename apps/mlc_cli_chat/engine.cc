/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file engine.cc
 */

#include "engine.h"

#include <iostream>

#include "base.h"

/// Helper function to get the json format of messages
std::string messagesToString(const std::vector<Message>& messages) {
  std::string result{""};
  for (size_t i = 0; i < messages.size(); ++i) {
    const auto& msg = messages[i];
    result += "{";

    bool firstItem = true;
    for (const auto& [key, value] : msg.content) {
      if (!firstItem) {
        result += ",";
      }
      result += "\"" + key + "\":";

      if (key == "role") {
        if (value == "user") {
          result += "\"" + value + "\"";
        } else if (value == "assistant") {
          result += "\"" + value + "\"";
        }
      } else if (key == "content") {
        if (i % 2 == 0) {
          result += "\"" + value + "\"";
        } else {
          result += value;
        }
      }
      firstItem = false;
    }
    result += "}";

    if (i != messages.size() - 1) {
      result += ",";
    }
  }
  return result;
}

// Helper function to print History
void printHistory(std::vector<Message> history) {
  for (int i = 0; i < history.size(); i++) {
    auto msg = history[i];
    std::cout << " content " << msg.content["content"];
  }
  std::cout << "\n";
}

EngineStateCli::EngineStateCli()
    : queue_cv(std::make_shared<std::condition_variable>()),
      queue_mutex(std::make_shared<std::mutex>()) {}

std::function<void(const std::string&)> EngineStateCli::get_request_stream_callback() {
  return [this](const std::string& response) -> void {
    {
      this->sync_queue.push(response);
      queue_cv->notify_one();
    }
  };
}

std::string EngineStateCli::handle_chat_completion(tvm::runtime::Module mod,
                                                const std::string& request_json, bool include_usage,
                                                const std::string& request_id) {
  // Clear the queue making sure that queue is empty
  // Not really required since this process should ideally make the queue empty
  {
    std::lock_guard<std::mutex> lock(*queue_mutex);
    std::queue<std::string> empty;
    std::swap(sync_queue, empty);
  }

  // TVM Global Function which generates the responses
  bool success = mod.GetFunction("chat_completion")(request_json, request_id);

  if (!success) {
    std::cerr << "Failed to start chat completion" << std::endl;
  }

  try {
    last_chunk_arrived = false;

    // Clear the ouput after every chat completion
    output = "";

    while (!last_chunk_arrived) {
      std::string json_str;
      std::unique_lock<std::mutex> lock(*queue_mutex);

      // Wait until the queue is not empty
      queue_cv->wait(lock, [this] { return !sync_queue.empty(); });
      std::string response = sync_queue.front();
      sync_queue.pop();

      picojson::value v;

      // Parse the JSON
      std::string err = picojson::parse(v, response);

      // Check for errors
      if (!err.empty()) {
        std::cerr << "JSON parsing error: " << err << std::endl;
      }

      // parsing successful, navigate through the array
      picojson::array& arr = v.get<picojson::array>();
      for (auto& item : arr) {
        picojson::object& obj = item.get<picojson::object>();

        // Extract 'delta' content if available
        if (obj.find("choices") != obj.end() && !obj["choices"].get<picojson::array>().empty()) {
          picojson::object& choices =
              obj["choices"].get<picojson::array>()[0].get<picojson::object>();

          if (!(choices["finish_reason"].is<picojson::null>())) {
            // Get the finish reason
            std::string finish_reason = choices["finish_reason"].get<std::string>();
            if (finish_reason == "length") {
              finish_reason = "length";
            }
          }
          if (choices.find("delta") != choices.end()) {
            picojson::object& delta = choices["delta"].get<picojson::object>();
            if (delta.find("content") != delta.end()) {
              std::string content = delta["content"].get<std::string>();

              std::cout << content << std::flush;
              output += content;
            }
          }
        }

        // Extract 'usage' details if available
        if (obj.find("usage") != obj.end()) {
          last_chunk_arrived = true;
          std::cout << std::endl;
          picojson::object& usage = obj["usage"].get<picojson::object>();

          // Access the 'usage' details
          double prompt_tokens = usage["prompt_tokens"].get<double>();
          double completion_tokens = usage["completion_tokens"].get<double>();
          double total_tokens = usage["total_tokens"].get<double>();

          // Access the 'extra' details
          picojson::object& extra = usage["extra"].get<picojson::object>();
          double prefill_tokens_per_s = extra["prefill_tokens_per_s"].get<double>();
          double decode_tokens_per_s = extra["decode_tokens_per_s"].get<double>();
          double end_to_end_latency_s = extra["end_to_end_latency_s"].get<double>();

          // fill the stats details
          this->decode_tokens_per_s = decode_tokens_per_s;
          this->prefill_tokens_per_s = prefill_tokens_per_s;
          this->prompt_tokens = prompt_tokens;
          this->completion_tokens = completion_tokens;
        }
      }
    }
  } catch (const std::exception& exception) {
    mod.GetFunction("abort")(request_id);
    throw;
  }
  return output;
}

void EngineStateCli::getStats() {
  std::cout << " decode : " << this->decode_tokens_per_s << " tok/sec (" << this->completion_tokens
            << " tokens in " << this->completion_tokens / this->decode_tokens_per_s << " sec)"
            << ", prefill : " << this->prefill_tokens_per_s << " tok/sec (" << this->prompt_tokens
            << " tokens in " << this->prompt_tokens / this->prefill_tokens_per_s << " sec)"
            << std::endl;
}

// Default Constructor
BackgroundLoops::BackgroundLoops() { terminated = false; }

// Parametrized constructor
BackgroundLoops::BackgroundLoops(tvm::runtime::Module mod) {
  this->__mod = mod;
  auto background_loop = mod.GetFunction("run_background_loop");
  auto background_stream_back_loop = mod.GetFunction("run_background_stream_back_loop");

  background_loop_thread = (std::thread)(background_loop);
  background_stream_back_loop_thread = (std::thread)(background_stream_back_loop);

  terminated = false;
}
BackgroundLoops::~BackgroundLoops() { terminate(); }

void BackgroundLoops::terminate() {
  if (!terminated) {
    terminated = true;

    try {
      __mod.GetFunction("exit_background_loop")();
    } catch (const std::exception& e) {
      std::cerr << "Error calling exit_background_loop: " << e.what() << std::endl;
    }

    if (background_loop_thread.joinable()) {
      background_loop_thread.join();
    }
    if (background_stream_back_loop_thread.joinable()) {
      background_stream_back_loop_thread.join();
    }
  }
}

// Default constructor
Completions::Completions() {}

Completions::Completions(std::shared_ptr<EngineStateCli> engine_state, tvm::runtime::Module mod) {
  this->engine_state = engine_state;
  this->__mod = mod;
}

// Method to generate a unique string for each process
inline std::string Completions::GenerateUUID(size_t length) {
  auto randchar = []() -> char {
    const char charset[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
    const size_t max_index = (sizeof(charset) - 1);
    return charset[rand() % max_index];
  };
  std::string str(length, 0);
  std::generate_n(str.begin(), length, randchar);
  return str;
}

std::string Completions::create(std::vector<Message>& messages) {
  std::string request_id{""};
  // Method to generate random string
  std::string generate_random_string{GenerateUUID(16)};

  // Unique ID for each chat completion process
  request_id = "chatcmpl-" + generate_random_string;

  std::string history_string{""};
  std::string left_braces{"{"};
  std::string right_braces{"}"};

  std::string prompt = messagesToString(messages);
  std::string jsonStart = R"({"messages":[)";
  std::string jsonEnd = "]}";

  std::string request_str = jsonStart + prompt + jsonEnd;

  std::string output_res =
      engine_state->handle_chat_completion(__mod, request_str, true, request_id);
  return output_res;
}

Chat::Chat() {}

Chat::Chat(std::shared_ptr<EngineStateCli> engine_state, tvm::runtime::Module mod) {
  this->completions = Completions(engine_state, mod);
}

// Device str to DLDevice map
DLDeviceType GetDevice(std::string device) {
  if ("cuda" == device) {
    return kDLCUDA;
  } else if ("cpu" == device || "llvm" == device) {
    return kDLCPU;
  } else if ("opencl" == device) {
    return kDLOpenCL;
  } else if ("vulkan" == device) {
    return kDLVulkan;
  } else if ("metal" == device) {
    return kDLMetal;
  } else {
    LOG(FATAL) << "Unsupported device :" << device;
  }
}

// Default constructor with No arguments
JSONFFIEngineWrapper::JSONFFIEngineWrapper() {}

JSONFFIEngineWrapper::JSONFFIEngineWrapper(std::string model_path, std::string model_lib_path,
                                           std::string mode, std::string device,
                                           int device_id = 0) {
  // Create an instance of EngineStateCli
  this->engine_state = std::make_shared<EngineStateCli>();

  auto engine = tvm::runtime::Registry::Get("mlc.json_ffi.CreateJSONFFIEngine");
  if (engine == nullptr) {
    std::cout << "\nError: Unable to access TVM global registry mlc.json_ffi.CreateJSONFFIEngine"
              << std::endl;
  }

  tvm::runtime::Module module_tvm = (*engine)();

  this->mod = module_tvm;

  // We can give mod as an argument to this
  background_loops = std::make_shared<BackgroundLoops>(mod);

  this->engine_config = std::make_shared<EngineConfig>(make_object<EngineConfigNode>());
  (*engine_config)->model = model_path;
  (*engine_config)->model_lib = model_lib_path;
  (*engine_config)->verbose = false;

  if (mode == "interactive") {
    (*engine_config)->mode = EngineMode::kInteractive;
  } else if (mode == "local") {
    (*engine_config)->mode = EngineMode::kLocal;
  } else if (mode == "server") {
    (*engine_config)->mode = EngineMode::kServer;
  }

  const std::string file_path = model_path + "/mlc-chat-config.json";
  std::ifstream file(file_path);
  if (!file.is_open()) {
    std::cerr << "Error: Unable to open " << file_path << std::endl;
    // return 1;
  }

  std::string config_content((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());

  // Parse the JSON object
  picojson::value config_object;
  std::string err;
  picojson::parse(config_object, config_content.begin(), config_content.end(), &err);
  if (!err.empty()) {
    std::cerr << "Error: Unable to parse the JSON object: " << err << std::endl;
  }

  // Accessing the parsed data
  if (config_object.is<picojson::object>()) {
    const picojson::object& model_config = config_object.get<picojson::object>();
    if (model_config.find("prefill_chunk_size") != model_config.end()) {
      double prefill_chunk_size = model_config.at("prefill_chunk_size").get<double>();
      (*engine_config)->prefill_chunk_size = prefill_chunk_size;
    } else {
      std::cerr << "Error: 'prefill_chunk_size' not found in the JSON object" << std::endl;
    }
  } else {
    std::cerr << "Error: Invalid JSON format" << std::endl;
  }

  auto call_back = engine_state->get_request_stream_callback();

  // Typecasting to the TVM Packed Function
  auto tvm_callback = tvm::runtime::TypedPackedFunc<void(std::string)>(call_back);

  // Call to Initialise Background Engine
  mod.GetFunction("init_background_engine")(static_cast<int>(GetDevice(device)), device_id,
                                            tvm_callback);

  std::string engine_config_json_str{(*engine_config)->AsJSONString()};

  // Call to Reload Function of JSONFFIEngineImpl
  mod.GetFunction("reload")(engine_config_json_str);

  chat = Chat(engine_state, mod);
}
