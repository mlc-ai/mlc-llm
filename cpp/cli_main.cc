/*!
 *  Copyright (c) 2023 by Contributors
 * \file cli_main.cc
 * \brief Implementation of a CLI version of chat
 */
// NOTE we only interact with the module through tvm runtime
// so there is no need to depend on a header interface
// the same set of operations can be implemented in other languages
#define PICOJSON_USE_INT64
#define __STDC_FORMAT_MACROS

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <argparse/argparse.hpp>
#include <bitset>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "llm_chat.h"
#include "picojson.h"

const std::vector<std::string> quantization_presets = {"q3f16_0",  //
                                                       "q4f16_0",  //
                                                       "q4f32_0",  //
                                                       "q0f32",    //
                                                       "q0f16"};

std::string DetectDeviceName(std::string device_name) {
  using tvm::runtime::DeviceAPI;
  if (device_name == "auto") {
    bool allow_missing = true;
    if (DeviceAPI::Get(DLDevice{kDLCUDA, 0}, allow_missing)) {
      return "cuda";
    }
    if (DeviceAPI::Get(DLDevice{kDLMetal, 0}, allow_missing)) {
      return "metal";
    }
    if (DeviceAPI::Get(DLDevice{kDLVulkan, 0}, allow_missing)) {
      return "vulkan";
    }
    if (DeviceAPI::Get(DLDevice{kDLOpenCL, 0}, allow_missing)) {
      return "opencl";
    }
    LOG(FATAL) << "Cannot auto detect device-name";
  }
  return device_name;
}

DLDevice GetDevice(const std::string& device_name, int device_id) {
  if (device_name == "cuda") return DLDevice{kDLCUDA, device_id};
  if (device_name == "metal") return DLDevice{kDLMetal, device_id};
  if (device_name == "vulkan") return DLDevice{kDLVulkan, device_id};
  if (device_name == "opencl") return DLDevice{kDLOpenCL, device_id};
  LOG(FATAL) << "Do not recognize device name " << device_name;
  return DLDevice{kDLCPU, 0};
}

/*!
 * \brief Search for file path return the first result.
 *
 * \param search_paths The paths to search for the file.
 * \param names The names of to look for.
 * \param suffixes The suffix to look for.
 */
std::optional<std::filesystem::path> FindFile(
    const std::vector<std::filesystem::path>& search_paths,  //
    const std::vector<std::string>& names,                   //
    const std::vector<std::string>& suffixes) {
  for (const std::filesystem::path& prefix : search_paths) {
    for (const std::string& name : names) {
      for (const std::string& suffix : suffixes) {
        try {
          std::filesystem::path path = std::filesystem::canonical(prefix / (name + suffix));
          if (std::filesystem::exists(path) && std::filesystem::is_regular_file(path)) {
            return path;
          }
        } catch (const std::filesystem::filesystem_error& e) {
        }
      }
    }
  }
  return std::nullopt;
}

/**
 * get default lib suffixes
 */
std::vector<std::string> GetLibSuffixes() {
#if defined(WIN32)
  return {".dll"};
#elif defined(__APPLE__)
  return {".dylib", ".so"};
#else
  return {".so"};
#endif
}

std::string GetArchSuffix() {
#if defined(__x86_64__)
  return "_x86_64";
#elif defined(__aarch64__)
  return "_arm64";
#endif
  return "";
}

std::vector<std::string> CountUTF8(const std::string& s) {
  // assume that the string is always valid utf8
  std::vector<std::string> ret;
  for (size_t pos = 0; pos < s.size();) {
    if ((s[pos] & 0x80) == 0x00) {
      ret.push_back(s.substr(pos, 1));
      pos += 1;
    } else if (pos + 1 < s.size() && (s[pos] & 0xE0) == 0xC0 && (s[pos + 1] & 0xC0) == 0x80) {
      ret.push_back(s.substr(pos, 2));
      pos += 2;
    } else if (pos + 1 < s.size() && (s[pos] & 0xF0) == 0xE0 && (s[pos + 1] & 0xC0) == 0x80 &&
               (s[pos + 2] & 0xC0) == 0x80) {
      ret.push_back(s.substr(pos, 3));
      pos += 3;
    } else if (pos + 2 < s.size() && (s[pos] & 0xF8) == 0xF0 && (s[pos + 1] & 0xC0) == 0x80 &&
               (s[pos + 2] & 0xC0) == 0x80 && (s[pos + 3] & 0xC0) == 0x80) {
      ret.push_back(s.substr(pos, 4));
      pos += 4;
    } else {
      LOG(FATAL) << "Invalid UTF8 string";
    }
  }
  return std::move(ret);
}

void PrintSpecialCommands() {
  std::cout << "You can use the following special commands:\n"
            << "  /help               print the special commands\n"
            << "  /exit               quit the cli\n"
            << "  /stats              print out the latest stats (token/sec)\n"
            << "  /reset              restart a fresh chat\n"
            << "  /reload [local_id]  reload model `local_id` from disk, or reload the current "
               "model if `local_id` is not specified\n"
            << std::endl
            << std::flush;
}

struct ModelPaths {
  /*!
   * \brief Path to mlc-llm-config.json
   */
  std::filesystem::path config;
  /*!
   * \brief Path to ndarray-cache.json
   */
  std::filesystem::path params;
  /*!
   * \brief Path to ${model}-${device}.{so|dylib}
   *
   * This dynamic library contains all the compute kernels used in LLM inference, and can be
   * loaded using tvm::runtime::Module::LoadFromFile.
   */
  std::filesystem::path lib;

  static ModelPaths Find(const std::filesystem::path& artifact_path, const std::string& device_name,
                         const std::string& local_id);
};

/*!
 * \brief Helper class to implement chat features.
 *
 * A common streaming chat flow can be implemented as follows:
 *
 * \code
 *
 * void SingleRound(const std::string& input) {
 *   // prefill and decode first token for given input
 *   chat->Prefill(input);
 *   // check if the current round stops
 *   while (!chat->Stopped()) {
 *     // get the latest message and display it
 *     RefreshCurrentReply(chat->GetMessage());
 *     // decode the next token
 *     chat->Decode();
 *   }
 * }
 *
 * \endcode
 *
 * \note GetMessage function will return the complete latest message.
 *       This is useful in most UIs that directly replaces the entire
 *       textbox content.
 *
 *       Implementation detail: this class is a thin wrapper of TVM runtime
 *       API that can also be exposed in other language runtimes.
 *       Look for the name ChatModule in other apps(android, iOS) and you will
 *       find functions with similar names.
 */
class ChatModule {
 public:
  /*!
   * \brief Constructor
   * \param device the device to run the chat on.
   */
  explicit ChatModule(const DLDevice& device) {
    this->chat_mod_ = mlc::llm::CreateChatModule(device);
    this->prefill_ = this->chat_mod_->GetFunction("prefill");
    this->decode_ = this->chat_mod_->GetFunction("decode");
    this->stopped_ = this->chat_mod_->GetFunction("stopped");
    this->get_message_ = this->chat_mod_->GetFunction("get_message");
    this->reload_ = this->chat_mod_->GetFunction("reload");
    this->get_role0_ = this->chat_mod_->GetFunction("get_role0");
    this->get_role1_ = this->chat_mod_->GetFunction("get_role1");
    this->runtime_stats_text_ = this->chat_mod_->GetFunction("runtime_stats_text");
    this->reset_chat_ = this->chat_mod_->GetFunction("reset_chat");
    this->process_system_prompts_ = this->chat_mod_->GetFunction("process_system_prompts");
    this->lib_path_ = "";
    this->executable_ = tvm::runtime::Module(nullptr);
    ICHECK(prefill_ != nullptr);
    ICHECK(decode_ != nullptr);
    ICHECK(stopped_ != nullptr);
    ICHECK(get_message_ != nullptr);
    ICHECK(reload_ != nullptr);
    ICHECK(get_role0_ != nullptr);
    ICHECK(get_role1_ != nullptr);
    ICHECK(runtime_stats_text_ != nullptr);
    ICHECK(reset_chat_ != nullptr);
  }
  /*!
   * \brief Reload the module to a new model path.
   * \param model The model path spec.
   */
  void Reload(const ModelPaths& model) {
    std::cout << "Loading model..." << std::endl;
    std::string new_lib_path = model.lib.string();
    std::string new_model_path = model.config.parent_path().string();
    if (this->lib_path_ != new_lib_path) {
      this->lib_path_ = new_lib_path;
      this->executable_ = tvm::runtime::Module::LoadFromFile(this->lib_path_);
    }
    reload_(this->executable_, tvm::runtime::String(new_model_path));
    std::cout << "Loading finished" << std::endl << std::flush;
  }

  /*!
   * \brief Reset the current chat session.
   * \note The model remains the same, to change model, call Reload.
   */
  void ResetChat() { reset_chat_(); }

  /*! \brief Process system prompts before starting conversation. */
  void ProcessSystemPrompts() {
    std::cout << "Running system prompts..." << std::endl << std::flush;
    process_system_prompts_();
    std::cout << "System prompts finished" << std::endl << std::flush;
  }

  /*! \return Role0(user) name in the chat template. */
  std::string GetRole0() { return get_role0_(); }

  /*! \return Role1(bot) name in the chat template. */
  std::string GetRole1() { return get_role1_(); }

  /*! \return A text describing the runtime statistics. */
  std::string RuntimeStatsText() { return runtime_stats_text_(); }

  /*!
   * \brief Run prefill stage for a given input and decode the first output token.
   * \param input the user input.
   */
  void Prefill(const std::string& input) { prefill_(input); }

  /*!
   * \brief Run one decode step to decode the next token.
   */
  void Decode() { decode_(); }

  /*! \return Whether the current round stopped. */
  bool Stopped() { return stopped_(); }

  /*!
   * \return Get the output message in the current round.
   * \note This function returns the message that corresponds to
   *       all the tokens decoded so far.
   */
  std::string GetMessage() { return get_message_(); }

 protected:
  // TVM Modules and functions with TVM's calling convention
  tvm::runtime::Module chat_mod_;
  tvm::runtime::PackedFunc prefill_;
  tvm::runtime::PackedFunc decode_;
  tvm::runtime::PackedFunc stopped_;
  tvm::runtime::PackedFunc get_message_;
  tvm::runtime::PackedFunc reload_;
  tvm::runtime::PackedFunc get_role0_;
  tvm::runtime::PackedFunc get_role1_;
  tvm::runtime::PackedFunc runtime_stats_text_;
  tvm::runtime::PackedFunc reset_chat_;
  tvm::runtime::PackedFunc process_system_prompts_;

  std::string lib_path_;
  tvm::runtime::Module executable_;
};

std::optional<std::filesystem::path> TryInferMLCChatConfig(
    const std::filesystem::path& artifact_path, const std::string& local_id) {
  return FindFile(
      {
          artifact_path / local_id / "params",
          artifact_path / "prebuilt" / local_id,
          artifact_path / "prebuilt" / ("mlc-chat-" + local_id),
      },
      {"mlc-chat-config"}, {".json"});
}

std::string ReadStringFromJSONFile(const std::filesystem::path& config_path,
                                   const std::string& key) {
  std::string config_json_str;
  {
    std::ifstream config_istream(config_path.string());
    ICHECK(config_istream);
    std::ostringstream config_ostream;
    config_ostream << config_istream.rdbuf();
    config_json_str = config_ostream.str();
  }
  // Parse MLC's config json to figure out where the model lib is
  picojson::value config_info;
  picojson::parse(config_info, config_json_str);
  auto config = config_info.get<picojson::object>();
  ICHECK(config[key].is<std::string>());
  return config[key].get<std::string>();
}

ModelPaths ModelPaths::Find(const std::filesystem::path& artifact_path,
                            const std::string& device_name, const std::string& local_id) {
  // Step 1. Find config path
  std::filesystem::path config_path;
  if (auto path = TryInferMLCChatConfig(artifact_path, local_id)) {
    config_path = path.value();
  } else {
    std::cerr << "Cannot find \"mlc-chat-config.json\" in path \"" << artifact_path << "/"
              << local_id;
    exit(1);
  }
  std::cout << "Use MLC config: " << config_path << std::endl;
  // Step 2. Find parameters
  std::filesystem::path params_json;
  if (auto path = FindFile({config_path.parent_path().string()}, {"ndarray-cache"}, {".json"})) {
    params_json = path.value();
  } else {
    std::cerr << "Cannot find \"ndarray-cache.json\" for params: " << config_path.parent_path()
              << std::endl;
    exit(1);
  }
  std::cout << "Use model weights: " << params_json << std::endl;
  // Step 3. Find model lib path
  std::string lib_local_id = ReadStringFromJSONFile(config_path, "model_lib");
  std::string lib_name = lib_local_id + "-" + device_name;
  std::filesystem::path lib_path;
  if (auto path = FindFile(
          {
              artifact_path / lib_local_id,              // Usually this is the candidate
              artifact_path / "prebuilt" / "lib",        // prebuild lib
              artifact_path / "prebuilt" / lib_local_id  // For prebuilts
          },
          {
              lib_name + GetArchSuffix(),
              lib_name,
          },
          GetLibSuffixes())) {
    lib_path = path.value();
  } else {
    std::cerr << "Cannot find library \"" << lib_name << GetLibSuffixes().back() << "\" in "
              << artifact_path << "/prebuilt/lib or other search paths" << std::endl;
    exit(1);
  }
  std::cout << "Use model library: " << lib_path << std::endl;
  return ModelPaths{config_path, params_json, lib_path};
}

/*!
 * \brief Implementation of one round chat.
 * \param chat The chat module.
 * \param input The input prompt.
 * \param stream_interval Refresh rate
 * \param os output stream
 */
void Converse(ChatModule* chat, const std::string& input, int stream_interval,
              std::ostream& os) {  // NOLINT(*)
  chat->Prefill(input);

  std::string cur_msg = "";
  std::vector<std::string> cur_utf8_chars = CountUTF8(cur_msg);

  os << chat->GetRole1() << ": " << std::flush;
  for (size_t i = 0; !chat->Stopped(); ++i) {
    chat->Decode();
    if (i % stream_interval == 0 || chat->Stopped()) {
      std::string new_msg = chat->GetMessage();
      // NOTE: display the new message.
      // The main complication here is that new_msg can be different
      // from previous message, so we need to find the diff,
      // delete previous messages that are different, then print it out.
      // This logic is only needed for simple stdout.
      //
      // For UI apps that can directly update output text
      // we can simply do last_reply.text = chat->GetMessage();
      std::vector<std::string> new_utf8_chars = CountUTF8(new_msg);
      // Step 1. Find the index of the first UTF8 char that differs
      size_t pos = std::mismatch(cur_utf8_chars.begin(), cur_utf8_chars.end(),
                                 new_utf8_chars.begin(), new_utf8_chars.end())
                       .first -
                   cur_utf8_chars.begin();
      // Step 2. Delete the previous message since `pos`
      std::string print = "";
      for (size_t j = pos; j < cur_utf8_chars.size(); ++j) {
        print += "\b \b";
      }
      // Step 3. Print the new message since `pos`
      for (size_t j = pos; j < new_utf8_chars.size(); ++j) {
        print += new_utf8_chars[j];
      }
      os << print << std::flush;
      cur_msg = std::move(new_msg);
      cur_utf8_chars = std::move(new_utf8_chars);
    }
  }
  os << std::endl << std::flush;
}

/*!
 * \brief Start a chat conversation.
 *
 * \param chat The chat module.
 * \param executable The model library to initialize the chat module.
 * \param model_path The model path with contains the model config, tokenizer and parameters.
 */
void Chat(ChatModule* chat, const std::filesystem::path& artifact_path,
          const std::string& device_name, std::string local_id, int stream_interval = 2) {
  ModelPaths model = ModelPaths::Find(artifact_path, device_name, local_id);
  PrintSpecialCommands();
  chat->Reload(model);
  chat->ProcessSystemPrompts();
  while (true) {
    std::string input;
    std::cout << chat->GetRole0() << ": " << std::flush;
    std::getline(std::cin, input);
    if (!std::cin.good()) {
      break;
    } else if (input.substr(0, 6) == "/reset") {
      chat->ResetChat();
      std::cout << "RESET CHAT SUCCESS" << std::endl << std::flush;
      chat->ProcessSystemPrompts();
    } else if (input.substr(0, 5) == "/exit") {
      break;
    } else if (input.substr(0, 6) == "/stats") {
      std::cout << chat->RuntimeStatsText() << std::endl << std::flush;
    } else if (input.substr(0, 7) == "/reload") {
      std::string new_local_id;
      {
        std::string reload_prompt;
        std::istringstream is(input);
        is >> reload_prompt >> new_local_id;
      }
      if (new_local_id.empty()) {
        new_local_id = local_id;
      }
      model = ModelPaths::Find(artifact_path, device_name, new_local_id);
      chat->Reload(model);
      local_id = new_local_id;
    } else if (input.substr(0, 5) == "/help") {
      PrintSpecialCommands();
    } else {
      Converse(chat, input, stream_interval, std::cout);
    }
  }
}

std::string GuessLocalId(const std::filesystem::path& artifact_path, const std::string& model,
                         const std::string& quantization) {
  std::vector<std::string> local_id_candidates;
  std::vector<std::string> quantization_candidates =
      (quantization == "auto") ? quantization_presets : std::vector<std::string>{quantization};
  for (std::string quantization_candidate : quantization_candidates) {
    local_id_candidates.push_back(model + "-" + quantization_candidate);
  }
  for (const std::string& guess_local_id : local_id_candidates) {
    if (std::optional<std::filesystem::path> config_path =
            TryInferMLCChatConfig(artifact_path, guess_local_id)) {
      return guess_local_id;
    }
  }
  std::cerr << "Cannot find \"mlc-chat-config.json\" in path \"" << artifact_path << "/"
            << local_id_candidates[0] << "/params/\", \"" << artifact_path
            << "/prebuilt/" + local_id_candidates[0] << "\" or other candidate paths.";
  exit(1);
}

int main(int argc, char* argv[]) {
  argparse::ArgumentParser args("mlc_chat");

  args.add_argument("--local-id").default_value("");
  args.add_argument("--model").default_value("vicuna-v1-7b");
  args.add_argument("--quantization").default_value("auto");
  args.add_argument("--device-name").default_value("auto");
  args.add_argument("--device_id").default_value(0).scan<'i', int>();
  args.add_argument("--artifact-path").default_value("dist");
  args.add_argument("--evaluate").default_value(false).implicit_value(true);

  try {
    args.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << args << std::endl;
    return 1;
  }

  std::string local_id = args.get<std::string>("--local-id");
  std::string model = args.get<std::string>("--model");
  std::string quantization = args.get<std::string>("--quantization");
  std::string device_name = DetectDeviceName(args.get<std::string>("--device-name"));
  int device_id = args.get<int>("--device_id");
  std::string artifact_path = args.get<std::string>("--artifact-path");

  if (local_id.empty()) {
    local_id = GuessLocalId(artifact_path, model, quantization);
  }

  try {
    ChatModule chat(GetDevice(device_name, device_id));
    if (args.get<bool>("--evaluate")) {
      // `--evaluate` is only used for performance debugging, and thus will call low-level APIs
      // that are not supposed to be used in chat app setting
      ModelPaths model = ModelPaths::Find(artifact_path, device_name, local_id);
      tvm::runtime::Module chat_mod = mlc::llm::CreateChatModule(GetDevice(device_name, device_id));
      std::string model_path = model.config.parent_path().string();
      tvm::runtime::Module lib = tvm::runtime::Module::LoadFromFile(model.lib.string());
      chat_mod.GetFunction("reload")(lib, tvm::String(model_path));
      chat_mod.GetFunction("evaluate")();
    } else {
      Chat(&chat, artifact_path, device_name, local_id);
    }
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    return 1;
  }
  return 0;
}
