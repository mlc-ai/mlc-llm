/*!
 *  Copyright (c) 2023 by Contributors
 * \file cli_main.cc
 * \brief Implementation of a CLI version of chat
 */
// NOTE we only interact with the module through tvm runtime
// so there is no need to depend on a header interface
// the same set of operations can be implemented in other languages
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
                                                       "q4f16_1",  //
                                                       "q4f16_2",  //
                                                       "q4f32_0",  //
                                                       "q8f16_0",  //
                                                       "q0f16",    //
                                                       "q0f32"};

std::pair<std::string, int> DetectDevice(std::string device) {
  using tvm::runtime::DeviceAPI;

  std::string device_name;
  int device_id;
  int delimiter_pos = device.find(":");

  if (delimiter_pos == std::string::npos) {
    device_name = device;
    device_id = 0;
  } else {
    device_name = device.substr(0, delimiter_pos);
    device_id = std::stoi(device.substr(delimiter_pos + 1, device.length()));
  }

  if (device_name == "auto") {
    bool allow_missing = true;
    if (DeviceAPI::Get(DLDevice{kDLCUDA, 0}, allow_missing)) {
      return {"cuda", device_id};
    }
    if (DeviceAPI::Get(DLDevice{kDLMetal, 0}, allow_missing)) {
      return {"metal", device_id};
    }
    if (DeviceAPI::Get(DLDevice{kDLROCM, 0}, allow_missing)) {
      return {"rocm", device_id};
    }
    if (DeviceAPI::Get(DLDevice{kDLVulkan, 0}, allow_missing)) {
      return {"vulkan", device_id};
    }
    if (DeviceAPI::Get(DLDevice{kDLOpenCL, 0}, allow_missing)) {
      return {"opencl", device_id};
    }
    // TODO: Auto-detect devices for mali
    LOG(FATAL) << "Cannot auto detect device-name";
  }
  return {device_name, device_id};
}

DLDevice GetDevice(const std::string& device_name, int device_id) {
  if (device_name == "cuda") return DLDevice{kDLCUDA, device_id};
  if (device_name == "metal") return DLDevice{kDLMetal, device_id};
  if (device_name == "rocm") return DLDevice{kDLROCM, device_id};
  if (device_name == "vulkan") return DLDevice{kDLVulkan, device_id};
  if (device_name == "opencl" || device_name == "mali") return DLDevice{kDLOpenCL, device_id};
  LOG(FATAL) << "Invalid device name: " << device_name
             << ". Please enter the device in the form 'device_name:device_id'"
                " or 'device_name', where 'device_name' needs to be one of 'cuda', 'metal', "
                "'vulkan', 'rocm', 'opencl', 'auto'.";
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

void PrintSpecialCommands() {
  std::cout << "You can use the following special commands:\n"
            << "  /help               print the special commands\n"
            << "  /exit               quit the cli\n"
            << "  /stats              print out the latest stats (token/sec)\n"
            << "  /reset              restart a fresh chat\n"
            << "  /reload [model]  reload model `model` from disk, or reload the current "
               "model if `model` is not specified\n"
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

  static ModelPaths Find(const std::string& device_name, const std::string& local_id,
                         const std::string& user_lib_path);
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
    this->verbose_runtime_stats_text_ = this->chat_mod_->GetFunction("verbose_runtime_stats_text");
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
    ICHECK(verbose_runtime_stats_text_ != nullptr);
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
    }
    reload_(tvm::runtime::String(lib_path_), tvm::runtime::String(new_model_path));
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

  /*! \return A text describing the token statistics. */
  std::string VerboseRuntimeStatsText() { return verbose_runtime_stats_text_(); }

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
  tvm::runtime::PackedFunc verbose_runtime_stats_text_;
  tvm::runtime::PackedFunc reset_chat_;
  tvm::runtime::PackedFunc process_system_prompts_;

  std::string lib_path_;
  tvm::runtime::Module executable_;
};

std::optional<std::filesystem::path> TryInferMLCChatConfig(const std::string& local_id) {
  return FindFile(
      {
          local_id,                              // full path, or just the name
          "dist/prebuilt/" + local_id,           // Using prebuilt workflow
          "dist/" + local_id + "/params",        // Default directory after mlc_llm.build_model()
          "dist/prebuilt/mlc-chat-" + local_id,  // Also prebuilt workflow, but missed prefix
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

ModelPaths ModelPaths::Find(const std::string& device_name, const std::string& local_id,
                            const std::string& user_lib_path) {
  // Step 1. Find config path
  std::filesystem::path config_path;
  if (auto path = TryInferMLCChatConfig(local_id)) {
    config_path = path.value();
  } else {
    LOG(FATAL)
        << "The model folder provided does not seem to refer to a valid mlc-llm model folder. "
           "Specifically, we cannot find `mlc-chat-config.json`, a required file. You should "
           "provide a path that contains the file. "
           "According to your input `"
        << local_id << "`, we looked at folder(s):\n"
        << "- " + local_id << "\n"
        << "- dist/prebuilt/" + local_id << "\n"
        << "- dist/" + local_id + "/params"
        << "\n"
        << "- dist/prebuilt/mlc-chat-" + local_id;
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
  std::filesystem::path lib_path;
  if (!user_lib_path.empty()) {
    lib_path = user_lib_path;
    if (!std::filesystem::exists(lib_path) || !std::filesystem::is_regular_file(lib_path)) {
      LOG(FATAL) << "The `lib_path` you passed in is not a file: " << user_lib_path << "\n";
      exit(1);
    }
  } else {
    std::string lib_local_id = ReadStringFromJSONFile(config_path, "model_lib");
    std::string lib_name = lib_local_id + "-" + device_name;
    if (auto path = FindFile({lib_local_id,
                              "dist/prebuilt/lib",  // Using prebuilt workflow
                              "dist/" + local_id, "dist/prebuilt/" + lib_local_id},
                             {
                                 lib_name + GetArchSuffix(),
                                 lib_name,
                             },
                             GetLibSuffixes())) {
      lib_path = path.value();
    } else {
      LOG(FATAL) << "Cannot find the model library that corresponds to `" << lib_local_id << "`.\n"
                 << "We searched over the following possible paths: \n"
                 << "- " + lib_local_id << "\n"
                 << "- dist/prebuilt/lib \n"
                 << "- dist/" + local_id << "\n"
                 << "- dist/prebuilt/" + lib_local_id << "\n"
                 << "If you would like to directly specify the full model library path, you may "
                 << "consider passing in the `--model-lib-path` argument.\n";
      exit(1);
    }
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
  os << chat->GetRole1() << ": " << std::flush;
  for (size_t i = 0; !chat->Stopped(); ++i) {
    chat->Decode();
    if (i % stream_interval == 0 || chat->Stopped()) {
      std::string new_msg = chat->GetMessage();
      std::string print = mlc::llm::GetDeltaMessage(cur_msg, new_msg);
      os << print << std::flush;
      cur_msg = std::move(new_msg);
    }
  }
  os << std::endl << std::flush;
}

/*!
 * \brief Start a chat conversation.
 *
 * \param chat The chat module.
 * \param device_name The device that the model should run on.
 * \param local_id The model path which contains the model config, tokenizer and parameters.
 * \param stream_interval The interval that should be used for streaming the response.
 */
void Chat(ChatModule* chat, const std::string& device_name, std::string local_id,
          std::string lib_path, int stream_interval = 2) {
  ModelPaths model = ModelPaths::Find(device_name, local_id, lib_path);
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
      model = ModelPaths::Find(device_name, new_local_id, lib_path);
      chat->Reload(model);
      local_id = new_local_id;
    } else if (input.substr(0, 5) == "/help") {
      PrintSpecialCommands();
    } else {
      Converse(chat, input, stream_interval, std::cout);
    }
  }
}

int main(int argc, char* argv[]) {
  argparse::ArgumentParser args("mlc_chat_cli");

  args.add_description(
      "MLCChat CLI is the command line tool to run MLC-compiled LLMs out of the box.\n"
      "Note: the --model argument is required. It can either be the model name with its "
      "quantization scheme or a full path to the model folder. In the former case, the "
      "provided name will be used to search for the model folder over possible paths. "
      "--model-lib-path argument is optional. If unspecified, the --model argument will be used "
      "to search for the library file over possible paths.");

  args.add_argument("--model").help("[required] the model to use");
  args.add_argument("--model-lib-path")
      .help("[optional] the full path to the model library file to use");
  args.add_argument("--device").default_value("auto");
  args.add_argument("--evaluate").default_value(false).implicit_value(true);
  args.add_argument("--eval-prompt-len").default_value(128).scan<'i', int>();
  args.add_argument("--eval-gen-len").default_value(1024).scan<'i', int>();

  try {
    args.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << args << std::endl;
    return 1;
  }

  std::string local_id = args.get<std::string>("--model");
  std::string lib_path;
  if (args.present("--model-lib-path")) {
    lib_path = args.get<std::string>("--model-lib-path");
  }
  auto [device_name, device_id] = DetectDevice(args.get<std::string>("--device"));

  try {
    ChatModule chat(GetDevice(device_name, device_id));
    if (args.get<bool>("--evaluate")) {
      // `--evaluate` is only used for performance debugging, and thus will call low-level APIs
      // that are not supposed to be used in chat app setting
      int prompt_len = args.get<int>("--eval-prompt-len");
      int gen_len = args.get<int>("--eval-gen-len");
      ModelPaths model = ModelPaths::Find(device_name, local_id, lib_path);
      tvm::runtime::Module chat_mod = mlc::llm::CreateChatModule(GetDevice(device_name, device_id));
      std::string model_path = model.config.parent_path().string();
      tvm::runtime::Module lib = tvm::runtime::Module::LoadFromFile(model.lib.string());
      chat_mod.GetFunction("reload")(lib, tvm::String(model_path));
      chat_mod.GetFunction("evaluate")(prompt_len, gen_len);
    } else {
      Chat(&chat, device_name, local_id, lib_path);
    }
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    return 1;
  }
  return 0;
}
