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
#include <optional>
#include <string>
#include <vector>

#include "llm_chat.h"

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
std::optional<std::filesystem::path> FindFile(const std::vector<std::string>& search_paths,
                                              const std::vector<std::string>& names,
                                              const std::vector<std::string>& suffixes) {
  std::vector<std::filesystem::path> ret;
  for (auto prefix : search_paths) {
    for (auto name : names) {
      for (auto suffix : suffixes) {
        std::filesystem::path path(prefix + "/" + name + suffix);
        if (std::filesystem::exists(path)) {
          path = std::filesystem::canonical(path);
          if (std::filesystem::is_regular_file(path)) {
            return path;
          }
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
            << "  /help    print the special commands\n"
            << "  /exit    quit the cli\n"
            << "  /stats   print out the latest stats (token/sec)\n"
            << "  /reset   restart a fresh chat\n"
            << std::endl
            << std::flush;
}

/*!
 * \brief Start a chat conversation.
 *
 * \param chat_mod The chat module.
 * \param temperature The temperature to use for sampling.
 * \param top_p The top_p to use for sampling.
 */
void Chat(tvm::runtime::Module chat_mod, double temperature = 0.7, double top_p = 0.95,
          int64_t stream_interval = 2, int max_window_size = 768, int mean_gen_len = 128,
          double shift_fill_factor = 0.3) {
  // initialize chat context
  chat_mod.GetFunction("init_chat")(temperature, top_p, stream_interval, mean_gen_len,
                                    shift_fill_factor);
  auto f_stop = chat_mod.GetFunction("stopped");
  auto f_encode = chat_mod.GetFunction("encode");
  auto f_decode = chat_mod.GetFunction("decode");
  auto f_stats = chat_mod.GetFunction("runtime_stats_text");
  std::string role0 = chat_mod.GetFunction("get_role0")();
  std::string role1 = chat_mod.GetFunction("get_role1")();

  while (true) {
    std::string inp;
    std::cout << role0 << ": " << std::flush;
    std::getline(std::cin, inp);
    if (!std::cin.good()) break;
    if (inp.substr(0, 6) == "/reset") {
      // initialize chat context
      chat_mod.GetFunction("reset_chat")();
      std::cout << "RESET CHAT SUCCESS" << std::endl << std::flush;
      continue;
    } else if (inp.substr(0, 5) == "/exit") {
      break;
    } else if (inp.substr(0, 6) == "/stats") {
      std::string stats_text = f_stats();
      std::cout << stats_text << std::endl << std::flush;
      continue;
    } else if (inp.substr(0, 5) == "/help") {
      PrintSpecialCommands();
      continue;
    }

    std::string prev_printed = "";
    auto printed_UTF8_chars = CountUTF8(prev_printed);

    std::cout << role1 << ": " << std::flush;
    f_encode(inp);
    for (size_t i = 0; !f_stop(); ++i) {
      f_decode();
      if (i % stream_interval == 0 || f_stop()) {
        std::string cur_msg = chat_mod.GetFunction("get_message")();
        // delete the previous message, and print the current message
        std::string print = "";

        auto cur_UTF8_chars = CountUTF8(cur_msg);

        size_t i = 0;
        for (; i < std::min(printed_UTF8_chars.size(), cur_UTF8_chars.size()); ++i) {
          if (printed_UTF8_chars[i] != cur_UTF8_chars[i]) {
            break;
          }
        }
        for (size_t j = i; j < printed_UTF8_chars.size(); ++j) {
          print += "\b \b";
        }
        for (size_t j = i; j < cur_UTF8_chars.size(); ++j) {
          print += cur_UTF8_chars[j];
        }
        std::cout << print << std::flush;

        prev_printed = cur_msg;
        printed_UTF8_chars = std::move(cur_UTF8_chars);
      }
    }

    std::cout << std::endl << std::flush;
  }
}

int main(int argc, char* argv[]) {
  using namespace tvm::runtime;
  argparse::ArgumentParser args("mlc_chat");

  args.add_argument("--local-id").default_value("");
  args.add_argument("--model").default_value("vicuna-v1-7b");
  args.add_argument("--quantization").default_value("auto");
  args.add_argument("--device-name").default_value("auto");
  args.add_argument("--device_id").default_value(0).scan<'i', int>();
  args.add_argument("--artifact-path").default_value("dist");
  args.add_argument("--params").default_value("auto");
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
  DLDevice device = GetDevice(device_name, device_id);
  std::string artifact_path = args.get<std::string>("--artifact-path");
  std::string params = args.get<std::string>("--params");

  std::string arch_suffix = GetArchSuffix();

  std::vector<std::string> local_id_candidates;
  std::optional<std::filesystem::path> config_path_opt;

  // Configure local id candidates.
  if (local_id != "") {
    local_id_candidates = {local_id};
  } else {
    std::vector<std::string> quantization_candidates;
    if (quantization == "auto") {
      quantization_candidates = quantization_presets;
    } else {
      quantization_candidates = {quantization};
    }
    for (std::string quantization_candidate : quantization_candidates) {
      local_id_candidates.push_back(model + "-" + quantization_candidate);
    }
  }

  // Search for mlc-chat-config.json.
  for (auto local_id_candidate : local_id_candidates) {
    std::vector<std::string> config_search_paths = {
        artifact_path + "/" + local_id_candidate + "/params",  //
        artifact_path + "/prebuilt/" + local_id_candidate};
    config_path_opt = FindFile(config_search_paths, {"mlc-chat-config"}, {".json"});
    if (config_path_opt) {
      local_id = local_id_candidate;
      break;
    }
  }
  if (!config_path_opt) {
    std::cerr << "Cannot find \"mlc-chat-config.json\" in path \"" << artifact_path << "/"
              << local_id_candidates[0] << "/params/\", \"" << artifact_path
              << "/prebuilt/" + local_id_candidates[0] << "\" or other candidate paths.";
    return 1;
  }
  std::cout << "Use config " << config_path_opt.value().string() << std::endl;
  std::filesystem::path model_path = config_path_opt.value().parent_path();

  // Locate the library.
  std::string lib_name = local_id + "-" + device_name;
  std::string lib_dir_path;
  if (model_path.string().compare(model_path.string().length() - 7, 7, "/params") == 0) {
    lib_dir_path = model_path.parent_path().string();
  } else {
    lib_dir_path = model_path.parent_path().string() + "/lib";
  }
  std::optional<std::filesystem::path> lib_path_opt =
      FindFile({lib_dir_path}, {lib_name, lib_name + arch_suffix}, GetLibSuffixes());
  if (!lib_path_opt) {
    std::cerr << "Cannot find library \"" << lib_name << GetLibSuffixes().back()
              << "\" and other library candidate in " << lib_dir_path << std::endl;
    return 1;
  }
  std::cout << "Use lib " << lib_path_opt.value().string() << std::endl;

  // Locate the params.
  if (params == "auto") {
    auto params_json_opt = FindFile({model_path}, {"ndarray-cache"}, {".json"});
    if (!params_json_opt) {
      std::cerr << "Cannot find ndarray-cache.json for params in " << model_path << std::endl;
      return 1;
    }
    params = params_json_opt.value().parent_path().string();
  } else if (!FindFile({params}, {"ndarray-cache"}, {".json"})) {
    std::cerr << "Cannot find ndarray-cache.json for params in " << params << std::endl;
    return 1;
  }

  try {
    auto lib = Module::LoadFromFile(lib_path_opt.value().string());
    std::cout << "Initializing the chat module..." << std::endl;
    Module chat_mod = mlc::llm::CreateChatModule(lib, model_path.string(), params, device);

    std::cout << "Finish loading" << std::endl;
    PrintSpecialCommands();

    if (args.get<bool>("--evaluate")) {
      chat_mod.GetFunction("evaluate")();
    } else {
      Chat(chat_mod);
    }
  } catch (const std::runtime_error& err) {
    // catch exception so error message
    // get reported here without silently quit.
    std::cerr << err.what() << std::endl;
    return 1;
  }
  return 0;
}
