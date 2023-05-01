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

#include "llm_chat.h"

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
    LOG(FATAL) << "Cannot auto detect device-name";
  }
  return device_name;
}

DLDevice GetDevice(const std::string& device_name) {
  if (device_name == "cuda") return DLDevice{kDLCUDA, 0};
  if (device_name == "metal") return DLDevice{kDLMetal, 0};
  if (device_name == "vulkan") return DLDevice{kDLVulkan, 0};
  if (device_name == "opencl") return DLDevice{kDLOpenCL, 0};
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
 * \param model The model to use.
 * \param max_gen_len The maximum length of the generated sequence.
 * \param temperature The temperature to use for sampling.
 * \param top_p The top_p to use for sampling.
 */
void Chat(tvm::runtime::Module chat_mod, const std::string& model, int64_t max_gen_len = 2048,
          double temperature = 0.7, double top_p = 0.95, int64_t stream_interval = 2,
          int max_window_size = 768, int mean_gen_len = 128, double shift_fill_factor = 0.3) {
  // conv template detect
  std::string conv_template;
  if (model.find("vicuna") == 0) {
    conv_template = "vicuna_v1.1";
  } else if (model.find("dolly-") == 0) {
    conv_template = "dolly";
  } else if (model.find("stablelm") == 0) {
    conv_template = "stablelm";
  } else {
    LOG(FATAL) << "Do not recognize model name " << model;
  }

  // initialize chat context
  chat_mod.GetFunction("init_chat")(model, conv_template, max_gen_len, temperature, top_p,
                                    stream_interval, max_window_size, mean_gen_len,
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

  args.add_argument("--device-name").default_value("auto");
  args.add_argument("--artifact-path").default_value("dist");
  args.add_argument("--model").default_value("vicuna-v1-7b");
  args.add_argument("--dtype").default_value("auto");
  args.add_argument("--params").default_value("auto");
  args.add_argument("--evaluate").default_value(false).implicit_value(true);

  try {
    args.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << args;
    return 1;
  }

  std::string device_name = DetectDeviceName(args.get<std::string>("--device-name"));
  DLDevice device = GetDevice(device_name);
  std::string artifact_path = args.get<std::string>("--artifact-path");
  std::string model = args.get<std::string>("--model");
  std::string dtype = args.get<std::string>("--dtype");
  std::string params = args.get<std::string>("--params");

  std::string lib_name = model + "_" + device_name;
  std::string arch_suffix = GetArchSuffix();

  std::optional<std::filesystem::path> lib_path_opt;
  std::vector<std::string> dtype_candidates;

  if (dtype == "auto") {
    dtype_candidates = {"float16", "float32"};
  } else {
    dtype_candidates = {dtype};
  }
  std::optional<std::filesystem::path> lib_path;
  for (auto candidate : dtype_candidates) {
    std::vector<std::string> search_paths = {artifact_path + "/" + model + "/" + candidate,
                                             artifact_path + "/models/" + model,
                                             artifact_path + "/" + model, artifact_path + "/lib"};
    // search for lib_x86_64 and lib
    lib_path_opt = FindFile(search_paths,
                            {lib_name + arch_suffix + "_" + candidate, lib_name + "_" + candidate},
                            GetLibSuffixes());
    if (lib_path_opt) {
      dtype = candidate;
      break;
    }
  }
  if (!lib_path_opt) {
    std::cerr << "Cannot find " << model << " lib in preferred path \"" << artifact_path << "/"
              << model << "/" << dtype_candidates[0] << "/" << lib_name << "_"
              << dtype_candidates[0] << GetLibSuffixes()[0] << "\" or other candidate paths";
    return 1;
  }
  std::cout << "Use lib " << lib_path_opt.value().string() << std::endl;
  std::string model_path = lib_path_opt.value().parent_path().string();
  // get artifact path lib name
  auto tokenizer_path_opt = FindFile(
      {
          model_path,
          artifact_path + "/models/" + model,
          artifact_path + "/" + model,
      },
      {"tokenizer"}, {".model", ".json"});

  if (!tokenizer_path_opt) {
    std::cerr << "Cannot find tokenizer{.model/.json} in " << model_path;
    return 1;
  }

  if (params == "auto") {
    auto params_json_opt = FindFile(
        {
            model_path + "/params",
            artifact_path + "/" + model + "/" + dtype,
            artifact_path + "/" + model,
        },
        {"ndarray-cache"}, {".json"});
    if (!params_json_opt) {
      std::cerr << "Cannot find ndarray-cache.json for params in preferred path \"" << model_path
                << "/params\", \"" << artifact_path << "/" + model << "/" + dtype << "\", and \""
                << artifact_path << "/" << model << "\".";
      return 1;
    }
    std::string params_json = params_json_opt.value().string();
    params = params_json.substr(0, params_json.length() - 18);
  } else if (!FindFile({params}, {"ndarray-cache"}, {".json"})) {
    std::cerr << "Cannot find params/ndarray-cache.json in " << model_path;
    return 1;
  }

  try {
    auto lib = Module::LoadFromFile(lib_path_opt.value().string());
    std::cout << "Initializing the chat module..." << std::endl;
    Module chat_mod =
        mlc::llm::CreateChatModule(lib, tokenizer_path_opt.value().string(), params, device);
    std::cout << "Finish loading" << std::endl;
    PrintSpecialCommands();

    if (args.get<bool>("--evaluate")) {
      chat_mod.GetFunction("evaluate")();
    } else {
      Chat(chat_mod, model);
    }
  } catch (const std::runtime_error& err) {
    // catch exception so error message
    // get reported here without silently quit.
    std::cerr << err.what();
    return 1;
  }
  return 0;
}
