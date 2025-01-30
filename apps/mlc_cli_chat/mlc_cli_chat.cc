/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file mlc_cli_chat.cc
 */

#include <iostream>

#include "chat_state.h"
#include "engine.h"

struct Args {
  std::string model;
  std::string model_lib_path;
  std::string device = "auto";
  bool evaluate = false;
  int eval_prompt_len = 128;
  int eval_gen_len = 1024;
  std::string prompt;
};

// Help Prompt
void printHelp() {
  std::cout
      << "MLCChat CLI is the command line tool to run MLC-compiled LLMs out of the box.\n"
      << "Note: the --model argument is required. It can either be the model name with its "
      << "quantization scheme or a full path to the model folder. In the former case, the "
      << "provided name will be used to search for the model folder over possible paths. "
      << "--model-lib-path argument is optional. If unspecified, the --model argument will be used "
      << "to search for the library file over possible paths.\n\n"
      << "Usage: mlc_cli_chat [options]\n"
      << "Options:\n"
      << "  --model             [required] the model to use\n"
      << "  --model-lib         [optional] the full path to the model library file to use\n"
      << "  --device            (default: auto)\n"
      << "  --with-prompt       [optional] runs one session with given prompt\n"
      << "  --help              [optional] Tool usage information\n"
      ;
}

// Method to parse the args
Args parseArgs(int argc, char* argv[]) {
  Args args;

  // Taking the arguments after the exectuable
  std::vector<std::string> arguments(argv + 1, argv + argc);

  for (size_t i = 0; i < arguments.size(); ++i) {
    if (arguments[i] == "--model" && i + 1 < arguments.size()) {
      args.model = arguments[++i];
    } else if (arguments[i] == "--model-lib" && i + 1 < arguments.size()) {
      args.model_lib_path = arguments[++i];
    } else if (arguments[i] == "--device" && i + 1 < arguments.size()) {
      args.device = arguments[++i];
    } else if (arguments[i] == "--evaluate") {
      args.evaluate = true;
    } else if (arguments[i] == "--eval-prompt-len" && i + 1 < arguments.size()) {
      args.eval_prompt_len = std::stoi(arguments[++i]);
    } else if (arguments[i] == "--eval-gen-len" && i + 1 < arguments.size()) {
      args.eval_gen_len = std::stoi(arguments[++i]);
    } else if (arguments[i] == "--with-prompt" && i + 1 < arguments.size()) {
      args.prompt = arguments[++i];
    } else if (arguments[i] == "--help") {
      printHelp();
      exit(0);
    } else {
      printHelp();
      throw std::runtime_error("Unknown or incomplete argument: " + arguments[i]);
    }
  }

  if (args.model.empty()) {
    printHelp();
    throw std::runtime_error("Invalid arguments");
  }

  return args;
}

// Method to detect the device
std::pair<std::string, int> DetectDevice(std::string device) {
  std::string device_name;
  int device_id;
  int delimiter_pos = device.find(":");

  // cuda:0 which means the device name is cuda and the device id is 0
  if (delimiter_pos == std::string::npos) {
    device_name = device;
    device_id = 0;
  } else {
    device_name = device.substr(0, delimiter_pos);
    device_id = std::stoi(device.substr(delimiter_pos + 1, device.length()));
  }
  return {device_name, device_id};
}

int main(int argc, char* argv[]) {
  Args args = parseArgs(argc, argv);

  // model path
  std::string model_path = args.model;

  // model-lib path
  std::string model_lib_path = args.model_lib_path;

  // Get the device name and device id
  auto [device_name, device_id] = DetectDevice(args.device);

  // mode of interaction
  std::string mode{"interactive"};

  ChatState chat_state(model_path, model_lib_path, mode, device_name, 0);

  return chat_state.chat(args.prompt);
}
