/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file chat_state.h
 */

#ifndef MLC_CLI_CHAT_CHAT_STATE_H
#define MLC_CLI_CHAT_CHAT_STATE_H

#include "base.h"
#include "engine.h"

void print_help_str();

class ChatState {
 public:
  std::vector<Message> history;
  size_t history_window_begin;
  std::shared_ptr<JSONFFIEngineWrapper> __json_wrapper;

  ChatState(std::string model_path, std::string model_lib_path, std::string mode,
            std::string device, int device_id = 0);

  void slide_history();
  std::vector<Message> get_current_history_window();
  int generate(const std::string& prompt);
  void reset();
  int chat(std::string prompt = "");
};

#endif
