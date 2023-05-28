/*!
 *  Copyright (c) 2023 by Contributors
 * \file llm_chat.cc
 * \brief Implementation of llm chat.
 */
#define PICOJSON_USE_INT64
#define __STDC_FORMAT_MACROS
#include <tvm/runtime/module.h>

#include <string>
#include <vector>

namespace mlc {
namespace llm {

enum class SeparatorStyle {
  /*! \brief add colon ": " beween role and message. */
  kAddColon,
  /*! \brief raw langauge model style, always only returns last message. */
  kLM,
};

/*!
 * \brief helper class to keep track of conversation.
 */
class Conversation {
 public:
  /*! \brief name of the conversation. */
  std::string name;
  /*! \brief The system prompt. */
  std::string system;
  /*! \brief The roles in the system. */
  std::vector<std::string> roles;
  /*! \brief The message history. */
  std::vector<std::vector<std::string>> messages = {};
  /*! \brief offset to point to the end of few short examples */
  int32_t offset = 0;
  /*! \brief the seperator style */
  SeparatorStyle separator_style = SeparatorStyle::kAddColon;
  /*!
   * \brief Seperator that appended to the messages, can be of size 1 or two
   */
  std::vector<std::string> seps;
  /*! \brief Matches stop str. */
  std::string stop_str = "";
  /*! \brief tokenlist that matches stop */
  std::vector<int32_t> stop_tokens = {};
  /*!
   * \brief Whether caller should consider add bos before system prompt.
   * \note This option is only used for llama models atm.
   */
  bool add_bos = false;

  Conversation() = default;

  /**
   * \brief Create conversation from existing registered template.
   * \param name The template name.
   */
  static Conversation FromTemplate(const std::string& name);

  // TODO(mlc-team): Implement this
  /*!
   * \brief Load overrides from JSON that partially
   * overrides some of the options.
   *
   * \param config_json A json config that partially specifies
   *        some of the options
   * \note This function overrides existing configurations.
   */
  void LoadJSONOverride(const std::string& config_json);

  /*!
   * \brief Get the entire prompt array
   */
  std::vector<std::string> GetPromptArray() { return GetPromptArrayInternal(0); }

  /**
   * \brief Get prompt array for the last round.
   * The last round convo is usually unprocessed by LM
   */
  std::vector<std::string> GetPrompArrayLastRound() {
    ICHECK_GE(this->messages.size(), 2);
    return GetPromptArrayInternal(this->messages.size() - 2);
  }

  void AppendMessage(std::string role, std::string message) {
    this->messages.push_back({role, message});
  }

  void AppendReplyHeader(std::string role) { this->messages.push_back({role}); }

  void FinishReply(std::string msg) {
    ICHECK_NE(this->messages.size(), 0);
    ICHECK_EQ(this->messages.back().size(), 1) << "Already assigned";
    this->messages.back().push_back(msg);
  }

 private:
  // Identity function
  static std::string Identity(std::string msg) { return msg; }
  /**
   * \brief Internal function to get prompted array
   * \param system_prefix The system prompt prefix that needs to be added if start_pos == 0
   * \param start_pos The start message position.
   * \param role_msg_sep The seperator between role and message.
   * \param role_empty_sep The seperator to appending to role when we do not yet have a message.
   */
  template <typename FProcMessage>
  std::vector<std::string> GetPromptArrayInternal(std::string system_prefix, size_t start_pos,
                                                  std::string role_msg_sep,
                                                  std::string role_empty_sep,
                                                  FProcMessage fproc_message) const {
    std::vector<std::string> ret;
    ret.reserve(messages.size() - start_pos + 1);
    if (start_pos == 0) {
      if (system_prefix.length() != 0) {
        ret.push_back(system_prefix);
      }
    } else {
      // need to add a sep of last response
      // which was not added in the processing step.
      ret.push_back(this->seps[1 % this->seps.size()]);
    }

    ICHECK_EQ(start_pos % 2, 0);
    for (size_t i = start_pos; i < this->messages.size(); ++i) {
      const auto& item = this->messages[i];
      // seps[0]  or seps[1] depending on current location.
      const auto& end_sep = this->seps[i % this->seps.size()];
      const auto& role = item[0];
      if (item.size() == 2) {
        const std::string message = fproc_message(item[1]);
        ret.push_back(role + role_msg_sep + message + end_sep);
      } else {
        ICHECK(item.size() == 1);
        ret.push_back(role + role_empty_sep);
      }
    }
    return ret;
  }
  // dispatcher based on separator style
  std::vector<std::string> GetPromptArrayInternal(size_t start_pos) {
    if (this->separator_style == SeparatorStyle::kAddColon) {
      std::string system_prefix;
      if (!this->system.empty()) {
        system_prefix = this->system + this->seps[0];
      }
      return GetPromptArrayInternal(
          /* system_prefix= */ system_prefix,
          /* start_pos= */ start_pos,
          /* role_msg_sep= */ ": ",
          /* role_empty_sep= */ ":",
          /* fproc_message= */ Identity);
    } else {
      ICHECK(this->separator_style == SeparatorStyle::kLM) << "Unsupported separator_style";
      // special handle LM, LM mode have no memory
      // and only returns last one
      return {this->messages[this->messages.size() - 2][1]};
    }
  }
};

}  // namespace llm
}  // namespace mlc
