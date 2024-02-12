/*!
 *  Copyright (c) 2023 by Contributors
 * \file conversation.h
 * \brief Header of conversation template in MLC-LLM.
 */
#include <picojson.h>
#include <tvm/runtime/module.h>

#include <string>
#include <vector>

namespace mlc {
namespace llm {

enum class SeparatorStyle {
  /*! \brief Add separator between role and message. */
  kSepRoleMsg,
  /*! \brief Code completion without separators or roles. No memory. */
  kCodeCompletion,
  /*! \brief raw language model style, always only returns last message. */
  kLM,
};

enum class PlaceInPrompt : int {
  /*! \brief The input message should have role names and corresponding seperators appended both
     prior to it and after it, making it a complete prompt. */
  kAll,
  /*! \brief The input message is only the beginning part of a prompt, no role name and separator
     should be appended after the message since there will be future messages appended after the
     message. */
  kBegin,
  /*! \brief The input message is in the middle of a prompt, nothing should be appended before or
     after the message. */
  kMiddle,
  /*! \brief The input message is the ending part of a prompt, no role name and separator should be
     appended prior to it since the message is concatenated to some prior messages. */
  kEnd,
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
  /*! \brief the separator style */
  SeparatorStyle separator_style = SeparatorStyle::kSepRoleMsg;
  /*! \brief Separator that appended to the messages, can be of size 1 or two */
  std::vector<std::string> seps;
  /*! \brief Separator between role and message. */
  std::string role_msg_sep = "";
  /*! \brief The separator to append to role when there is no message yet. */
  std::string role_empty_sep = "";
  /*! \brief Matches stop str. */
  std::string stop_str = "";
  /*! \brief token list that matches stop */
  std::vector<int32_t> stop_tokens = {};
  /*! \brief token list prefixing the conversation */
  std::vector<int32_t> prefix_tokens = {};
  /*!
   * \brief Whether caller should consider add bos before system prompt.
   * \note This option is only used for llama models atm.
   */
  bool add_bos = false;
  /*! \brief Whether the pixel values are also expected as input*/
  bool use_pixel_values = false;
  /*! \brief The image token index (if use_pixel_values is true) */
  int32_t image_token_index = -1;

  Conversation() = default;

  inline bool operator==(const Conversation& other) const {
    bool eq_roles = true;
    if (roles.size() != other.roles.size()) {
      eq_roles = false;
    } else {
      eq_roles = std::equal(roles.begin(), roles.end(), other.roles.begin());
    }
    bool eq_messages = true;
    if (messages.size() != other.messages.size()) {
      eq_messages = false;
    } else {
      for (size_t i = 0; i < messages.size(); ++i) {
        const std::vector<std::string>& lhs_message_i = messages[i];
        const std::vector<std::string>& rhs_message_i = other.messages[i];
        if (lhs_message_i.size() != rhs_message_i.size()) {
          eq_messages = false;
          break;
        } else {
          eq_messages &=
              std::equal(lhs_message_i.begin(), lhs_message_i.end(), rhs_message_i.begin());
        }
      }
    }
    bool eq_seps = true;
    if (seps.size() != other.seps.size()) {
      eq_seps = false;
    } else {
      eq_seps = std::equal(seps.begin(), seps.end(), other.seps.begin());
    }
    bool eq_stop_tokens = true;
    if (stop_tokens.size() != other.stop_tokens.size()) {
      eq_stop_tokens = false;
    } else {
      eq_stop_tokens =
          std::equal(stop_tokens.begin(), stop_tokens.end(), other.stop_tokens.begin());
    }
    bool eq_prefix_tokens = true;
    if (prefix_tokens.size() != other.prefix_tokens.size()) {
      eq_prefix_tokens = false;
    } else {
      eq_prefix_tokens =
          std::equal(prefix_tokens.begin(), prefix_tokens.end(), other.prefix_tokens.begin());
    }
    return (name == other.name) && (system == other.system) && (offset == other.offset) &&
           (separator_style == other.separator_style) && (role_msg_sep == other.role_msg_sep) &&
           (role_empty_sep == other.role_empty_sep) && (stop_str == other.stop_str) &&
           (add_bos == other.add_bos) && eq_roles && eq_messages && eq_seps && eq_stop_tokens &&
           eq_prefix_tokens;
  }

  /**
   * \brief Create conversation from existing registered template.
   * \param name The template name.
   */
  static Conversation FromTemplate(const std::string& name);

  /*!
   * \brief Load JSON config in raw string and overrides options.
   *
   * \param config_str A json config in raw string that partially specifies
   *        some of the options.
   * \param partial_update Whether it's a partial update or full update, if set to true,
   *        we perform a partial update on some of the provided options; if set to false, all
   *        options must be provided.
   * \note This function overrides existing configurations.
   */
  void LoadJSONOverride(const std::string& config_str, bool partial_update = false);

  /*!
   * \brief Load JSON config and overrides options.
   *
   * \param config_json A json config in picojson type that is partially specifies
   *        some of the options.
   * \param partial_update Whether it's a partial update or full update, if set to true,
   *        we perform a partial update on some of the provided options; if set to false, all
   *        options must be provided.
   * \note This function overrides existing configurations.
   */
  void LoadJSONOverride(const picojson::value& config_json, bool partial_update = false);

  /*!
   * \brief Serialize the Conversation to JSON.
   * \return Serialized conversion in JSON format.
   */
  picojson::value SerializeToJSON() const;

  /*!
   * \brief Serialize the Conversation to JSON String.
   * \return A string storing the serialized conversation in JSON format.
   */
  std::string GetConfigJSON() const;

  /*!
   * \brief Get the entire prompt array
   * \param place_in_prompt The place of the input message in the prompt.
   * \return A vector of strings storing the prompt array.
   */
  std::vector<std::string> GetPromptArray(PlaceInPrompt place_in_prompt = PlaceInPrompt::kAll) {
    return GetPromptArrayInternal(0, place_in_prompt);
  }

  /**
   * \brief Get prompt array for the last round.
   * The last round conversation is usually unprocessed by LM
   * \param place_in_prompt The place of the input message in the prompt.
   */
  std::vector<std::string> GetPromptArrayLastRound(
      PlaceInPrompt place_in_prompt = PlaceInPrompt::kAll) {
    ICHECK_GE(this->messages.size(), 2);
    return GetPromptArrayInternal(this->messages.size() - 2, place_in_prompt);
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

  void Reset() { this->messages.resize(this->offset); }

 private:
  // Identity function
  static std::string Identity(std::string msg) { return msg; }
  /**
   * \brief Internal function to get prompted array
   * \param system_prefix The system prompt prefix that needs to be added if start_pos == 0
   * \param start_pos The start message position.
   * \param role_msg_sep The separator between role and message.
   * \param role_empty_sep The separator to appending to role when we do not yet have a message.
   * \param place_in_prompt The place of the input message in the prompt.
   */
  template <typename FProcMessage>
  std::vector<std::string> GetPromptArrayInternal(
      std::string system_prefix, size_t start_pos, std::string role_msg_sep,
      std::string role_empty_sep, FProcMessage fproc_message,
      PlaceInPrompt place_in_prompt = PlaceInPrompt::kAll) const {
    std::vector<std::string> ret;
    ret.reserve(messages.size() - start_pos + 1);
    if (place_in_prompt == PlaceInPrompt::kBegin || place_in_prompt == PlaceInPrompt::kAll) {
      if (start_pos == 0) {
        if (system_prefix.length() != 0) {
          ret.push_back(system_prefix);
        }
      } else {
        // need to add a sep of last response
        // which was not added in the processing step.
        ret.push_back(this->seps[1 % this->seps.size()]);
      }
    }

    ICHECK_EQ(start_pos % 2, 0);
    for (size_t i = start_pos; i < this->messages.size(); ++i) {
      const auto& item = this->messages[i];
      // seps[0] or seps[1] depending on current location.
      const auto& end_sep = this->seps[i % this->seps.size()];
      const auto& role = item[0];
      if (item.size() == 2) {
        const std::string message = fproc_message(item[1]);
        if (i == this->messages.size() - 2 && i == start_pos &&
            place_in_prompt == PlaceInPrompt::kMiddle) {
          ret.push_back(message);
        } else if (i == this->messages.size() - 2 && (place_in_prompt == PlaceInPrompt::kBegin ||
                                                      place_in_prompt == PlaceInPrompt::kMiddle)) {
          ret.push_back(role + role_msg_sep + message);
        } else if (i == start_pos && (place_in_prompt == PlaceInPrompt::kEnd ||
                                      place_in_prompt == PlaceInPrompt::kMiddle)) {
          ret.push_back(message + end_sep);
        } else {
          ret.push_back(role + role_msg_sep + message + end_sep);
        }

      } else {
        ICHECK(item.size() == 1);
        if (!(i == this->messages.size() - 1) || place_in_prompt == PlaceInPrompt::kEnd ||
            place_in_prompt == PlaceInPrompt::kAll) {
          ret.push_back(role + role_empty_sep);
        }
      }
    }
    return ret;
  }
  /**
   * \brief dispatcher based on separator style
   * \param place_in_prompt The place of the input message in the prompt.
   */
  std::vector<std::string> GetPromptArrayInternal(
      size_t start_pos, PlaceInPrompt place_in_prompt = PlaceInPrompt::kAll) {
    if (this->separator_style == SeparatorStyle::kSepRoleMsg) {
      std::string system_prefix;
      if (!this->system.empty()) {
        system_prefix = this->system + this->seps[0];
      }
      return GetPromptArrayInternal(
          /* system_prefix= */ system_prefix,
          /* start_pos= */ start_pos,
          /* role_msg_sep= */ role_msg_sep,
          /* role_empty_sep= */ role_empty_sep,
          /* fproc_message= */ Identity,
          /* place_in_prompt= */ place_in_prompt);
    } else {
      ICHECK(this->separator_style == SeparatorStyle::kLM ||
             this->separator_style == SeparatorStyle::kCodeCompletion)
          << "Unsupported separator_style";
      // special handle LM, LM mode have no memory
      // and only returns last one
      if (this->messages.size() >= 2) {
        return {this->messages[this->messages.size() - 2][1]};
      } else {
        return {};
      }
    }
  }
};

}  // namespace llm
}  // namespace mlc
