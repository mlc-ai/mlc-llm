#include <string>
#include <unordered_map>

#include "conversation.h"

namespace mlc {
namespace llm {

Conversation VicunaV11() {
  Conversation conv;
  conv.name = "vicuna_v1.1";
  conv.system =
      ("A chat between a curious user and an artificial intelligence assistant. "
       "The assistant gives helpful, detailed, and polite answers to the user's questions.");
  conv.roles = {"USER", "ASSISTANT"};
  conv.messages = {};
  conv.offset = 0;
  conv.separator_style = SeparatorStyle::kAddColon;
  conv.seps = {" ", "</s>"};
  // TODO(mlc-team): add eos to mlc-chat-config
  // and remove eos from stop token setting.
  conv.stop_tokens = {2};
  conv.stop_str = "</s>";
  conv.add_bos = true;
  return conv;
}

Conversation ConvOneShot() {
  Conversation conv;
  conv.name = "conv_one_shot";
  conv.system =
      ("A chat between a curious human and an artificial intelligence assistant. "
       "The assistant gives helpful, detailed, and polite answers to the human's questions.");
  conv.roles = {"Human", "Assistant"};
  conv.messages = {
      {"Human", "What are the key differences between renewable and non-renewable energy sources?"},
      {"Assistant",
       "Renewable energy sources are those that can be replenished naturally in a relatively "
       "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
       "Non-renewable energy sources, on the other hand, are finite and will eventually be "
       "depleted, such as coal, oil, and natural gas. Here are some key differences between "
       "renewable and non-renewable energy sources:\n"
       "1. Availability: Renewable energy sources are virtually inexhaustible, while "
       "non-renewable "
       "energy sources are finite and will eventually run out.\n"
       "2. Environmental impact: Renewable energy sources have a much lower environmental "
       "impact "
       "than non-renewable sources, which can lead to air and water pollution, greenhouse gas "
       "emissions, "
       "and other negative effects.\n"
       "3. Cost: Renewable energy sources can be more expensive to initially set up, but they "
       "typically "
       "have lower operational costs than non-renewable sources.\n"
       "4. Reliability: Renewable energy sources are often more reliable and can be used in "
       "more remote "
       "locations than non-renewable sources.\n"
       "5. Flexibility: Renewable energy sources are often more flexible and can be adapted "
       "to different "
       "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
       "6. Sustainability: Renewable energy sources are more sustainable over the long term, "
       "while "
       "non-renewable sources are not, and their depletion can lead to economic and social "
       "instability."}};
  conv.separator_style = SeparatorStyle::kAddColon;
  conv.offset = 2;
  conv.seps = {"\n###"};
  conv.stop_str = "###";
  // TODO(mlc-team): add eos to mlc-chat-config
  // and remove eos from stop token setting.
  conv.stop_tokens = {2};
  conv.add_bos = true;
  return conv;
}

Conversation RedPajamaChat() {
  Conversation conv;
  conv.name = "redpajama_chat";
  conv.system = "";
  conv.roles = {"<human>", "<bot>"};
  conv.messages = {};
  conv.separator_style = SeparatorStyle::kAddColon;
  conv.offset = 0;
  conv.seps = {"\n"};
  conv.stop_str = "<human>";
  // TODO(mlc-team): add eos to mlc-chat-config
  // and remove eos from stop token setting.
  conv.stop_tokens = {0};
  conv.add_bos = false;
  return conv;
}

Conversation VanillaLM() {
  Conversation conv;
  conv.name = "LM";
  conv.system = "";
  conv.roles = {"Prompt", "LM"};
  conv.messages = {};
  conv.separator_style = SeparatorStyle::kLM;
  conv.offset = 0;
  conv.seps = {""};
  // TODO(mlc-team): add eos to mlc-chat-config
  // and remove eos from stop token setting.
  // so the same template works for more tokenizers
  conv.stop_tokens = {2};
  conv.add_bos = true;
  return conv;
}

using ConvFactory = Conversation (*)();

Conversation Conversation::FromTemplate(const std::string& name) {
  static std::unordered_map<std::string, ConvFactory> factory = {
      {"vicuna_v1.1", VicunaV11},
      {"conv_one_shot", ConvOneShot},
      {"redpajama_chat", RedPajamaChat},
      {"LM", VanillaLM},
  };
  auto it = factory.find(name);
  if (it == factory.end()) {
    LOG(FATAL) << "Unknown conversation template: " << name;
  }
  return it->second();
}

}  // namespace llm
}  // namespace mlc
