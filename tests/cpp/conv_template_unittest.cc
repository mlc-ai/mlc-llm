#include "json_ffi/conv_template.h"

#include <gtest/gtest.h>

namespace mlc {
namespace llm {
namespace json_ffi {

void _TestConvTemplateLoadJSONTextContent() {
  std::string conv_template =
      "{\n"
      "    \"name\": \"test\",\n"
      "    \"system_template\": \"abc{system_message}\",\n"
      "    \"system_message\": \"de\",\n"
      "    \"roles\": {\n"
      "      \"user\": \"Instruct\",\n"
      "      \"assistant\": \"Output\",\n"
      "      \"tool\": \"Instruct\"\n"
      "    },\n"
      "    \"role_templates\": {\n"
      "      \"user\": \"{user_message}\",\n"
      "      \"assistant\": \"{assistant_message}\",\n"
      "      \"tool\": \"{tool_message}\"\n"
      "    },\n"
      "    \"messages\": [[\"Instruct\", \"Hello\"], [\"Output\", \"Hey\"]],\n"
      "    \"seps\": [\n"
      "      \"\\n\"\n"
      "    ],\n"
      "    \"role_content_sep\": \": \",\n"
      "    \"role_empty_sep\": \":\",\n"
      "    \"stop_str\": [\n"
      "      \"<|endoftext|>\"\n"
      "    ],\n"
      "    \"add_role_after_system_message\": false,\n"
      "    \"stop_token_ids\": [\n"
      "      50256\n"
      "    ]"
      "}";

  auto res = Conversation::FromJSON(conv_template).IsOk();
  ASSERT_TRUE(res);
  const Conversation& conv = Conversation::FromJSON(conv_template).Unwrap();
  ASSERT_EQ(conv.name, "test");
  ASSERT_EQ(conv.system_template, "abc{system_message}");
  ASSERT_EQ(conv.system_message, "de");
  ASSERT_EQ(conv.roles.at("user"), "Instruct");
  ASSERT_EQ(conv.roles.at("assistant"), "Output");
  ASSERT_EQ(conv.roles.at("tool"), "Instruct");
  ASSERT_EQ(conv.role_templates.at("user"), "{user_message}");
  ASSERT_EQ(conv.role_templates.at("assistant"), "{assistant_message}");
  ASSERT_EQ(conv.role_templates.at("tool"), "{tool_message}");
  ASSERT_EQ(conv.messages.at(0).role, "Instruct");
  ASSERT_EQ(conv.messages.at(0).content.Text(), "Hello");
  ASSERT_EQ(conv.messages.at(1).role, "Output");
  ASSERT_EQ(conv.messages.at(1).content.Text(), "Hey");
  ASSERT_EQ(conv.seps.at(0), "\n");
  ASSERT_EQ(conv.role_content_sep, ": ");
  ASSERT_EQ(conv.role_empty_sep, ":");
  ASSERT_EQ(conv.stop_str.at(0), "<|endoftext|>");
  ASSERT_EQ(conv.add_role_after_system_message, false);
  ASSERT_EQ(conv.stop_token_ids.at(0), 50256);
}

void _TestConvTemplateLoadJSONPartsContent() {
  std::string conv_template =
      "{\n"
      "    \"name\": \"test\",\n"
      "    \"system_template\": \"abc{system_message}\",\n"
      "    \"system_message\": \"de\",\n"
      "    \"roles\": {\n"
      "      \"user\": \"Instruct\",\n"
      "      \"assistant\": \"Output\",\n"
      "      \"tool\": \"Instruct\"\n"
      "    },\n"
      "    \"role_templates\": {\n"
      "      \"user\": \"{user_message}\",\n"
      "      \"assistant\": \"{assistant_message}\",\n"
      "      \"tool\": \"{tool_message}\"\n"
      "    },\n"
      "    \"messages\": [[\"Instruct\", "
      "    [{\"type\": \"text\", \"text\": \"What's in the image?\"},\n"
      "     {\"type\": \"image_url\", \"image_url\": \"https://example.com/image.jpg\"}]\n"
      "    ]],\n"
      "    \"seps\": [\n"
      "      \"\\n\"\n"
      "    ],\n"
      "    \"role_content_sep\": \": \",\n"
      "    \"role_empty_sep\": \":\",\n"
      "    \"stop_str\": [\n"
      "      \"<|endoftext|>\"\n"
      "    ],\n"
      "    \"add_role_after_system_message\": false,\n"
      "    \"stop_token_ids\": [\n"
      "      50256\n"
      "    ]"
      "}";

  auto res = Conversation::FromJSON(conv_template).IsOk();
  ASSERT_TRUE(res);
  const Conversation& conv = Conversation::FromJSON(conv_template).Unwrap();
  ASSERT_EQ(conv.name, "test");
  ASSERT_EQ(conv.system_template, "abc{system_message}");
  ASSERT_EQ(conv.system_message, "de");
  ASSERT_EQ(conv.roles.at("user"), "Instruct");
  ASSERT_EQ(conv.roles.at("assistant"), "Output");
  ASSERT_EQ(conv.roles.at("tool"), "Instruct");
  ASSERT_EQ(conv.role_templates.at("user"), "{user_message}");
  ASSERT_EQ(conv.role_templates.at("assistant"), "{assistant_message}");
  ASSERT_EQ(conv.role_templates.at("tool"), "{tool_message}");
  ASSERT_EQ(conv.messages.at(0).role, "Instruct");
  ASSERT_EQ(conv.messages.at(0).content.Parts().at(0).at("type"), "text");
  ASSERT_EQ(conv.messages.at(0).content.Parts().at(0).at("text"), "What's in the image?");
  ASSERT_EQ(conv.messages.at(0).content.Parts().at(1).at("type"), "image_url");
  ASSERT_EQ(conv.messages.at(0).content.Parts().at(1).at("image_url"),
            "https://example.com/image.jpg");
  ASSERT_EQ(conv.seps.at(0), "\n");
  ASSERT_EQ(conv.role_content_sep, ": ");
  ASSERT_EQ(conv.role_empty_sep, ":");
  ASSERT_EQ(conv.stop_str.at(0), "<|endoftext|>");
  ASSERT_EQ(conv.add_role_after_system_message, false);
  ASSERT_EQ(conv.stop_token_ids.at(0), 50256);
}

std::string _RenderUserPrompt(const std::string& render_empty_system_field,
                              const std::string& request_messages) {
  std::string conv_template =
      "{\n"
      "    \"name\": \"test\",\n"
      "    \"system_template\": \"<|im_start|>system\\n{system_message}<|im_end|>\\n\",\n"
      "    \"system_message\": \"\",\n" +
      render_empty_system_field +
      "    \"roles\": {\"user\": \"<|im_start|>user\", \"assistant\": \"<|im_start|>assistant\"},\n"
      "    \"role_templates\": {\"user\": \"{user_message}\", \"assistant\": "
      "\"{assistant_message}\"},\n"
      "    \"messages\": [],\n"
      "    \"seps\": [\"<|im_end|>\\n\"],\n"
      "    \"role_content_sep\": \"\\n\",\n"
      "    \"role_empty_sep\": \"\\n\",\n"
      "    \"stop_str\": [],\n"
      "    \"add_role_after_system_message\": true,\n"
      "    \"stop_token_ids\": [2]\n"
      "}";
  Conversation conv = Conversation::FromJSON(conv_template).Unwrap();
  ChatCompletionRequest request =
      ChatCompletionRequest::FromJSON("{\"messages\": " + request_messages + "}").Unwrap();
  std::vector<serve::Data> prompt =
      CreatePrompt(conv, request, ModelConfig(), DLDevice{kDLCPU, 0}).Unwrap();
  std::string text;
  for (const serve::Data& data : prompt) {
    const auto* text_data = data.as<serve::TextDataNode>();
    EXPECT_NE(text_data, nullptr);
    if (text_data != nullptr) text += text_data->text;
  }
  return text;
}

void _TestConvTemplateRenderEmptySystemMessage() {
  const std::string user_only = "[{\"role\": \"user\", \"content\": \"Hi\"}]";
  const std::string with_system =
      "[{\"role\": \"system\", \"content\": \"Be brief.\"}, {\"role\": \"user\", \"content\": "
      "\"Hi\"}]";
  const std::string empty_system_turn = "<|im_start|>system\n<|im_end|>\n";
  const std::string user_turn = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";
  // Unset defaults to true: the empty system turn is rendered, as before.
  ASSERT_EQ(_RenderUserPrompt("", user_only), empty_system_turn + user_turn);
  ASSERT_EQ(_RenderUserPrompt("    \"render_empty_system_message\": true,\n", user_only),
            empty_system_turn + user_turn);
  // False skips the system template only when the system message is empty.
  ASSERT_EQ(_RenderUserPrompt("    \"render_empty_system_message\": false,\n", user_only),
            user_turn);
  ASSERT_EQ(_RenderUserPrompt("    \"render_empty_system_message\": false,\n", with_system),
            "<|im_start|>system\nBe brief.<|im_end|>\n" + user_turn);
}

TEST(JsonFFIConvTest, LoadJSONTextContentTest) { _TestConvTemplateLoadJSONTextContent(); }
TEST(JsonFFIConvTest, LoadJSONPartsContentTest) { _TestConvTemplateLoadJSONPartsContent(); }
TEST(JsonFFIConvTest, RenderEmptySystemMessageTest) { _TestConvTemplateRenderEmptySystemMessage(); }

}  // namespace json_ffi
}  // namespace llm
}  // namespace mlc
