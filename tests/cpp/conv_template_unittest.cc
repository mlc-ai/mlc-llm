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

TEST(JsonFFIConvTest, LoadJSONTextContentTest) { _TestConvTemplateLoadJSONTextContent(); }
TEST(JsonFFIConvTest, LoadJSONPartsContentTest) { _TestConvTemplateLoadJSONPartsContent(); }

}  // namespace json_ffi
}  // namespace llm
}  // namespace mlc
