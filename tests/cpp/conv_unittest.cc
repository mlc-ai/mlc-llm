#include <conversation.h>
#include <gtest/gtest.h>

void _TestConversationLoadJSON() {
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
      "    \"stop_token_ids\": [\n"
      "      50256\n"
      "    ],\n"
      "    \"function_string\": \"\",\n"
      "    \"use_function_calling\": false\n"
      "}";
  mlc::llm::Conversation conv;
  conv.LoadJSONOverride(conv_template, true);
  ASSERT_EQ(conv.name, "test");
  ASSERT_EQ(conv.system, "abcde");

  std::vector<std::string> expected_roles{"Instruct", "Output"};
  ASSERT_EQ(conv.roles, expected_roles);

  std::vector<std::vector<std::string>> expected_messages = {{"Instruct", "Hello"},
                                                             {"Output", "Hey"}};
  ASSERT_EQ(conv.messages, expected_messages);
  ASSERT_EQ(conv.offset, 2);

  std::vector<std::string> expected_seps = {"\n"};
  ASSERT_EQ(conv.seps, expected_seps);

  ASSERT_EQ(conv.role_msg_sep, ": ");
  ASSERT_EQ(conv.role_empty_sep, ":");
  ASSERT_EQ(conv.stop_str, "<|endoftext|>");

  std::vector<int32_t> expected_stop_tokens = {50256};
  ASSERT_EQ(conv.stop_tokens, expected_stop_tokens);
}

void _TestConversationJSONRoundTrip(std::string templ_name) {
  mlc::llm::Conversation conv = mlc::llm::Conversation::FromTemplate(templ_name);
  std::string conv_json = conv.GetConfigJSON();
  mlc::llm::Conversation conv_new;
  conv_new.LoadJSONOverride(conv_json, false);
  ASSERT_EQ(conv, conv_new);
}

void _TestConversationPartialUpdate() {
  mlc::llm::Conversation conv;
  std::string json_str = "{\"name\": \"test\"}";
  ASSERT_ANY_THROW(conv.LoadJSONOverride(json_str, false));
  conv.LoadJSONOverride(json_str, true);
  ASSERT_EQ(conv.name, "test");
}

TEST(ConversationTest, ConversationLoadJSONTest) { _TestConversationLoadJSON(); }

TEST(ConversationTest, ConversationJSONRoundTripTest) {
  _TestConversationJSONRoundTrip("vicuna_v1.1");
  _TestConversationJSONRoundTrip("conv_one_shot");
  _TestConversationJSONRoundTrip("redpajama_chat");
  _TestConversationJSONRoundTrip("LM");
}

TEST(ConversationTest, ConversationPartialUpdateTest) { _TestConversationPartialUpdate(); }
