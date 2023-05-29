#include <conversation.h>
#include <gtest/gtest.h>

void _TestConversationJSONRoundTrip(std::string templ_name) {
  mlc::llm::Conversation conv = mlc::llm::Conversation::FromTemplate(templ_name);
  std::string conv_json = conv.GetConfigJSON();
  mlc::llm::Conversation conv_new;
  conv_new.LoadJSONOverride(conv_json, false);
  ASSERT_EQ(conv, conv_new);
}

void _TestConversationPartialUpdate() {
  mlc::llm::Conversation conv;
  std::string json_str = "{\"offset\": -1}";
  ASSERT_ANY_THROW(conv.LoadJSONOverride(json_str, false));
  conv.LoadJSONOverride(json_str, true);
  ASSERT_EQ(conv.offset, -1);
}

TEST(ConversationTest, ConversationJSONRoundTripTest) {
  _TestConversationJSONRoundTrip("vicuna_v1.1");
  _TestConversationJSONRoundTrip("conv_one_shot");
  _TestConversationJSONRoundTrip("redpajama_chat");
  _TestConversationJSONRoundTrip("LM");
}

TEST(ConversationTest, ConversationPartialUpdateTest) {
  _TestConversationPartialUpdate();
}
