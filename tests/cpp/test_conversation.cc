#include <conversation.h>
#include <gtest/gtest.h>

void _TestConversationJSONRoundTrip(std::string templ_name) {
  mlc::llm::Conversation conv = mlc::llm::Conversation::FromTemplate(templ_name);
  std::string conv_json = conv.SerializeToJSON();
  mlc::llm::Conversation conv_new;
  conv_new.LoadJSONOverride(conv_json);
  ASSERT_EQ(conv, conv_new);
}

TEST(ConversationTest, ConversationJSONRoundTripTest) {
  _TestConversationJSONRoundTrip("vicuna_v1.1");
  _TestConversationJSONRoundTrip("conv_one_shot");
  _TestConversationJSONRoundTrip("redpajama_chat");
  _TestConversationJSONRoundTrip("LM");
}