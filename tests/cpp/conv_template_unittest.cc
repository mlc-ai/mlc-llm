#include "json_ffi/conv_template.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>

#include "support/json_parser.h"

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
      "    [{\"type\": \"text\", \"text\": \"What's in the image?\", \"score\": 3},\n"
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
  ASSERT_EQ(conv.messages.at(0).content.Parts().at(0).at("score"), "3");
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

TEST(JsonFFIConvTest, Qwen38PinnedPlainTextHistory) {
  const auto fixture_path = std::filesystem::path(__FILE__).parent_path().parent_path() /
                            "python/protocol/fixtures/qwen3_8/hf.json";
  std::ifstream file(fixture_path);
  ASSERT_TRUE(file.good());
  std::stringstream buffer;
  buffer << file.rdbuf();
  auto fixture = json::ParseToJSONObjectWithResultReturn(buffer.str());
  ASSERT_TRUE(fixture.IsOk());
  auto cases = fixture.Unwrap().at("cases").cast<tvm::ffi::json::Array>();
  ASSERT_EQ(cases.size(), 52);
  std::ifstream template_file(fixture_path.parent_path() / "mlc_templates.json");
  ASSERT_TRUE(template_file.good());
  std::stringstream template_buffer;
  template_buffer << template_file.rdbuf();
  auto templates = json::ParseToJSONObjectWithResultReturn(template_buffer.str());
  ASSERT_TRUE(templates.IsOk());
  auto template_map = templates.Unwrap();
  for (const auto& value : cases) {
    auto item = value.cast<tvm::ffi::json::Object>();
    std::string name = item.at("template").cast<std::string>();
    SCOPED_TRACE(name + ":" + item.at("name").cast<std::string>());
    auto parsed_conv = Conversation::FromJSON(template_map.at(name).cast<tvm::ffi::json::Object>());
    ASSERT_TRUE(parsed_conv.IsOk());
    Conversation conv = parsed_conv.Unwrap();
    tvm::ffi::json::Object request_json;
    request_json.Set("messages", item.at("messages"));
    auto request = ChatCompletionRequest::FromJSON(tvm::ffi::json::Stringify(request_json));
    ASSERT_TRUE(request.IsOk());
    auto result = CreatePrompt(conv, request.Unwrap(), ModelConfig(), DLDevice{kDLCPU, 0});
    ASSERT_TRUE(result.IsOk());
    std::string text;
    for (const auto& part : result.Unwrap()) {
      auto node = part.as<TextDataNode>();
      ASSERT_NE(node, nullptr);
      text += node->text;
    }
    EXPECT_EQ(text, item.at("rendered").cast<std::string>());
    auto null_extra_json = request_json;
    null_extra_json.Set("extra_body", nullptr);
    auto null_extra = ChatCompletionRequest::FromJSON(tvm::ffi::json::Stringify(null_extra_json));
    ASSERT_TRUE(null_extra.IsOk());
    EXPECT_TRUE(CreatePrompt(conv, null_extra.Unwrap(), ModelConfig(), DLDevice{kDLCPU, 0}).IsOk());
    for (const char* choice : {"none", "auto"}) {
      auto choice_json = request_json;
      choice_json.Set("tool_choice", choice);
      auto rejected = ChatCompletionRequest::FromJSON(tvm::ffi::json::Stringify(choice_json));
      ASSERT_TRUE(rejected.IsOk());
      EXPECT_TRUE(
          CreatePrompt(conv, rejected.Unwrap(), ModelConfig(), DLDevice{kDLCPU, 0}).IsErr());
    }
    for (const char* option : {"enable_thinking", "reasoning_effort", "preserve_thinking"}) {
      auto rejected_json = request_json;
      rejected_json.Set(option, false);
      auto rejected = ChatCompletionRequest::FromJSON(tvm::ffi::json::Stringify(rejected_json));
      ASSERT_TRUE(rejected.IsOk());
      EXPECT_TRUE(
          CreatePrompt(conv, rejected.Unwrap(), ModelConfig(), DLDevice{kDLCPU, 0}).IsErr());
    }
  }
}
TEST(JsonFFIConvTest, Qwen36PinnedPlainTextHistory) {
  const auto fixture_path = std::filesystem::path(__FILE__).parent_path().parent_path() /
                            "python/protocol/fixtures/qwen3_6/hf.json";
  std::ifstream file(fixture_path);
  ASSERT_TRUE(file.good());
  std::stringstream buffer;
  buffer << file.rdbuf();
  auto fixture = json::ParseToJSONObjectWithResultReturn(buffer.str());
  ASSERT_TRUE(fixture.IsOk());
  auto cases = fixture.Unwrap().at("cases").cast<tvm::ffi::json::Array>();
  ASSERT_EQ(cases.size(), 32);
  std::ifstream template_file(fixture_path.parent_path() / "mlc_templates.json");
  ASSERT_TRUE(template_file.good());
  std::stringstream template_buffer;
  template_buffer << template_file.rdbuf();
  auto templates = json::ParseToJSONObjectWithResultReturn(template_buffer.str());
  ASSERT_TRUE(templates.IsOk());
  auto template_map = templates.Unwrap();
  for (const auto& value : cases) {
    auto item = value.cast<tvm::ffi::json::Object>();
    std::string name = item.at("template").cast<std::string>();
    SCOPED_TRACE(name + ":" + item.at("name").cast<std::string>());
    auto parsed_conv = Conversation::FromJSON(template_map.at(name).cast<tvm::ffi::json::Object>());
    ASSERT_TRUE(parsed_conv.IsOk());
    Conversation conv = parsed_conv.Unwrap();
    tvm::ffi::json::Object request_json;
    request_json.Set("messages", item.at("messages"));
    auto request = ChatCompletionRequest::FromJSON(tvm::ffi::json::Stringify(request_json));
    ASSERT_TRUE(request.IsOk());
    auto result = CreatePrompt(conv, request.Unwrap(), ModelConfig(), DLDevice{kDLCPU, 0});
    ASSERT_TRUE(result.IsOk());
    std::string text;
    for (const auto& part : result.Unwrap()) {
      auto node = part.as<TextDataNode>();
      ASSERT_NE(node, nullptr);
      text += node->text;
    }
    EXPECT_EQ(text, item.at("rendered").cast<std::string>());
    auto null_extra_json = request_json;
    null_extra_json.Set("extra_body", nullptr);
    auto null_extra = ChatCompletionRequest::FromJSON(tvm::ffi::json::Stringify(null_extra_json));
    ASSERT_TRUE(null_extra.IsOk());
    EXPECT_TRUE(CreatePrompt(conv, null_extra.Unwrap(), ModelConfig(), DLDevice{kDLCPU, 0}).IsOk());
    for (const char* choice : {"none", "auto"}) {
      auto choice_json = request_json;
      choice_json.Set("tool_choice", choice);
      auto rejected = ChatCompletionRequest::FromJSON(tvm::ffi::json::Stringify(choice_json));
      ASSERT_TRUE(rejected.IsOk());
      EXPECT_TRUE(
          CreatePrompt(conv, rejected.Unwrap(), ModelConfig(), DLDevice{kDLCPU, 0}).IsErr());
    }
    for (const char* option : {"enable_thinking", "reasoning_effort", "preserve_thinking"}) {
      auto rejected_json = request_json;
      rejected_json.Set(option, false);
      auto rejected = ChatCompletionRequest::FromJSON(tvm::ffi::json::Stringify(rejected_json));
      ASSERT_TRUE(rejected.IsOk());
      EXPECT_TRUE(
          CreatePrompt(conv, rejected.Unwrap(), ModelConfig(), DLDevice{kDLCPU, 0}).IsErr());
    }
  }
}

}  // namespace json_ffi
}  // namespace llm
}  // namespace mlc
