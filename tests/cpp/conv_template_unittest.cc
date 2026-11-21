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

TEST(JsonFFIConvTest, Gemma4LargePlainText) {
  const auto path = std::filesystem::path(__FILE__).parent_path().parent_path() /
                    "fixtures/gemma4/large_text_chat.json";
  std::ifstream input(path);
  ASSERT_TRUE(input.good());
  std::string fixture((std::istreambuf_iterator<char>(input)), {});
  auto parsed = json::ParseToJSONObjectWithResultReturn(fixture);
  ASSERT_TRUE(parsed.IsOk());
  auto cases = parsed.Unwrap().at("cases").cast<tvm::ffi::json::Array>();
  ASSERT_GT(cases.size(), 0);

  for (const auto& value : cases) {
    auto item = value.cast<tvm::ffi::json::Object>();
    SCOPED_TRACE(item.at("name").cast<std::string>());
    bool thinking = item.at("thinking").cast<bool>();
    Conversation conv;
    conv.reasoning_history_style = "gemma4";
    conv.system_template = std::string("<|turn>system\n") + (thinking ? "<|think|>\n" : "") +
                           "{system_message}<turn|>\n";
    conv.render_empty_system_message = thinking;
    conv.roles = {{"user", "<|turn>user"}, {"assistant", "<|turn>model"}};
    conv.role_content_sep = "\n";
    conv.role_empty_sep = thinking ? "\n" : "\n<|channel>thought\n<channel|>";
    conv.seps = {"<turn|>\n"};
    conv.system_prefix_token_ids = std::vector<int>{2};
    auto request = ChatCompletionRequest::FromJSON(
        "{\"messages\":" + tvm::ffi::json::Stringify(item.at("messages")) + "}");
    ASSERT_TRUE(request.IsOk());
    auto result = CreatePrompt(conv, request.Unwrap(), ModelConfig(), DLDevice{kDLCPU, 0});
    ASSERT_TRUE(result.IsOk());
    auto prompt = result.Unwrap();
    ASSERT_EQ(prompt.size(), 2);
    const auto* tokens = prompt[0].as<serve::TokenDataNode>();
    ASSERT_NE(tokens, nullptr);
    ASSERT_EQ(tokens->token_ids.size(), 1);
    EXPECT_EQ(tokens->token_ids[0], 2);
    ASSERT_NE(prompt[1].as<serve::TextDataNode>(), nullptr);
    EXPECT_EQ(prompt[1].as<serve::TextDataNode>()->text, item.at("prompt").cast<std::string>());
  }
}

TEST(JsonFFIConvTest, Lfm3PlainTextHistory) {
  auto parsed = Conversation::FromJSON(R"({
    "name":"lfm3", "add_role_after_system_message":true, "system_template":"<|im_start|>system\n{system_message}<|im_end|>\n",
    "system_message":"", "render_empty_system_message":false,
    "roles":{"user":"<|im_start|>user","assistant":"<|im_start|>assistant"},
    "messages":[], "seps":["<|im_end|>\n"], "role_content_sep":"\n",
    "role_empty_sep":"\n<think>", "stop_str":["<|im_end|>"],
    "stop_token_ids":[124900], "reasoning_history_style":"lfm3"
  })");
  ASSERT_TRUE(parsed.IsOk()) << (parsed.IsErr() ? parsed.UnwrapErr() : "");
  auto conv = parsed.Unwrap();
  ASSERT_EQ(conv.reasoning_history_style, "lfm3");
  ASSERT_EQ(conv.GetSystemText(""), "");
  ASSERT_EQ(conv.GetSystemText("brief"), "<|im_start|>system\nbrief<|im_end|>\n");
  auto message = [](const std::string& role, const std::string& text) {
    ChatCompletionMessage msg;
    msg.role = role;
    msg.content = ChatCompletionMessageContent(text);
    return msg;
  };
  for (const auto& item : std::vector<std::pair<std::string, std::string>>{
           {"answer", "answer"},
           {"  answer  ", "  answer  "},
           {"<think>secret</think> \tanswer\n ", "answer"},
           {"x</think>y</think>\u2003answer\u00a0", "answer"},
           {"<think>x</think>\t\n", ""},
           {"", ""}}) {
    ChatCompletionRequest request;
    // Exercise normalization across both stored and request histories.
    conv.messages = {message("user", "first"), message("assistant", item.first)};
    request.messages = {message("user", "next")};
    auto result = CreatePrompt(conv, request, ModelConfig(), DLDevice{kDLCPU, 0});
    ASSERT_TRUE(result.IsOk());
    std::string text;
    for (const auto& part : result.Unwrap()) {
      const auto* node = part.as<TextDataNode>();
      ASSERT_NE(node, nullptr);
      text += node->text;
    }
    ASSERT_EQ(text,
              "<|im_start|>user\nfirst<|im_end|>\n"
              "<|im_start|>assistant\n<think></think>" +
                  item.second +
                  "<|im_end|>\n<|im_start|>user\nnext<|im_end|>\n"
                  "<|im_start|>assistant\n<think>");
    ASSERT_EQ(conv.messages[1].content.Text(), item.first);
  }
  conv.messages.clear();
  conv.role_empty_sep = "\n<think></think>";
  ChatCompletionRequest single_turn;
  single_turn.messages = {message("user", "next")};
  auto no_think = CreatePrompt(conv, single_turn, ModelConfig(), DLDevice{kDLCPU, 0});
  ASSERT_TRUE(no_think.IsOk());
  ASSERT_EQ(no_think.Unwrap().back().as<TextDataNode>()->text,
            "<|im_start|>user\nnext<|im_end|>\n<|im_start|>assistant\n<think></think>");
  single_turn.tools = std::vector<ChatTool>();
  ASSERT_TRUE(CreatePrompt(conv, single_turn, ModelConfig(), DLDevice{kDLCPU, 0}).IsErr());
  conv.render_empty_system_message = true;
  ASSERT_EQ(conv.GetSystemText(""), "<|im_start|>system\n<|im_end|>\n");
  ChatCompletionRequest continuation;
  continuation.messages = {message("assistant", "partial")};
  ASSERT_TRUE(CreatePrompt(conv, continuation, ModelConfig(), DLDevice{kDLCPU, 0}).IsErr());
}

void _TestQwenPlainTextHistory(const std::string& family) {
  const auto fixture_path = std::filesystem::path(__FILE__).parent_path().parent_path() /
                            "python/protocol/fixtures" / family / "hf.json";
  std::ifstream file(fixture_path);
  ASSERT_TRUE(file.good());
  std::stringstream buffer;
  buffer << file.rdbuf();
  auto fixture = json::ParseToJSONObjectWithResultReturn(buffer.str());
  ASSERT_TRUE(fixture.IsOk());
  auto cases = fixture.Unwrap().at("cases").cast<tvm::ffi::json::Array>();
  ASSERT_GT(cases.size(), 0);
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
    auto conv = parsed_conv.Unwrap();
    tvm::ffi::json::Object request_json;
    request_json.Set("messages", item.at("messages"));
    auto request = ChatCompletionRequest::FromJSON(tvm::ffi::json::Stringify(request_json));
    ASSERT_TRUE(request.IsOk());
    auto result = CreatePrompt(conv, request.Unwrap(), ModelConfig(), DLDevice{kDLCPU, 0});
    ASSERT_TRUE(result.IsOk());
    std::string text;
    for (const auto& part : result.Unwrap()) {
      const auto* node = part.as<TextDataNode>();
      ASSERT_NE(node, nullptr);
      text += node->text;
    }
    EXPECT_EQ(text, item.at("rendered").cast<std::string>());
  }

  auto parsed_conv = Conversation::FromJSON(template_map.at(family).cast<tvm::ffi::json::Object>());
  ASSERT_TRUE(parsed_conv.IsOk());
  auto conv = parsed_conv.Unwrap();
  tvm::ffi::json::Object request_json;
  request_json.Set("messages", cases[0].cast<tvm::ffi::json::Object>().at("messages"));
  request_json.Set("extra_body", nullptr);
  auto request = ChatCompletionRequest::FromJSON(tvm::ffi::json::Stringify(request_json));
  ASSERT_TRUE(request.IsOk());
  EXPECT_TRUE(CreatePrompt(conv, request.Unwrap(), ModelConfig(), DLDevice{kDLCPU, 0}).IsOk());
  for (const char* choice : {"none", "auto"}) {
    auto choice_json = request_json;
    choice_json.Set("tool_choice", choice);
    auto rejected = ChatCompletionRequest::FromJSON(tvm::ffi::json::Stringify(choice_json));
    ASSERT_TRUE(rejected.IsOk());
    EXPECT_TRUE(CreatePrompt(conv, rejected.Unwrap(), ModelConfig(), DLDevice{kDLCPU, 0}).IsErr());
  }
  for (const char* option : {"enable_thinking", "reasoning_effort", "preserve_thinking"}) {
    auto rejected_json = request_json;
    rejected_json.Set(option, false);
    auto rejected = ChatCompletionRequest::FromJSON(tvm::ffi::json::Stringify(rejected_json));
    ASSERT_TRUE(rejected.IsOk());
    EXPECT_TRUE(CreatePrompt(conv, rejected.Unwrap(), ModelConfig(), DLDevice{kDLCPU, 0}).IsErr());
  }
}

TEST(JsonFFIConvTest, Qwen38PinnedPlainTextHistory) { _TestQwenPlainTextHistory("qwen3_8"); }
TEST(JsonFFIConvTest, Qwen36PinnedPlainTextHistory) { _TestQwenPlainTextHistory("qwen3_6"); }

}  // namespace json_ffi
}  // namespace llm
}  // namespace mlc
