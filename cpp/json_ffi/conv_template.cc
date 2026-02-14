#include "conv_template.h"

#include <tvm/ffi/function.h>

#include "../support/json_parser.h"
#include "image_utils.h"

namespace mlc {
namespace llm {
namespace json_ffi {

using namespace mlc::llm;

/****************** Model vision config ******************/

ModelVisionConfig ModelVisionConfig::FromJSON(const tvm::ffi::json::Object& json_obj) {
  ModelVisionConfig config;

  Result<int64_t> hidden_size_res = json::LookupWithResultReturn<int64_t>(json_obj, "hidden_size");
  if (hidden_size_res.IsOk()) {
    config.hidden_size = static_cast<int>(hidden_size_res.Unwrap());
  }

  Result<int64_t> image_size_res = json::LookupWithResultReturn<int64_t>(json_obj, "image_size");
  if (image_size_res.IsOk()) {
    config.image_size = static_cast<int>(image_size_res.Unwrap());
  }

  Result<int64_t> intermediate_size_res =
      json::LookupWithResultReturn<int64_t>(json_obj, "intermediate_size");
  if (intermediate_size_res.IsOk()) {
    config.intermediate_size = static_cast<int>(intermediate_size_res.Unwrap());
  }

  Result<int64_t> num_attention_heads_res =
      json::LookupWithResultReturn<int64_t>(json_obj, "num_attention_heads");
  if (num_attention_heads_res.IsOk()) {
    config.num_attention_heads = static_cast<int>(num_attention_heads_res.Unwrap());
  }

  Result<int64_t> num_hidden_layers_res =
      json::LookupWithResultReturn<int64_t>(json_obj, "num_hidden_layers");
  if (num_hidden_layers_res.IsOk()) {
    config.num_hidden_layers = static_cast<int>(num_hidden_layers_res.Unwrap());
  }

  Result<int64_t> patch_size_res = json::LookupWithResultReturn<int64_t>(json_obj, "patch_size");
  if (patch_size_res.IsOk()) {
    config.patch_size = static_cast<int>(patch_size_res.Unwrap());
  }

  Result<int64_t> projection_dim_res =
      json::LookupWithResultReturn<int64_t>(json_obj, "projection_dim");
  if (projection_dim_res.IsOk()) {
    config.projection_dim = static_cast<int>(projection_dim_res.Unwrap());
  }

  Result<int64_t> vocab_size_res = json::LookupWithResultReturn<int64_t>(json_obj, "vocab_size");
  if (vocab_size_res.IsOk()) {
    config.vocab_size = static_cast<int>(vocab_size_res.Unwrap());
  }

  Result<std::string> dtype_res = json::LookupWithResultReturn<std::string>(json_obj, "dtype");
  if (dtype_res.IsOk()) {
    config.dtype = dtype_res.Unwrap();
  }

  Result<int64_t> num_channels_res =
      json::LookupWithResultReturn<int64_t>(json_obj, "num_channels");
  if (num_channels_res.IsOk()) {
    config.num_channels = static_cast<int>(num_channels_res.Unwrap());
  }

  Result<double> layer_norm_eps_res =
      json::LookupWithResultReturn<double>(json_obj, "layer_norm_eps");
  if (layer_norm_eps_res.IsOk()) {
    config.layer_norm_eps = layer_norm_eps_res.Unwrap();
  }

  return config;
}

/****************** Model config ******************/

ModelConfig ModelConfig::FromJSON(const tvm::ffi::json::Object& json_obj) {
  ModelConfig config;

  Result<int64_t> vocab_size_res = json::LookupWithResultReturn<int64_t>(json_obj, "vocab_size");
  if (vocab_size_res.IsOk()) {
    config.vocab_size = static_cast<int>(vocab_size_res.Unwrap());
  }

  Result<int64_t> context_window_size_res =
      json::LookupWithResultReturn<int64_t>(json_obj, "context_window_size");
  if (context_window_size_res.IsOk()) {
    config.context_window_size = static_cast<int>(context_window_size_res.Unwrap());
  }

  Result<int64_t> sliding_window_size_res =
      json::LookupWithResultReturn<int64_t>(json_obj, "sliding_window_size");
  if (sliding_window_size_res.IsOk()) {
    config.sliding_window_size = static_cast<int>(sliding_window_size_res.Unwrap());
  }

  Result<int64_t> prefill_chunk_size_res =
      json::LookupWithResultReturn<int64_t>(json_obj, "prefill_chunk_size");
  if (prefill_chunk_size_res.IsOk()) {
    config.prefill_chunk_size = static_cast<int>(prefill_chunk_size_res.Unwrap());
  }

  Result<int64_t> tensor_parallel_shards_res =
      json::LookupWithResultReturn<int64_t>(json_obj, "tensor_parallel_shards");
  if (tensor_parallel_shards_res.IsOk()) {
    config.tensor_parallel_shards = static_cast<int>(tensor_parallel_shards_res.Unwrap());
  }

  Result<int64_t> pipeline_parallel_stages_res =
      json::LookupWithResultReturn<int64_t>(json_obj, "pipeline_parallel_stages");
  if (pipeline_parallel_stages_res.IsOk()) {
    config.pipeline_parallel_stages = static_cast<int>(pipeline_parallel_stages_res.Unwrap());
  }

  Result<int64_t> max_batch_size_res =
      json::LookupWithResultReturn<int64_t>(json_obj, "max_batch_size");
  if (max_batch_size_res.IsOk()) {
    config.max_batch_size = static_cast<int>(max_batch_size_res.Unwrap());
  }

  if (json_obj.count("vision_config")) {
    const tvm::ffi::json::Object& vision_config_obj =
        json_obj.at("vision_config").cast<tvm::ffi::json::Object>();
    config.vision_config = ModelVisionConfig::FromJSON(vision_config_obj);
  }

  return config;
}

/****************** Conversation template ******************/

std::unordered_map<MessagePlaceholders, std::string> PLACEHOLDERS = {
    {MessagePlaceholders::SYSTEM, "{system_message}"},
    {MessagePlaceholders::USER, "{user_message}"},
    {MessagePlaceholders::ASSISTANT, "{assistant_message}"},
    {MessagePlaceholders::TOOL, "{tool_message}"},
    {MessagePlaceholders::FUNCTION, "{function_string}"}};

MessagePlaceholders MessagePlaceholderFromString(const std::string& role) {
  static const std::unordered_map<std::string, MessagePlaceholders> enum_map = {
      {"system", MessagePlaceholders::SYSTEM},       {"user", MessagePlaceholders::USER},
      {"assistant", MessagePlaceholders::ASSISTANT}, {"tool", MessagePlaceholders::TOOL},
      {"function", MessagePlaceholders::FUNCTION},
  };

  return enum_map.at(role);
}

Conversation::Conversation()
    : role_templates({{"user", PLACEHOLDERS[MessagePlaceholders::USER]},
                      {"assistant", PLACEHOLDERS[MessagePlaceholders::ASSISTANT]},
                      {"tool", PLACEHOLDERS[MessagePlaceholders::TOOL]}}) {}

std::string Conversation::GetSystemText(const std::string& system_msg) const {
  std::string system_text = this->system_template;
  static std::string system_placeholder = PLACEHOLDERS[MessagePlaceholders::SYSTEM];
  size_t pos = system_text.find(system_placeholder);
  if (pos != std::string::npos) {
    system_text.replace(pos, system_placeholder.length(), system_msg);
  }
  return system_text;
}

std::string Conversation::GetRoleText(const std::string& role, const std::string& content,
                                      const std::optional<std::string>& fn_call_string) const {
  std::string role_text = this->role_templates.at(role);
  std::string placeholder = PLACEHOLDERS[MessagePlaceholderFromString(role)];
  size_t pos = role_text.find(placeholder);
  if (pos != std::string::npos) {
    role_text.replace(pos, placeholder.length(), content);
  }
  if (fn_call_string) {
    // replace placeholder[FUNCTION] with function_string
    // this assumes function calling is used for a single request scenario only
    pos = role_text.find(PLACEHOLDERS[MessagePlaceholders::FUNCTION]);
    if (pos != std::string::npos) {
      role_text.replace(pos, PLACEHOLDERS[MessagePlaceholders::FUNCTION].length(),
                        fn_call_string.value());
    }
  }
  return role_text;
}

/// Try to detect if function calling is needed, if so, return the function calling string
Result<std::optional<std::string>> TryGetFunctionCallingString(
    const Conversation& conv, const ChatCompletionRequest& request) {
  using TResult = Result<std::optional<std::string>>;
  if (!request.tools.has_value() ||
      (request.tool_choice.has_value() && request.tool_choice.value() == "none")) {
    return TResult::Ok(std::nullopt);
  }
  std::vector<ChatTool> tools_ = request.tools.value();
  std::string tool_choice_ = request.tool_choice.value();

  // TODO: support with tool choice as dict
  for (const auto& tool : tools_) {
    if (tool.function.name == tool_choice_) {
      tvm::ffi::json::Value function_str(tool.function.AsJSON());
      return TResult::Ok(tvm::ffi::json::Stringify(function_str));
    }
  }

  if (tool_choice_ != "auto") {
    return TResult::Error("Invalid tool_choice value in the request: " + tool_choice_);
  }

  tvm::ffi::json::Array function_list;
  for (const auto& tool : tools_) {
    function_list.push_back(tool.function.AsJSON());
  }

  tvm::ffi::json::Value function_list_json(function_list);
  return TResult::Ok(tvm::ffi::json::Stringify(function_list_json));
};

Result<std::vector<Data>> CreatePrompt(const Conversation& conv,
                                       const ChatCompletionRequest& request,
                                       const ModelConfig& config, DLDevice device) {
  using TResult = Result<std::vector<Data>>;

  Result<std::optional<std::string>> fn_call_str_tmp = TryGetFunctionCallingString(conv, request);
  if (fn_call_str_tmp.IsErr()) {
    return TResult::Error(fn_call_str_tmp.UnwrapErr());
  }
  std::optional<std::string> fn_call_string = fn_call_str_tmp.Unwrap();

  // Handle system message
  // concz
  bool has_custom_system = false;
  std::string custom_system_inputs;

  auto f_populate_system_message = [&](const std::vector<ChatCompletionMessage>& msg_vec) {
    for (ChatCompletionMessage msg : msg_vec) {
      if (msg.role == "system") {
        ICHECK(msg.content.IsText()) << "System message must be text";
        custom_system_inputs += msg.content.Text();
        has_custom_system = true;
      }
    }
  };
  // go through messages in template and passed in.
  f_populate_system_message(conv.messages);
  f_populate_system_message(request.messages);

  // pending text records the text to be put into data
  // we lazily accumulate the pending text
  // to reduce amount of segments in the Data vector
  std::string pending_text =
      conv.GetSystemText(has_custom_system ? custom_system_inputs : conv.system_message);

  // Get the message strings
  std::vector<Data> message_list;
  size_t non_system_msg_count = 0;

  // returns error if error happens
  auto f_process_messages =
      [&](const std::vector<ChatCompletionMessage>& msg_vec) -> std::optional<TResult> {
    for (size_t i = 0; i < msg_vec.size(); ++i) {
      const ChatCompletionMessage& msg = msg_vec[i];
      // skip system message as it is already processed
      if (msg.role == "system") continue;

      auto role_it = conv.roles.find(msg.role);
      if (role_it == conv.roles.end()) {
        return TResult::Error("Role \"" + msg.role + "\" is not supported");
      }
      const std::string& role_name = role_it->second;
      // skip when content is empty
      if (msg.content.IsNull()) {
        pending_text += role_name + conv.role_empty_sep;
        continue;
      }
      ++non_system_msg_count;
      // assistant uses conv.seps[1] if there are two seps
      int sep_offset = msg.role == "assistant" ? 1 : 0;
      const std::string& seperator = conv.seps[sep_offset % conv.seps.size()];
      // setup role prefix
      std::string role_prefix = "";
      // Do not append role prefix if this is the first message and there is already a system
      // message
      if (conv.add_role_after_system_message || pending_text.empty() || non_system_msg_count != 1) {
        role_prefix = role_name + conv.role_content_sep;
      }
      pending_text += role_prefix;

      if (msg.content.IsParts()) {
        for (const auto& item : msg.content.Parts()) {
          auto it_type = item.find("type");
          if (it_type == item.end()) {
            return TResult::Error("The content of a message does not have \"type\" field");
          }
          if (it_type->second == "text") {
            auto it_text = item.find("text");
            if (it_text == item.end()) {
              return TResult::Error(
                  "The text type content of a message does not have \"text\" field");
            }
            // replace placeholder[ROLE] with input message from role
            pending_text += conv.GetRoleText(msg.role, it_text->second, fn_call_string);
          } else if (it_type->second == "image_url") {
            if (item.find("image_url") == item.end()) {
              return TResult::Error("Content should have an image_url field");
            }
            std::string image_url =
                item.at("image_url");  // TODO(mlc-team): According to OpenAI API reference this
                                       // should be a map, with a "url" key containing the URL, but
                                       // we are just assuming this as the URL for now
            std::string base64_image = image_url.substr(image_url.find(",") + 1);
            Result<Tensor> image_data_res = LoadImageFromBase64(base64_image);
            if (image_data_res.IsErr()) {
              return TResult::Error(image_data_res.UnwrapErr());
            }
            if (!config.vision_config.has_value()) {
              return TResult::Error("Vision config is required for image input");
            }
            int image_size = config.vision_config.value().image_size;
            int patch_size = config.vision_config.value().patch_size;

            int embed_size = (image_size * image_size) / (patch_size * patch_size);

            Tensor image_data = image_data_res.Unwrap();
            std::vector<int64_t> new_shape = {1, image_size, image_size, 3};
            Tensor image_tensor = image_data.CreateView(new_shape, image_data.DataType());
            // TODO: Not sure if commenting will affect other functions. But
            // python part will do clip preprocessing. auto image_tensor =
            // ClipPreprocessor(image_data_res.Unwrap(), image_size, device);
            // lazily commit text data
            if (pending_text.length() != 0) {
              message_list.push_back(TextData(pending_text));
              pending_text = "";
            }
            message_list.push_back(ImageData(image_tensor, embed_size));
          } else {
            return TResult::Error("Unsupported content type: " + it_type->second);
          }
        }
      } else {
        ICHECK(msg.content.IsText());
        pending_text += conv.GetRoleText(msg.role, msg.content.Text(), fn_call_string);
      }
      pending_text += seperator;
    }
    return std::nullopt;
  };

  if (auto err = f_process_messages(conv.messages)) {
    return err.value();
  }
  if (auto err = f_process_messages(request.messages)) {
    return err.value();
  }
  // append last assistant begin message
  ChatCompletionMessage last_assistant_begin;
  last_assistant_begin.role = "assistant";
  last_assistant_begin.content = std::nullopt;
  if (auto err = f_process_messages({last_assistant_begin})) {
    return err.value();
  }
  if (pending_text.length() != 0) {
    message_list.push_back(TextData(pending_text));
  }
  // Handle system_prefix_token_ids
  if (conv.system_prefix_token_ids.has_value()) {
    message_list.insert(message_list.begin(), TokenData(conv.system_prefix_token_ids.value()));
  }
  return TResult::Ok(message_list);
}

Result<Conversation> Conversation::FromJSON(const tvm::ffi::json::Object& json_obj) {
  using TResult = Result<Conversation>;
  Conversation conv;

  Result<std::optional<std::string>> name_res =
      json::LookupOptionalWithResultReturn<std::string>(json_obj, "name");
  if (name_res.IsErr()) {
    return TResult::Error(name_res.UnwrapErr());
  }
  conv.name = name_res.Unwrap();

  Result<std::string> system_template_res =
      json::LookupWithResultReturn<std::string>(json_obj, "system_template");
  if (system_template_res.IsErr()) {
    return TResult::Error(system_template_res.UnwrapErr());
  }
  conv.system_template = system_template_res.Unwrap();

  Result<std::string> system_message_res =
      json::LookupWithResultReturn<std::string>(json_obj, "system_message");
  if (system_message_res.IsErr()) {
    return TResult::Error(system_message_res.UnwrapErr());
  }
  conv.system_message = system_message_res.Unwrap();

  Result<std::optional<tvm::ffi::json::Array>> system_prefix_token_ids_arr_res =
      json::LookupOptionalWithResultReturn<tvm::ffi::json::Array>(json_obj,
                                                                  "system_prefix_token_ids");
  if (system_prefix_token_ids_arr_res.IsErr()) {
    return TResult::Error(system_prefix_token_ids_arr_res.UnwrapErr());
  }
  std::optional<tvm::ffi::json::Array> system_prefix_token_ids_arr =
      system_prefix_token_ids_arr_res.Unwrap();
  if (system_prefix_token_ids_arr.has_value()) {
    std::vector<int> system_prefix_token_ids;
    system_prefix_token_ids.reserve(system_prefix_token_ids_arr.value().size());
    for (const auto& token_id : system_prefix_token_ids_arr.value()) {
      if (!token_id.try_cast<int64_t>().has_value()) {
        return TResult::Error("A system prefix token id is not integer.");
      }
      system_prefix_token_ids.push_back(static_cast<int>(token_id.cast<int64_t>()));
    }
    conv.system_prefix_token_ids = std::move(system_prefix_token_ids);
  }

  Result<bool> add_role_after_system_message_res =
      json::LookupWithResultReturn<bool>(json_obj, "add_role_after_system_message");
  if (add_role_after_system_message_res.IsErr()) {
    return TResult::Error(add_role_after_system_message_res.UnwrapErr());
  }
  conv.add_role_after_system_message = add_role_after_system_message_res.Unwrap();

  Result<tvm::ffi::json::Object> roles_object_res =
      json::LookupWithResultReturn<tvm::ffi::json::Object>(json_obj, "roles");
  if (roles_object_res.IsErr()) {
    return TResult::Error(roles_object_res.UnwrapErr());
  }
  for (const auto& role : roles_object_res.Unwrap()) {
    if (!role.second.try_cast<std::string>().has_value()) {
      return TResult::Error("A role value in the conversation template is not a string.");
    }
    conv.roles[role.first.cast<tvm::ffi::String>()] = role.second.cast<std::string>();
  }

  Result<std::optional<tvm::ffi::json::Object>> role_templates_object_res =
      json::LookupOptionalWithResultReturn<tvm::ffi::json::Object>(json_obj, "role_templates");
  if (role_templates_object_res.IsErr()) {
    return TResult::Error(role_templates_object_res.UnwrapErr());
  }
  std::optional<tvm::ffi::json::Object> role_templates_object = role_templates_object_res.Unwrap();
  if (role_templates_object.has_value()) {
    for (const auto& [role, msg] : role_templates_object.value()) {
      if (!msg.try_cast<std::string>().has_value()) {
        return TResult::Error("A value in \"role_templates\" is not a string.");
      }
      conv.role_templates[role.cast<tvm::ffi::String>()] = msg.cast<std::string>();
    }
  }

  Result<tvm::ffi::json::Array> messages_arr_res =
      json::LookupWithResultReturn<tvm::ffi::json::Array>(json_obj, "messages");
  if (messages_arr_res.IsErr()) {
    return TResult::Error(messages_arr_res.UnwrapErr());
  }
  for (const auto& message : messages_arr_res.Unwrap()) {
    if (!message.try_cast<tvm::ffi::json::Array>().has_value() ||
        message.cast<tvm::ffi::json::Array>().size() != 2) {
      return TResult::Error(
          "A message in the conversation template is not an array of [role, content].");
    }
    tvm::ffi::json::Array message_arr = message.cast<tvm::ffi::json::Array>();
    if (!message_arr[0].try_cast<std::string>().has_value()) {
      return TResult::Error("The role of a message in the conversation template is not a string.");
    }
    std::string role = message_arr[0].cast<std::string>();
    // content can be a string or an array of objects
    if (message_arr[1].try_cast<std::string>().has_value()) {
      ChatCompletionMessage msg;
      msg.role = role;
      msg.content = message_arr[1].cast<std::string>();
      conv.messages.push_back(msg);
      continue;
    } else if (message_arr[1].try_cast<tvm::ffi::json::Array>().has_value()) {
      tvm::ffi::json::Array content_arr = message_arr[1].cast<tvm::ffi::json::Array>();
      std::vector<std::unordered_map<std::string, std::string>> content;
      content.reserve(content_arr.size());
      for (const auto& item : content_arr) {
        if (!item.try_cast<tvm::ffi::json::Object>().has_value()) {
          return TResult::Error("The content of conversation template message is not an object");
        }
        std::unordered_map<std::string, std::string> item_map;
        for (const auto& [key, value] : item.cast<tvm::ffi::json::Object>()) {
          item_map[key.cast<tvm::ffi::String>()] = tvm::ffi::json::Stringify(value);
        }
        content.push_back(std::move(item_map));
      }
      ChatCompletionMessage msg;
      msg.role = role;
      msg.content = content;
      conv.messages.push_back(msg);
      continue;
    } else {
      return TResult::Error(
          "The content of a message in the conversation template is not a string or an array.");
    }
  }

  Result<tvm::ffi::json::Array> seps_arr_res =
      json::LookupWithResultReturn<tvm::ffi::json::Array>(json_obj, "seps");
  if (seps_arr_res.IsErr()) {
    return TResult::Error(seps_arr_res.UnwrapErr());
  }
  std::vector<std::string> seps;
  for (const auto& sep : seps_arr_res.Unwrap()) {
    if (!sep.try_cast<std::string>().has_value()) {
      return TResult::Error("A separator (\"seps\") of the conversation template is not a string");
    }
    conv.seps.push_back(sep.cast<std::string>());
  }

  Result<std::string> role_content_sep_res =
      json::LookupWithResultReturn<std::string>(json_obj, "role_content_sep");
  if (role_content_sep_res.IsErr()) {
    return TResult::Error(role_content_sep_res.UnwrapErr());
  }
  conv.role_content_sep = role_content_sep_res.Unwrap();

  Result<std::string> role_empty_sep_res =
      json::LookupWithResultReturn<std::string>(json_obj, "role_empty_sep");
  if (role_empty_sep_res.IsErr()) {
    return TResult::Error(role_empty_sep_res.UnwrapErr());
  }
  conv.role_empty_sep = role_empty_sep_res.Unwrap();

  Result<tvm::ffi::json::Array> stop_str_arr_res =
      json::LookupWithResultReturn<tvm::ffi::json::Array>(json_obj, "stop_str");
  if (stop_str_arr_res.IsErr()) {
    return TResult::Error(stop_str_arr_res.UnwrapErr());
  }
  for (const auto& stop : stop_str_arr_res.Unwrap()) {
    if (!stop.try_cast<std::string>().has_value()) {
      return TResult::Error(
          "A stop string (\"stop_str\") of the conversation template is not a string.");
    }
    conv.stop_str.push_back(stop.cast<std::string>());
  }

  Result<tvm::ffi::json::Array> stop_token_ids_arr_res =
      json::LookupWithResultReturn<tvm::ffi::json::Array>(json_obj, "stop_token_ids");
  if (stop_token_ids_arr_res.IsErr()) {
    return TResult::Error(stop_token_ids_arr_res.UnwrapErr());
  }
  for (const auto& stop : stop_token_ids_arr_res.Unwrap()) {
    if (!stop.try_cast<int64_t>().has_value()) {
      return TResult::Error(
          "A stop token id (\"stop_token_ids\") of the conversation template is not an integer.");
    }
    conv.stop_token_ids.push_back(static_cast<int>(stop.cast<int64_t>()));
  }
  return TResult::Ok(conv);
}

Result<Conversation> Conversation::FromJSON(const std::string& json_str) {
  Result<tvm::ffi::json::Object> json_obj = json::ParseToJSONObjectWithResultReturn(json_str);
  if (json_obj.IsErr()) {
    return Result<Conversation>::Error(json_obj.UnwrapErr());
  }
  return Conversation::FromJSON(json_obj.Unwrap());
}

}  // namespace json_ffi
}  // namespace llm
}  // namespace mlc
