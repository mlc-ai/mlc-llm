use serde::{Deserialize, Serialize};

/// A struct that represents user-defined partial configuration for conversation template.
///
/// This can be passed in to the instantiation of a [ChatModule](crate::chat_module::ChatModule)
/// instance to override the default setting in `mlc-chat-config.json` under the
/// model folder. Note that we will first load the predefined template
/// with the name specified in `conv_template`.
///
/// Since the configuration is partial, everything will be optional.
#[derive(Clone, Default, Builder, Debug, Serialize, Deserialize)]
#[builder(default)]
pub struct ConvConfig {
    /// Token list prefixing the conversation.
    prefix_tokens: Option<Vec<i32>>,

    /// Name of the conversation.
    name: Option<String>,

    /// The prompt encoded before starting the chat.
    system: Option<String>,

    /// An array that describes the role names of the user and the model.
    roles: Option<Vec<String>>,

    /// The chat history represented as an array of string pairs.
    pub messages: Option<Vec<Vec<String>>>,

    /// The offset used to begin the chat from the chat history.
    pub offset: Option<usize>,

    /// Specifies whether we are in chat-bot mode (`0`) or pure LM prompt mode (`1`).
    separator_style: Option<i32>,

    /// An array of strings indicating the separators to be used after a user message and a model message respectively.
    seps: Option<Vec<String>>,

    /// A string indicating the separator between a role and a message.
    role_msg_sep: Option<String>,

    /// A string indicating the separator to append to a role when there is no message yet.
    role_empty_sep: Option<String>,

    /// When the `stop_str` is encountered, the model will stop generating output.
    stop_str: Option<String>,

    /// A list of token IDs that act as stop tokens.
    stop_tokens: Option<Vec<i32>>,

    /// Determines whether a beginning-of-string (bos) token should be added before the input tokens.
    add_bos: Option<bool>,
}

impl ConvConfig {
    pub fn post_init(&mut self) {
        if let Some(messages) = &self.messages {
            if self.offset.is_none() {
                self.offset = Some(messages.len());
            }
        }
    }
}

/// A struct that represents user-defined partial configuration for the chat config file.
///
/// An instance of [ChatConfig] can be passed in to override the default setting.
/// Since the configuration is partial, everything will be optional.
///
/// Note: This struct is used to represent the chat config during intermediate processing.
#[derive(Builder, Debug, Default, Serialize, Deserialize)]
#[builder(default)]
pub struct ChatConfig {
    /// The necessary model library to launch this model architecture.
    /// Recommended to reuse model library when possible.
    pub model_lib: Option<String>,

    /// Uniquely identifying the model in application. Also used by
    /// CLI to specify which model to run.
    pub local_id: Option<String>,

    /// The name of the conversation template that this chat uses.
    pub conv_template: Option<String>,

    /// Temperature applied to logits before sampling. Encourages diverse outputs if higher.
    pub temperature: Option<f32>,

    /// Controls the likelihood of the model generating repeated texts.
    /// See the CTRL paper for more details: <https://arxiv.org/pdf/1909.05858.pdf>
    repetition_penalty: Option<f32>,

    /// Determines the set of tokens from which we sample during decoding.
    /// More info on top-p sampling: <https://huggingface.co/blog/how-to-generate#top-p-nucleus-sampling>
    top_p: Option<f32>,

    /// Approximated average number of generated tokens in each round.
    mean_gen_len: Option<usize>,

    /// Maximum number of tokens to be generated in each round.
    max_gen_len: Option<usize>,

    /// Fraction of maximum window size to shift when it is exceeded.
    shift_fill_factor: Option<f32>,

    /// List of tokenizer files of the model.
    tokenizer_files: Option<Vec<String>>,

    /// Partial overriding configuration for conversation template.
    pub conv_config: Option<ConvConfig>,

    /// The category of the model's architecture (e.g. `llama`, `gpt_neox`, `rwkv`).
    model_category: Option<String>,

    /// Name of the model (e.g. `Llama-2-7b-chat-hf`).
    model_name: Option<String>,

    /// Tensor parallel degree.
    num_shards: Option<usize>,

    /// Maximum kv cache window size.
    max_window_size: Option<usize>,
}

impl ChatConfig {
    pub fn from_json(json_str: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json_str)
    }
}

/// A struct that represents user-defined generation configuration.
///
/// An instance of [GenerationConfig] can be passed into the
/// [ChatModule::generate](crate::chat_module::ChatModule::generate) function
/// to override the default generation settings specified in `mlc-chat-config.json`
/// and `ChatConfig` under the model folder.
///
/// Once the generation ends, `GenerationConfig` is discarded, as the values
/// are only intended to override the `ChatConfig` generation settings during a
/// single generation, unless it is recurrently passed to the `generate` function.
/// This allows for changing generation settings over time, without permanently
/// overriding the `ChatConfig`.
///
/// Since the configuration is partial, all fields are optional.
#[derive(Builder, Debug, Default, Serialize, Deserialize)]
#[builder(default)]
pub struct GenerationConfig {
    /// The temperature applied to logits before sampling. The default value is
    /// `0.7`. A higher temperature encourages more diverse outputs, while a
    /// lower temperature produces more deterministic outputs.
    temperature: Option<f32>,

    /// The repetition penalty controls the likelihood of the model generating
    /// repeated texts. The default value is set to `1.0`, indicating that no
    /// repetition penalty is applied. Increasing the value reduces the
    /// likelihood of repeat text generation. However, setting a high
    /// `repetition_penalty` may result in the model generating meaningless
    /// texts. The ideal choice of repetition penalty may vary among models. Only
    /// Active when presence_penalty and frequency_penalty are both `0.0`.

    /// For more details on how repetition penalty controls text generation, please
    /// check out the CTRL paper <https://arxiv.org/pdf/1909.05858.pdf>.
    repetition_penalty: Option<f32>,

    /// This parameter determines the set of tokens from which we sample during
    /// decoding. The default value is set to `0.95`. At each step, we select
    /// tokens from the minimal set that has a cumulative probability exceeding
    /// the ``top_p` parameter.

    /// For additional information on top-p sampling, please refer to this blog
    /// post: <https://huggingface.co/blog/how-to-generate#top-p-nucleus-sampling>.
    top_p: Option<f32>,

    /// The approximated average number of generated tokens in each round. Used
    /// to determine whether the maximum window size would be exceeded.
    mean_gen_len: Option<usize>,

    /// This parameter determines the maximum length of the generated text. If it is
    /// not set, the model will generate text until it encounters a stop token.
    max_gen_len: Option<usize>,

    /// Number between `-2.0` and `2.0`. Positive values penalize new tokens based on
    /// whether they appear in the text so far, increasing the model's likelihood
    /// to talk about new topics. Negative values can increase the likelihood of
    /// repetition.
    presence_penalty: Option<f32>,

    /// Number between `-2.0` and `2.0`. Positive values penalize new tokens based on their
    /// existing frequency in the text so far, decreasing the model's likelihood to
    /// repeat the same line verbatim. Negative values can increase the likelihood of
    /// repetition.
    frequency_penalty: Option<f32>,

    /// This parameter determines the number of text samples to generate. The default
    /// value is `1`. Note that this parameter is only used when `stream` is set to
    /// `false`.
    pub n: Option<usize>,

    /// When `stop` is encountered, the model will stop generating output.
    /// It can be a string or a list of strings. If it is a list of strings, the model
    /// will stop generating output when any of the strings in the list is encountered.
    /// Note that this parameter does not override the default stop string of the model.
    stop: Option<Vec<String>>,
}

impl GenerationConfig {
    pub fn from_chat_config(chat_config: &ChatConfig) -> Self {
        Self {
            temperature: chat_config.temperature,
            repetition_penalty: chat_config.repetition_penalty,
            top_p: chat_config.top_p,
            mean_gen_len: chat_config.mean_gen_len,
            max_gen_len: chat_config.max_gen_len,
            presence_penalty: Some(0.0),
            frequency_penalty: Some(0.0),
            n: Some(0),
            stop: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv_config() {
        let mut config = ConvConfig {
            messages: Some(vec![vec!["User: Hi".to_string(), "Assistant: Hello".to_string()]]),
            offset: None,
            ..Default::default()
        };
        config.post_init();
        assert_eq!(config.offset, Some(1));
    }

    #[test]
    fn test_chat_config() {
        let json_data = r#"
        {
            "model_lib": "some_lib",
            "local_id": "id123",
            "temperature": 0.7
        }
        "#;

        let config = ChatConfig::from_json(json_data).unwrap();

        assert_eq!(config.model_lib, Some("some_lib".to_string()));
        assert_eq!(config.local_id, Some("id123".to_string()));
        assert_eq!(config.temperature, Some(0.7));
        let _pretty_json = serde_json::to_string_pretty(&config).unwrap();
    }

    #[test]
    fn test_generation_config() {
        let chat_config = ChatConfigBuilder::default()
            .temperature(Some(0.7))
            .top_p(Some(0.8))
            .mean_gen_len(Some(50))
            .max_gen_len(Some(75))
            .build()
            .unwrap();

        let gen_config = GenerationConfig::from_chat_config(&chat_config);

        assert_eq!(gen_config.temperature, chat_config.temperature);
        assert_eq!(gen_config.repetition_penalty, chat_config.repetition_penalty);
        assert_eq!(gen_config.top_p, chat_config.top_p);
        assert_eq!(gen_config.mean_gen_len, chat_config.mean_gen_len);
        assert_eq!(gen_config.max_gen_len, chat_config.max_gen_len);
        assert_eq!(gen_config.presence_penalty, Some(0.0));
        assert_eq!(gen_config.frequency_penalty, Some(0.0));
    }
}
