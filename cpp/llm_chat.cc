/*!
 *  Copyright (c) 2023 by Contributors
 * \file llm_chat.cc
 * \brief Implementation of llm chat.
 */
#include "llm_chat.h"

#include <sentencepiece_processor.h>
#include <tokenizers.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/relax_vm/memory_manager.h>

#include <cctype>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <list>
#include <memory>
#include <random>
#include <string>

namespace mlc {
namespace llm {

using tvm::Device;
using namespace tvm::runtime;

/*!
 * \brief helper class to keep track of conversation.
 */
class Conversation {
 public:
  enum class SeparatorStyle { kSingle = 0, kTwo = 1, kDolly = 2, kOasst_Pythia = 3 };

  static Conversation Create(const std::string& template_name = "vicuna_v1.1") {
    if (template_name == "vicuna_v1.1") {
      return Conversation(
          /*conv_template=*/"vicuna_v1.1",
          /*system=*/
          "A chat between a curious user and an artificial intelligence assistant. "
          "The assistant gives helpful, detailed, and polite answers to the user's questions.",
          /*roles=*/{"USER", "ASSISTANT"},
          /*messages=*/{},
          /*offset=*/0,
          /*separator_style=*/Conversation::SeparatorStyle::kTwo,
          /*sep=*/" ",
          /*sep2=*/"</s>");
    } else if (template_name == "conv_one_shot") {
      return Conversation(
          /*conv_template=*/"conv_one_shot",
          /*system=*/
          "A chat between a curious human and an artificial intelligence assistant. "
          "The assistant gives helpful, detailed, and polite answers to the human's questions.",
          /*roles=*/{"Human", "Assistant"},
          /*messages=*/
          {{"Human",
            "What are the key differences between renewable and non-renewable energy sources?"},
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
            "instability."}},
          /*offset=*/2,
          /*separator_style=*/Conversation::SeparatorStyle::kSingle,
          /*sep=*/"###",
          /*sep2=*/"");
    } else if (template_name == "koala_v1") {
      return Conversation(
          /*conv_template=*/"koala_v1",
          /*system=*/"BEGINNING OF CONVERSATION:",
          /*roles=*/{"USER", "GPT"},
          /*messages=*/{},
          /*offset=*/0,
          /*separator_style=*/Conversation::SeparatorStyle::kTwo,
          /*sep=*/" ",
          /*sep2=*/"</s>");
    } else if (template_name == "dolly") {
      return Conversation(
          /*conv_template=*/"dolly",
          /*system=*/
          "Below is an instruction that describes a task. Write a response that appropriately "
          "completes the request.\n\n",
          /*roles=*/{"### Instruction", "### Response"},
          /*messages=*/{},
          /*offset=*/0,
          /*separator_style=*/Conversation::SeparatorStyle::kDolly,
          /*sep=*/"\n\n",
          /*sep2=*/"### End");
    } else if (template_name == "oasst") {
      return Conversation(
          /*conv_template=*/"oasst",
          /*system=*/"",
          /*roles=*/{"<|prompter|>", "<|assistant|>"},
          /*messages=*/{},
          /*offset=*/0,
          /*separator_style=*/Conversation::SeparatorStyle::kOasst_Pythia,
          /*sep=*/"<|endoftext|>",
          /*sep2=*/"");
    } else if (template_name == "stablelm") {
      return Conversation(
          /*conv_template=*/"stablelm",
          /*system=*/
          "<|SYSTEM|># StableLM Tuned (Alpha version)\n"
          "- StableLM is a helpful and harmless open-source AI language model developed by "
          "StabilityAI.\n"
          "- StableLM is excited to be able to help the user, but will refuse to do anything that "
          "could be considered harmful to the user.\n"
          "- StableLM is more than just an information source, StableLM is also able to write "
          "poetry, short stories, and make jokes.\n"
          "- StableLM will refuse to participate in anything that could harm a human.",
          /*roles=*/{"<|USER|>", "<|ASSISTANT|>"},
          /*messages=*/{},
          /*offset=*/0,
          /*separator_style=*/Conversation::SeparatorStyle::kOasst_Pythia,
          /*sep=*/"",
          /*sep2=*/"");
    } else {
      LOG(FATAL) << "Unknown conversation template: " << template_name;
    }
  }

  Conversation() = default;

  Conversation(std::string conv_template, std::string system, std::vector<std::string> roles,
               std::vector<std::vector<std::string>> messages, int32_t offset,
               SeparatorStyle separator_style, std::string sep, std::string sep2)
      : conv_template(conv_template),
        system_(system),
        roles(roles),
        messages(messages),
        separator_style(separator_style),
        sep(sep),
        sep2(sep2) {}

  std::vector<std::string> GetPromptArray() {
    std::vector<std::string> ret;
    if (this->separator_style == SeparatorStyle::kSingle) {
      ret.push_back(this->system_);
      for (const auto& message : this->messages) {
        if (message.size() == 2) {
          ret.push_back(this->sep + " " + message[0] + ": " + message[1]);
        } else if (message.size() == 1) {
          ret.push_back(this->sep + " " + message[0] + ":");
        } else {
          LOG(FATAL) << "Invalid message size: " << message.size();
        }
      }
      return ret;
    } else if (this->separator_style == SeparatorStyle::kTwo) {
      std::vector<std::string> seps{this->sep, this->sep2};
      ret.push_back(this->system_ + seps[0]);
      for (size_t i = 0; i < this->messages.size(); ++i) {
        if (this->messages[i].size() == 2) {
          ret.push_back(this->messages[i][0] + ": " + this->messages[i][1] + seps[i % 2]);
        } else if (this->messages[i].size() == 1) {
          ret.push_back(this->messages[i][0] + ":");
        } else {
          LOG(FATAL) << "Invalid message size: " << this->messages[i].size();
        }
      }
      return ret;
    } else if (this->separator_style == SeparatorStyle::kDolly) {
      std::vector<std::string> seps{this->sep, this->sep2};
      ret.push_back(this->system_);
      for (size_t i = 0; i < this->messages.size(); ++i) {
        if (this->messages[i].size() == 2) {
          if (i % 2 == 1) {
            ret.push_back(this->messages[i][0] + ":\n" + this->messages[i][1] + seps[i % 2] + "\n");
          } else {
            ret.push_back(this->messages[i][0] + ":\n" + this->messages[i][1] + seps[i % 2]);
          }
        } else if (this->messages[i].size() == 1) {
          ret.push_back(this->messages[i][0] + ":\n");
        } else {
          LOG(FATAL) << "Invalid message size: " << this->messages[i].size();
        }
      }
      return ret;
    } else if (this->separator_style == SeparatorStyle::kOasst_Pythia) {
      ret.push_back(this->system_);
      for (const auto& message : this->messages) {
        if (message.size() == 2) {
          ret.push_back(message[0] + message[1] + this->sep);
        } else if (message.size() == 1) {
          ret.push_back(message[0]);
        } else {
          LOG(FATAL) << "Invalid message size: " << message.size();
        }
      }
      return ret;
    } else {
      LOG(FATAL) << "Unknown separator style: " << (int)this->separator_style;
    }
  }

  std::vector<std::string> GetPromptArrayUnprocessed() {
    std::vector<std::string> ret;
    if (this->messages.size() <= 2) {
      LOG(FATAL) << "needs to call getLastPromptArray for the first message";
    }
    if (this->separator_style == SeparatorStyle::kTwo) {
      std::vector<std::string> seps{this->sep, this->sep2};
      ret.push_back(seps[1]);
      for (int i = this->messages.size() - 2; i < this->messages.size(); ++i) {
        if (this->messages[i].size() == 2) {
          ret.push_back(this->messages[i][0] + ": " + this->messages[i][1] + seps[i % 2]);
        } else if (this->messages[i].size() == 1) {
          ret.push_back(this->messages[i][0] + ":");
        } else {
          LOG(FATAL) << "Invalid message size: " << this->messages[i].size();
        }
      }
      return ret;
    } else if (this->separator_style == SeparatorStyle::kDolly) {
      std::vector<std::string> seps{this->sep, this->sep2};
      ret.push_back(seps[1]);
      for (int i = this->messages.size() - 2; i < this->messages.size(); ++i) {
        if (this->messages[i].size() == 2) {
          if (i % 2 == 1) {
            ret.push_back(this->messages[i][0] + ":\n" + this->messages[i][1] + seps[i % 2] + "\n");
          } else {
            ret.push_back(this->messages[i][0] + ":\n" + this->messages[i][1] + seps[i % 2]);
          }
        } else if (this->messages[i].size() == 1) {
          ret.push_back(this->messages[i][0] + ":\n");
        } else {
          LOG(FATAL) << "Invalid message size: " << this->messages[i].size();
        }
      }
      return ret;
    } else if (this->separator_style == SeparatorStyle::kOasst_Pythia) {
      ret.push_back(this->sep);
      for (int i = this->messages.size() - 2; i < this->messages.size(); ++i) {
        if (this->messages[i].size() == 2) {
          ret.push_back(this->messages[i][0] + this->messages[i][1] + this->sep);
        } else if (this->messages[i].size() == 1) {
          ret.push_back(this->messages[i][0]);
        } else {
          LOG(FATAL) << "Invalid message size: " << this->messages[i].size();
        }
      }
      return ret;
    } else {
      LOG(FATAL) << "Unknown separator style: " << (int)this->separator_style;
    }
  }

  void AppendMessage(std::string role, std::string message) {
    this->messages.push_back({role, message});
  }

  void AppendMessage(std::string role) { this->messages.push_back({role}); }

  std::string conv_template;
  SeparatorStyle separator_style{SeparatorStyle::kSingle};
  std::string sep{"###"}, sep2{""};
  std::vector<std::string> roles;
  std::vector<std::vector<std::string>> messages;

 private:
  std::string system_;
};

//----------------------------
// Tokenizers
//----------------------------
std::string LoadBytesFromFile(const std::string& path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  ICHECK(!fs.fail()) << "Cannot open " << path;
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

inline bool EndsWith(std::string const& value, std::string const& end) {
  if (end.size() <= value.size()) {
    return std::equal(end.rbegin(), end.rend(), value.rbegin());
  }
  return false;
}

/*!
 * \brief a universal tokenizer that loads
 * either HF's tokenizer or sentence piece, depending on the type.
 */
class Tokenizer {
 public:
  // bos token
  int32_t bos_token_id{1};
  // eos token id
  int32_t eos_token_id{2};

  virtual ~Tokenizer() {}
  virtual std::vector<int32_t> Encode(const std::string& text) = 0;
  virtual std::string Decode(const std::vector<int32_t>& ids) = 0;

  static std::unique_ptr<Tokenizer> FromFile(const std::string& path);
};

class SentencePieceTokenizer : public Tokenizer {
 public:
  SentencePieceTokenizer(const std::string& path) { sentence_piece_.Load(path); }

  std::vector<int32_t> Encode(const std::string& text) final {
    std::vector<int32_t> tokens;
    sentence_piece_.Encode(text, &tokens).IgnoreError();
    return tokens;
  }

  std::string Decode(const std::vector<int32_t>& ids) final {
    std::string text;
    sentence_piece_.Decode(ids, &text).IgnoreError();
    return text;
  }

 private:
  // the tokenizer
  sentencepiece::SentencePieceProcessor sentence_piece_;
};

class HFTokenizer : public Tokenizer {
 public:
  HFTokenizer(const std::string& path)
      : tokenizer_(tokenizers::Tokenizer::FromJSON(LoadBytesFromFile(path))) {}

  std::vector<int32_t> Encode(const std::string& text) final {
    return tokenizer_.Encode(text, false);
  }

  std::string Decode(const std::vector<int32_t>& ids) final {
    return tokenizer_.Decode(ids, false);
  }

 private:
  // the tokenizer
  tokenizers::Tokenizer tokenizer_;
};

std::unique_ptr<Tokenizer> Tokenizer::FromFile(const std::string& path) {
  if (EndsWith(path, ".model")) {
    return std::make_unique<SentencePieceTokenizer>(path);
  } else {
    return std::make_unique<HFTokenizer>(path);
  }
}

//------------------------------
// Chat module
//------------------------------
/*!
 * \brief Implements the chat conversation wrapper
 */
class LLMChatModule : public ModuleNode {
 public:
  // overrides
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "evaluate") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) { this->Evaluate(); });
    } else if (name == "try_tokenizer") {
      return PackedFunc(
          [this, sptr_to_self](TVMArgs args, TVMRetValue* rv) { this->TryTokenizer(); });
    } else if (name == "encode") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.size(), 1);
        this->EncodeStep(args[0]);
      });
    } else if (name == "decode") {
      return PackedFunc(
          [this, sptr_to_self](TVMArgs args, TVMRetValue* rv) { this->DecodeStep(); });
    } else if (name == "init_chat") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.size(), 9);
        this->model_name_ = args[0].operator std::string();
        this->conversation_ = Conversation::Create(args[1]);
        this->max_gen_len_ = args[2];
        this->temperature_ = args[3];
        this->top_p_ = args[4];
        this->stream_interval_ = args[5];
        this->max_window_size_ = args[6];
        this->mean_gen_len_ = args[7];
        this->shift_fill_factor_ = args[8];
        this->ClearKVCache();
        this->total_seq_len_ = 0;
        this->start_pos_ = 0;
        this->cur_pos_ = 0;
        this->add_bos_ = true;
        this->stop_tokens_ = (args[1] == "stablelm")
                                 ? std::vector<int32_t>({50278, 50279, 50277, 1, 0})
                                 : std::vector<int32_t>({this->tokenizer_.get()->eos_token_id});
        this->stop_str_ =
            this->conversation_.separator_style == Conversation::SeparatorStyle::kSingle
                ? this->conversation_.sep
                : this->conversation_.sep2;
      });
    } else if (name == "reset_chat") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.size(), 0);
        this->conversation_.messages.clear();
        this->ClearKVCache();
        this->total_seq_len_ = 0;
        this->start_pos_ = 0;
        this->cur_pos_ = 0;
        this->add_bos_ = true;
      });
    } else if (name == "get_role0") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        *rv = this->conversation_.roles[0];
      });
    } else if (name == "get_role1") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        *rv = this->conversation_.roles[1];
      });
    } else if (name == "stopped") {
      return PackedFunc(
          [this, sptr_to_self](TVMArgs args, TVMRetValue* rv) { *rv = this->Stopped(); });
    } else if (name == "get_message") {
      return PackedFunc(
          [this, sptr_to_self](TVMArgs args, TVMRetValue* rv) { *rv = this->GetMessage(); });
    } else if (name == "runtime_stats_text") {
      return PackedFunc(
          [this, sptr_to_self](TVMArgs args, TVMRetValue* rv) { *rv = this->RuntimeStatsText(); });
    } else if (name == "reset_runtime_stats") {
      return PackedFunc(
          [this, sptr_to_self](TVMArgs args, TVMRetValue* rv) { this->ResetRuntimeStats(); });
    } else {
      return PackedFunc(nullptr);
    }
  }

  const char* type_key() const final { return "mlc.llm_chat"; }

  /*!
   * \return Text describing runtime stats.
   */
  std::string RuntimeStatsText() {
    std::ostringstream os;
    os << "encode: " << std::setprecision(1) << std::fixed
       << this->encode_total_tokens / this->encode_total_time << " tok/s"
       << ", decode: " << std::setprecision(1) << std::fixed
       << this->decode_total_tokens / this->decode_total_time << " tok/s";
    return os.str();
  }

  /*! \brief reset the runtime stats. */
  void ResetRuntimeStats() {
    this->encode_total_tokens = 0;
    this->decode_total_tokens = 0;
    this->encode_total_time = 0;
    this->decode_total_time = 0;
  }

  std::vector<int32_t> GetPromptTokens() {
    std::vector<std::string> prompts;
    if (this->conversation_.messages.size() <= 2) {
      prompts = this->conversation_.GetPromptArray();
    } else {
      prompts = this->conversation_.GetPromptArrayUnprocessed();
    }

    std::vector<int32_t> tokens;
    if (this->add_bos_) {
      tokens.insert(tokens.begin(), tokenizer_->bos_token_id);
    }
    auto first_prompt_tokens = this->tokenizer_->Encode(prompts[0]);
    tokens.insert(tokens.end(), first_prompt_tokens.begin(), first_prompt_tokens.end());
    int ctx_length = tokens.size();
    std::list<std::vector<int32_t>> context;

    bool need_shift_window = false;
    for (int i = prompts.size() - 1; i > 0; i--) {
      auto encoded = this->tokenizer_->Encode(prompts[i]);
      ctx_length += encoded.size();
      if (this->total_seq_len_ + ctx_length + this->mean_gen_len_ >= this->max_window_size_) {
        need_shift_window = true;
        break;
      }
      context.push_front(encoded);
    }
    if (!need_shift_window) {
      for (const auto& ctx : context) {
        tokens.insert(tokens.end(), ctx.begin(), ctx.end());
      }
      return tokens;
    }
    // need shift window and re-encode
    this->total_seq_len_ = 0;
    this->ClearKVCache();
    context.clear();
    tokens.clear();
    if (this->add_bos_) {
      tokens.insert(tokens.begin(), tokenizer_->bos_token_id);
    }
    auto all_prompts = this->conversation_.GetPromptArray();
    first_prompt_tokens = this->tokenizer_->Encode(all_prompts[0]);
    tokens.insert(tokens.end(), first_prompt_tokens.begin(), first_prompt_tokens.end());
    ctx_length = tokens.size();
    for (int i = all_prompts.size() - 1; i > 0; i--) {
      auto encoded = this->tokenizer_->Encode(all_prompts[i]);
      ctx_length += encoded.size();
      if (ctx_length >= this->shift_fill_factor_ * this->max_window_size_ &&
          i + 2 < all_prompts.size()) {
        break;
      }
      context.push_front(encoded);
    }
    for (const auto& ctx : context) {
      tokens.insert(tokens.end(), ctx.begin(), ctx.end());
    }
    if (tokens.size() + this->mean_gen_len_ >= this->max_window_size_) {
      LOG(FATAL) << "Exceed max window length curr=" << tokens.size();
    }
    return tokens;
  }

  // get statically allocated input token
  NDArray GetInputTokenNDArray(const std::vector<int32_t>& token_ids) {
    if (!input_token_ids_.defined()) {
      input_token_ids_ = NDArray::Empty({1, max_window_size_}, DataType::Int(32), device_);
    }
    ICHECK_LE(token_ids.size(), input_token_ids_->shape[1]) << "Input tokens exceed window size";
    NDArray view = input_token_ids_.CreateView(
        ShapeTuple({1, static_cast<int64_t>(token_ids.size())}), input_token_ids_->dtype);
    view.CopyFromBytes(token_ids.data(), token_ids.size() * sizeof(int32_t));
    return view;
  }

  /*!
   * \brief Generate the next token given a prompt.
   */
  void EncodeStep(std::string inp) {
    if (reset_stats_per_encode_) {
      this->ResetRuntimeStats();
    }
    output_ids_.clear();
    output_message_.clear();
    encounter_stop_str_ = false;

    conversation_.AppendMessage(conversation_.roles[0], inp);
    conversation_.AppendMessage(conversation_.roles[1]);

    auto prompt_tokens = this->GetPromptTokens();
    int64_t token_len = static_cast<int64_t>(prompt_tokens.size());

    auto input_data = this->GetInputTokenNDArray(prompt_tokens);

    total_seq_len_ += token_len;
    cur_pos_ = token_len;
    start_pos_ = token_len;

    auto tstart = std::chrono::high_resolution_clock::now();
    this->UpdateLogitsOnCPU(this->Forward(input_data, total_seq_len_));
    TVMSynchronize(device_.device_type, device_.device_id, nullptr);
    auto tend = std::chrono::high_resolution_clock::now();

    this->encode_total_time += static_cast<double>((tend - tstart).count()) / 1e9;
    this->encode_total_tokens += token_len;

    next_token_ = this->SampleFromLogitsOnCPU();

    if (model_name_.find("vicuna") == 0) {
      add_bos_ = false;
    }
  }

  void DecodeStep() {
    output_ids_.push_back(next_token_);
    output_message_ = RemoveStopStr(tokenizer_->Decode(output_ids_));

    auto input_data = GetInputTokenNDArray({next_token_});

    total_seq_len_ += 1;
    cur_pos_ += 1;

    auto tstart = std::chrono::high_resolution_clock::now();
    this->UpdateLogitsOnCPU(this->Forward(input_data, total_seq_len_));
    TVMSynchronize(device_.device_type, device_.device_id, nullptr);

    next_token_ = this->SampleFromLogitsOnCPU();
    auto tend = std::chrono::high_resolution_clock::now();

    this->decode_total_time += static_cast<double>((tend - tstart).count()) / 1e9;
    this->decode_total_tokens += 1;
  }

  bool Stopped() {
    if (std::any_of(stop_tokens_.begin(), stop_tokens_.end(),
                    [this](int32_t token) { return token == next_token_; }))
      return true;
    return cur_pos_ - start_pos_ == max_gen_len_ - 1 || encounter_stop_str_ ||
           total_seq_len_ >= max_window_size_;
  }

  size_t FindEffectiveUTF8Pos(const std::string& s, size_t start_pos) {
    size_t pos = start_pos;
    while (pos < s.size()) {
      if ((s[pos] & 0x80) == 0x00) {
        pos += 1;
      } else if (pos + 1 < s.size() && (s[pos] & 0xE0) == 0xC0 && (s[pos + 1] & 0xC0) == 0x80) {
        pos += 2;
      } else if (pos + 2 < s.size() && (s[pos] & 0xF0) == 0xE0 && (s[pos + 1] & 0xC0) == 0x80 &&
                 (s[pos + 2] & 0xC0) == 0x80) {
        pos += 3;
      } else if (pos + 3 < s.size() && (s[pos] & 0xF8) == 0xF0 && (s[pos + 1] & 0xC0) == 0x80 &&
                 (s[pos + 2] & 0xC0) == 0x80 && (s[pos + 3] & 0xC0) == 0x80) {
        pos += 4;
      } else {
        break;
      }
    }
    return pos;
  }

  std::string GetMessage() {
    // remove non-utf8 characters
    std::string cropped_message =
        output_message_.substr(0, FindEffectiveUTF8Pos(output_message_, 0));
    return cropped_message;
  }

  // do some quick evaluation of the tokenizer
  void TryTokenizer() {
    std::string input = "The capital of Canada is";
    std::vector<int32_t> ids = tokenizer_->Encode(input);
    std::ostringstream os;

    for (size_t i = 0; i < ids.size(); ++i) {
      if (i != 0) os << ", ";
      os << ids[i];
    }
    LOG(INFO) << "TryTokenizer: input=" << input;
    LOG(INFO) << "TryTokenizer: tokenize-ids=[" << os.str() << "]";
    std::string result = tokenizer_->Decode(ids);
    ICHECK_EQ(result, input);
  }

  // do some quick evaluation of the pipeline
  void Evaluate() {
    this->ClearKVCache();
    std::string test_prompt = "The capital of Canada is";
    std::vector<int32_t> tokens = tokenizer_->Encode(test_prompt);
    tokens.insert(tokens.begin(), tokenizer_->bos_token_id);
    int64_t token_len = static_cast<int64_t>(tokens.size());

    auto input_data = NDArray::Empty({1, token_len}, DataType::Int(32), device_);
    input_data.CopyFromBytes(tokens.data(), tokens.size() * sizeof(int32_t));
    auto first_sample_token = NDArray::Empty({1, 1}, DataType::Int(32), device_);
    std::vector<int32_t> first_sample_data = {6234};
    first_sample_token.CopyFromBytes(first_sample_data.data(), sizeof(int32_t));

    // warm up: skip first run
    this->Forward(input_data, token_len);
    this->Forward(first_sample_token, token_len + 1);
    this->ClearKVCache();

    // start recording
    auto encoding_start = std::chrono::high_resolution_clock::now();
    this->Forward(input_data, token_len);
    TVMSynchronize(device_.device_type, device_.device_id, nullptr);

    auto decoding_start = std::chrono::high_resolution_clock::now();
    this->UpdateLogitsOnCPU(this->Forward(first_sample_token, token_len + 1));
    TVMSynchronize(device_.device_type, device_.device_id, nullptr);
    auto decoding_end = std::chrono::high_resolution_clock::now();

    // print first few logits for eyeballs
    std::ostringstream os;
    for (int i = 0; i < 10; ++i) {
      if (i != 0) os << ", ";
      os << static_cast<float*>(logits_on_cpu_->data)[i];
    }
    LOG(INFO) << "logits[:10] =[" << os.str() << "]";

    double encoding_ms = static_cast<double>((decoding_start - encoding_start).count()) / 1e6;
    double decoding_ms = static_cast<double>((decoding_end - decoding_start).count()) / 1e6;

    LOG(INFO) << "encoding-time=" << encoding_ms << "ms, "
              << "decoding-time=" << decoding_ms << "ms.";
  }

  /*!
   * \brief Load necessary component from related places.
   *
   * \param executable The executable information.
   * \param tokenizer_path The root path to params
   * \param param_path The root path to params
   * \param device The device to run the mdoel on
   */
  void Init(Module executable, const std::string& tokenizer_path, const std::string& param_path,
            tvm::Device device) {
    // setup members
    device_ = device;

    // initialize tokenizer
    tokenizer_ = Tokenizer::FromFile(tokenizer_path);

    // load in nd-arracy cache
    const PackedFunc* fload_cache = tvm::runtime::Registry::Get("vm.builtin.ndarray_cache.load");
    ICHECK(fload_cache) << "TVM runtime cannot find vm.builtin.ndarray_cache.load";
    (*fload_cache)(param_path, static_cast<int32_t>(device_.device_type), device.device_id);

    // initialize vm, we use the packed function mechanism
    // so there is no explicit abi dependency on these extra
    // classes other than basic tvm runtime.
    vm_ = executable->GetFunction("vm_load_executable")();
    vm_->GetFunction("vm_initialization")(static_cast<int>(device.device_type), device.device_id,
                                          static_cast<int>(relax_vm::AllocatorType::kPooled),
                                          static_cast<int>(kDLCPU), 0,
                                          static_cast<int>(relax_vm::AllocatorType::kPooled));

    encoding_func_ = vm_->GetFunction("encoding");
    decoding_func_ = vm_->GetFunction("decoding");
    encoding_without_cache_func_ = vm_->GetFunction("encoding_without_cache");
    auto kv_cache_func = vm_->GetFunction("create_kv_cache");

    // parameter loading
    const PackedFunc* fload_params =
        tvm::runtime::Registry::Get("vm.builtin.param_array_from_cache");
    ICHECK(fload_params) << "Cannot find env function vm.builtin.param_array_from_cache";
    params_ = (*fload_params)("param", -1);

    // KV cache creation
    kv_cache_ = vm_->GetFunction("create_kv_cache")();
    // Other system function
    // Get bos
  }

 private:
  int CountSubstr(const std::string& str, const std::string& sub) {
    if (sub.length() == 0) return 0;
    int count = 0;
    for (size_t offset = str.find(sub); offset != std::string::npos;
         offset = str.find(sub, offset + sub.length())) {
      ++count;
    }
    return count;
  }

  int64_t ComputeSkipEchoLen(const std::string& prompt) {
    int64_t skip_echo_len = 0;
    std::string model_name(model_name_);
    std::transform(model_name.begin(), model_name.end(), model_name.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (model_name.find("chatglm") != std::string::npos) {
      skip_echo_len = conversation_.messages[conversation_.messages.size() - 1][1].length() + 1;
    } else if (model_name.find("dolly") != std::string::npos) {
      std::vector<std::string> special_toks{"### Instruction:", "### Response:", "### End"};
      skip_echo_len = prompt.length();
      for (const auto& tok : special_toks) {
        skip_echo_len -= CountSubstr(prompt, tok) * tok.length();
      }
    } else if (model_name.find("oasst") != std::string::npos &&
               model_name.find("pythia") != std::string::npos) {
      std::vector<std::string> special_toks{"<|prompter|>", "<|assistant|>", "<|endoftext|>"};
      skip_echo_len = prompt.length();
      for (const auto& tok : special_toks) {
        skip_echo_len -= CountSubstr(prompt, tok) * tok.length();
      }
    } else if (model_name.find("stablelm") != std::string::npos) {
      std::vector<std::string> special_toks{"<|SYSTEM|>", "<|USER|>", "<|ASSISTANT|>"};
      skip_echo_len = prompt.length();
      for (const auto& tok : special_toks) {
        skip_echo_len -= CountSubstr(prompt, tok) * tok.length();
      }
    } else {
      skip_echo_len = prompt.length() + 1 - CountSubstr(prompt, "</s>") * 3;
    }
    return skip_echo_len;
  }

  // run forward compute
  NDArray Forward(NDArray inputs, int64_t cur_pos) {
    Array<ObjectRef> ret;
    if (inputs->shape[1] > 1) {
      ret = encoding_func_(inputs, ShapeTuple({cur_pos}), kv_cache_, params_);
    } else {
      ret = decoding_func_(inputs, ShapeTuple({cur_pos}), kv_cache_, params_);
    }
    return Downcast<NDArray>(ret[0]);
  }

  void UpdateLogitsOnCPU(NDArray logits) {
    if (!logits_on_cpu_.defined()) {
      logits_on_cpu_ = logits.CopyTo(DLDevice{kDLCPU, 0});
    } else {
      ICHECK_EQ(logits_on_cpu_->shape[0], logits->shape[0])
          << "Expect size of logits remain unchanged";
      logits_on_cpu_.CopyFrom(logits);
    }
  }

  // Clear kv cache
  void ClearKVCache() {
    const PackedFunc* fkv_clear =
        tvm::runtime::Registry::Get("vm.builtin.attention_kv_cache_array_clear");
    ICHECK(fkv_clear);
    (*fkv_clear)(kv_cache_);
  }

  // Utils
  static double GetRandomNumber() {
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_real_distribution<> dis(0.0, 1.0);
    return dis(gen);
  }

  int32_t SampleFromLogitsOnCPU() {
    ICHECK(logits_on_cpu_.defined()) << "logits_on_cpu_ is not defined";
    ICHECK_EQ(logits_on_cpu_->ndim, 3) << "logits_on_cpu_ should be 3D";
    ICHECK_EQ(logits_on_cpu_->shape[0], 1) << "logits_on_cpu_ should be 1 batch";
    const PackedFunc* fsample_topp =
        tvm::runtime::Registry::Get("vm.builtin.sample_top_p_from_logits");
    ICHECK(fsample_topp) << "Cannot find env function vm.builtin.sample_top_p_from_logits";
    return (*fsample_topp)(logits_on_cpu_, top_p_, temperature_, GetRandomNumber());
  }

  std::string DeltaMessage(const std::string& cur, const std::string& old) {
    std::string ret;
    int pos = std::min(old.length(), cur.length()) - 1;
    for (; pos >= 0 && cur[pos] != '\n'; --pos)
      ;
    ret += '\r';
    ret += cur.substr(pos + 1);
    return ret;
  }

  std::string RemoveStopStr(std::string str) {
    size_t pos = str.rfind(stop_str_);
    if (pos != std::string::npos) {
      encounter_stop_str_ = true;
      str = str.substr(0, pos);
    }
    return str;
  }

  //----------------------------
  // Statistics
  //----------------------------
  bool reset_stats_per_encode_ = true;
  double decode_total_time = 0;
  double encode_total_time = 0;
  int64_t decode_total_tokens = 0;
  int64_t encode_total_tokens = 0;
  //----------------------------
  // Conversation
  //----------------------------
  // model name
  std::string model_name_;
  // conversation
  Conversation conversation_;
  // max_gen_len
  int64_t max_gen_len_{2048};
  // total sequence len, start position, current position
  int64_t total_seq_len_{0}, start_pos_{0}, cur_pos_{0}, skip_echo_len_{0};
  // max window size, mean generation length
  int64_t max_window_size_{768}, mean_gen_len_{128};
  // shift window fill factor
  double shift_fill_factor_{0.3};
  // temperature
  double temperature_{0.8};
  // top_p
  double top_p_{0.95};
  // stream interval
  int64_t stream_interval_{1};
  // next_token
  int32_t next_token_{0};
  // output ids till now (refresh after encoding step)
  std::vector<int32_t> output_ids_;
  // output message till now (refresh after encoding step)
  std::string output_message_;
  // whether to add bos as the first token
  bool add_bos_{true};
  // stop tokens
  std::vector<int32_t> stop_tokens_;
  // stop str
  std::string stop_str_;
  // Whether encounter stop str
  bool encounter_stop_str_{false};
  //----------------------------
  // Tokenizer
  //----------------------------
  // internal tokenizer
  std::unique_ptr<Tokenizer> tokenizer_;
  //----------------------------
  // TVM related states
  //----------------------------
  // runtime device
  Device device_;
  // The vm module
  Module vm_;
  // encoding function
  PackedFunc encoding_func_;
  // decoding function
  PackedFunc decoding_func_;
  // encoding without cache
  PackedFunc encoding_without_cache_func_;
  // input token id
  NDArray input_token_ids_{nullptr};
  // local params
  Array<NDArray> params_;
  // KV cache
  Array<ObjectRef> kv_cache_;
  // Temp logits on cpu
  NDArray logits_on_cpu_{nullptr};
};

tvm::runtime::Module CreateChatModule(tvm::runtime::Module executable,
                                      const tvm::runtime::String& tokenizer_path,
                                      const tvm::runtime::String& param_path, DLDevice device) {
  ObjectPtr<LLMChatModule> n = make_object<LLMChatModule>();
  n->Init(executable, tokenizer_path, param_path, device);
  return Module(n);
}

// register as a system function that can be queried
TVM_REGISTER_GLOBAL("mlc.llm_chat_create")
    .set_body_typed([](tvm::runtime::Module executable, const tvm::runtime::String& tokenizer_path,
                       const tvm::runtime::String& param_path, int device_type, int device_id) {
      return CreateChatModule(executable, tokenizer_path, param_path,
                              DLDevice{static_cast<DLDeviceType>(device_type), device_id});
    });
}  // namespace llm
}  // namespace mlc
