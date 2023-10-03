/*!
 *  Copyright (c) 2023 by Contributors
 * \file llm_chat.cc
 * \brief Implementation of llm chat.
 */
#define PICOJSON_USE_INT64
#define __STDC_FORMAT_MACROS

#include "llm_chat.h"

#include <picojson.h>
#include <tokenizers_cpp.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/disco/session.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/relax_vm/memory_manager.h>

#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <list>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "conversation.h"

namespace mlc {
namespace llm {

using tvm::Device;
using namespace tvm::runtime;
namespace {
//----------------------------
// Tokenizers
//----------------------------
using tokenizers::Tokenizer;

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

std::unique_ptr<Tokenizer> TokenizerFromPath(const std::string& _path) {
  std::filesystem::path path(_path);
  std::filesystem::path sentencepiece;
  std::filesystem::path huggingface;
  std::filesystem::path rwkvworld;
  CHECK(std::filesystem::exists(path)) << "Cannot find tokenizer via path: " << _path;
  if (std::filesystem::is_directory(path)) {
    sentencepiece = path / "tokenizer.model";
    huggingface = path / "tokenizer.json";
    rwkvworld = path / "tokenizer_model";
    // Check ByteLevelBPE
    {
      std::filesystem::path merges_path = path / "merges.txt";
      std::filesystem::path vocab_path = path / "vocab.json";
      std::filesystem::path added_tokens_path = path / "added_tokens.json";
      if (std::filesystem::exists(merges_path) && std::filesystem::exists(vocab_path) &&
          std::filesystem::exists(added_tokens_path)) {
        std::string vocab = LoadBytesFromFile(vocab_path.string());
        std::string merges = LoadBytesFromFile(merges_path.string());
        std::string added_tokens = LoadBytesFromFile(added_tokens_path.string());
        return Tokenizer::FromBlobByteLevelBPE(vocab, merges, added_tokens);
      }
    }
  } else {
    sentencepiece = path.parent_path() / "tokenizer.model";
    huggingface = path.parent_path() / "tokenizer.json";
    rwkvworld = path.parent_path() / "tokenizer_model";
  }
  if (std::filesystem::exists(sentencepiece)) {
    return Tokenizer::FromBlobSentencePiece(LoadBytesFromFile(sentencepiece.string()));
  }
  if (std::filesystem::exists(huggingface)) {
    return Tokenizer::FromBlobJSON(LoadBytesFromFile(huggingface.string()));
  }
  if (std::filesystem::exists(rwkvworld)) {
    return Tokenizer::FromBlobRWKVWorld(rwkvworld.string());
  }
  LOG(FATAL) << "Cannot find any tokenizer under: " << _path;
}

//------------------------------
// support functions
//------------------------------
inline size_t FindEffectiveUTF8Pos(const std::string& s) {
  int pos = s.size() - 1;
  for (; pos >= 0; pos--) {
    if ((s[pos] & 0x80) == 0x00) {
      return pos + 1;
    } else if (pos - 1 >= 0 && (s[pos - 1] & 0xE0) == 0xC0 && (s[pos] & 0xC0) == 0x80) {
      return pos + 1;
    } else if (pos - 2 >= 0 && (s[pos - 2] & 0xF0) == 0xE0 && (s[pos - 1] & 0xC0) == 0x80 &&
               (s[pos] & 0xC0) == 0x80) {
      return pos + 1;
    } else if (pos - 3 >= 0 && (s[pos - 3] & 0xF8) == 0xF0 && (s[pos - 2] & 0xC0) == 0x80 &&
               (s[pos - 1] & 0xC0) == 0x80 && (s[pos] & 0xC0) == 0x80) {
      return pos + 1;
    }
  }
  return pos + 1;
}

inline std::string Concat(const std::vector<std::string>& inputs) {
  std::ostringstream os;
  for (const auto& x : inputs) {
    os << x;
  }
  return os.str();
}

struct FunctionTable {
  static PackedFunc SessionFuncAsPackedFunc(Session sess, DRef sess_func, String name) {
    return PackedFunc([sess, func = std::move(sess_func), name = std::move(name)](
                          TVMArgs args, TVMRetValue* rv) -> void {
      std::vector<TVMValue> tvm_values(args.num_args + 3);
      std::vector<int> tvm_type_codes(args.num_args + 3);
      TVMArgsSetter setter(tvm_values.data(), tvm_type_codes.data());
      setter(0, static_cast<int>(DiscoAction::kCallPacked));
      setter(1, 0);
      setter(2, func);
      for (int i = 0; i < args.num_args; ++i) {
        tvm_values[i + 3] = args.values[i];
        tvm_type_codes[i + 3] = args.type_codes[i];
      }
      *rv = sess->CallWithPacked(
          TVMArgs(tvm_values.data(), tvm_type_codes.data(), args.num_args + 3));
    });
  }

  void Init(TVMArgValue reload_lib, Device device, int num_shards) {
    Device null_device{DLDeviceType(0), 0};
    if (num_shards > 1) {
      String lib_path{nullptr};
      try {
        lib_path = reload_lib.operator String();
      } catch (...) {
        LOG(FATAL)
            << "ValueError: In multi-GPU inference, we expect the first argument to Reload to be a "
               "string path to the model library (.so on Linux or .dll on Windows), but got: "
            << ArgTypeCode2Str(reload_lib.type_code());
      }
      constexpr const char* f_create_process_pool = "runtime.disco.create_process_pool";
      if (Registry::Get(f_create_process_pool) == nullptr) {
        LOG(FATAL) << "Cannot find process launcher `" << f_create_process_pool << "`. "
                   << "Multi-GPU inference depends on MLC LLM Python API to launch process.";
      }
      std::vector<int64_t> device_ids(num_shards);
      for (int i = 0; i < num_shards; ++i) {
        device_ids[i] = i;
      }
      this->use_disco = true;
      this->sess = Session::ProcessSession(num_shards, f_create_process_pool);
      this->sess->InitCCL("nccl", ShapeTuple(device_ids));
      this->disco_mod = sess->CallPacked(sess->GetGlobalFunc("runtime.disco.load_vm_module"),
                                         lib_path, null_device);
      this->mod_get_func = [this, fmodule_get_function =
                                      sess->GetGlobalFunc("runtime.ModuleGetFunction")](
                               const std::string& name) -> PackedFunc {
        DRef func = sess->CallPacked(fmodule_get_function, this->disco_mod, name, false);
        bool exists = (func->DebugGetFromRemote(0).operator PackedFunc()) != nullptr;
        if (!exists) {
          return PackedFunc(nullptr);
        }
        return SessionFuncAsPackedFunc(sess, func, name);
      };
      this->get_global_func = [this](const std::string& name) -> PackedFunc {
        return SessionFuncAsPackedFunc(sess, sess->GetGlobalFunc(name), name);
      };
      this->_InitFunctions();
      {
        Module mod = this->disco_mod->DebugGetFromRemote(0);
        this->softmax_func_ = mod->GetFunction("softmax_with_temperature");
      }
    } else {
      Module executable{nullptr};
      if (reload_lib.type_code() == kTVMModuleHandle) {
        executable = reload_lib.operator Module();
      } else {
        String lib_path = reload_lib.operator String();
        executable = tvm::runtime::Module::LoadFromFile(lib_path);
      }
      this->use_disco = false;
      auto fload_exec = executable->GetFunction("vm_load_executable");
      ICHECK(fload_exec.defined()) << "TVM runtime cannot find vm_load_executable";
      this->local_vm = fload_exec();
      this->local_vm->GetFunction("vm_initialization")(
          static_cast<int>(device.device_type), device.device_id,
          static_cast<int>(relax_vm::AllocatorType::kPooled), static_cast<int>(kDLCPU), 0,
          static_cast<int>(relax_vm::AllocatorType::kPooled));
      this->mod_get_func = [this](const std::string& name) -> PackedFunc {
        return this->local_vm->GetFunction(name, false);
      };
      this->get_global_func = [](const std::string& name) -> PackedFunc {
        const auto* f = tvm::runtime::Registry::Get(name);
        CHECK(f != nullptr) << "ValueError: Cannot find function " << name;
        return *f;
      };
      this->_InitFunctions();
    }
  }

  ObjectRef LoadParams(const std::string& model_path, Device device) {
    if (this->use_disco) {
      std::filesystem::path fs_model_path = model_path;
      std::string metadata_path = (fs_model_path / "ndarray-cache.json").string();
      std::string ndarray_cache_metadata = LoadBytesFromFile(metadata_path);
      PackedFunc loader_create = this->get_global_func("runtime.disco.ShardLoader");
      PackedFunc loader_load_all = this->get_global_func("runtime.disco.ShardLoaderLoadAll");
      CHECK(loader_create != nullptr);
      CHECK(loader_load_all != nullptr);
      DRef loader = loader_create(metadata_path, ndarray_cache_metadata, "", this->disco_mod);
      DRef params = loader_load_all(loader);
      return params;
    } else {
      const PackedFunc* fload_cache = tvm::runtime::Registry::Get("vm.builtin.ndarray_cache.load");
      ICHECK(fload_cache) << "TVM runtime cannot find vm.builtin.ndarray_cache.load";
      (*fload_cache)(model_path, static_cast<int32_t>(device.device_type), device.device_id);
      const PackedFunc* fload_params =
          tvm::runtime::Registry::Get("vm.builtin.param_array_from_cache");
      ICHECK(fload_params) << "Cannot find env function vm.builtin.param_array_from_cache";
      Array<NDArray> params = (*fload_params)("param", -1);
      // after we get params, it is safe to simply clear the cached version
      // as these params are referenced by params_
      const PackedFunc* fclear_ndarray_cache =
          tvm::runtime::Registry::Get("vm.builtin.ndarray_cache.clear");
      ICHECK(fclear_ndarray_cache) << "Cannot find env function vm.builtin.ndarray_cache.clear";
      (*fclear_ndarray_cache)();
      return params;
    }
  }

  void _InitFunctions() {
    this->prefill_func_ = mod_get_func("prefill");
    this->embed_func_ = mod_get_func("embed");
    this->prefill_with_embed_func_ = mod_get_func("prefill_with_embed");
    this->decode_func_ = mod_get_func("decode");
    this->softmax_func_ = mod_get_func("softmax_with_temperature");
    this->encoding_without_cache_func_ = mod_get_func("encoding_without_cache");
    this->create_kv_cache_func_ = mod_get_func("create_kv_cache");
    this->reset_kv_cache_func_ = mod_get_func("reset_kv_cache");
    if (this->reset_kv_cache_func_ == nullptr) {
      this->reset_kv_cache_func_ = get_global_func("vm.builtin.attention_kv_cache_array_clear");
      support_backtracking_kv_ = true;
    } else {
      support_backtracking_kv_ = false;
    }
    this->fkvcache_array_popn_ = get_global_func("vm.builtin.attention_kv_cache_array_popn");
  }

  ObjectRef Empty(ShapeTuple shape, DataType dtype, Device device) const {
    Device null_device{DLDeviceType(0), 0};
    if (this->use_disco) {
      DRef empty_func = sess->GetGlobalFunc("runtime.disco.empty");
      return sess->CallPacked(empty_func, shape, dtype, null_device);
    } else {
      return NDArray::Empty(shape, dtype, device);
    }
  }

  ObjectRef CopyToWorker0(const NDArray& host_array) {
    Device null_device{DLDeviceType(0), 0};
    if (this->use_disco) {
      DRef array =
          Downcast<DRef>(this->Empty(host_array.Shape(), host_array.DataType(), null_device));
      sess->CopyToWorker0(host_array, array);
      return array;
    } else {
      return host_array;
    }
  }

  bool use_disco = false;
  Session sess{nullptr};
  DRef disco_mod{nullptr};
  tvm::runtime::Module local_vm{nullptr};

  TypedPackedFunc<PackedFunc(const std::string&)> mod_get_func;
  TypedPackedFunc<PackedFunc(const std::string&)> get_global_func;

  PackedFunc prefill_func_;
  PackedFunc embed_func_;
  PackedFunc prefill_with_embed_func_;
  PackedFunc decode_func_;
  PackedFunc encoding_without_cache_func_;
  PackedFunc softmax_func_;
  PackedFunc create_kv_cache_func_;
  PackedFunc reset_kv_cache_func_;
  bool support_backtracking_kv_;
  PackedFunc fkvcache_array_popn_;
};

class RandomGenerator {
 private:
  std::mt19937 gen;
  std::uniform_real_distribution<> dis;

  RandomGenerator(int seed) : gen(seed), dis(0.0, 1.0) {}

 public:
  static RandomGenerator& GetInstance(int seed = std::random_device{}()) {
    static RandomGenerator instance(seed);
    return instance;
  }

  double GetRandomNumber() { return dis(gen); }

  void SetSeed(int seed) { gen.seed(seed); }
};
}  // namespace

//------------------------------
// Chat module
//------------------------------
class LLMChatModule;

/*!
 * \brief Implements the chat conversation wrapper
 */
class LLMChat {
  friend class LLMChatModule;

 public:
  explicit LLMChat(DLDevice device) : device_(device) {}

  /*!
   * \return Text describing runtime stats.
   */
  std::string RuntimeStatsText() {
    std::ostringstream os;
    os << "prefill: " << std::setprecision(1) << std::fixed
       << this->prefill_total_tokens / (this->prefill_total_time + this->embed_total_time)
       << " tok/s"
       << ", decode: " << std::setprecision(1) << std::fixed
       << this->decode_total_tokens / this->decode_total_time << " tok/s";
    return os.str();
  }

  /*!
   * \brief Load JSON config and override options.
   * \param config_json A json config in picojson type that is partially specifies
   *        some of the options.
   * \param partial_update Whether it's a partial update or full update, if set to true,
   *        we perform a partial update on some of the provided options; if set to false, all
   *        options must be provided.
   * \note This function overrides existing configurations.
   */
  void LoadJSONOverride(const picojson::value& config_json, bool partial_update = false) {
    picojson::object config = config_json.get<picojson::object>();
    if (config.count("temperature")) {
      CHECK(config["temperature"].is<double>());
      this->temperature_ = config["temperature"].get<double>();
    } else {
      CHECK(partial_update) << "Key \"temperature\" not found.";
    }
    if (config.count("repetition_penalty")) {
      CHECK(config["repetition_penalty"].is<double>());
      CHECK(this->repetition_penalty_ > 0) << "Repetition penalty must be a positive number!";
      this->repetition_penalty_ = config["repetition_penalty"].get<double>();
    } else {
      CHECK(partial_update) << "Key \"repetition_penalty\" not found.";
    }
    if (config.count("vocab_size")) {
      CHECK(config["vocab_size"].is<int64_t>());
      this->vocab_size_ = config["vocab_size"].get<int64_t>();
    } else {
      CHECK(partial_update) << "Key \"vocab_size\" not found.";
    }
    if (config.count("num_shards")) {
      CHECK(config["num_shards"].is<int64_t>());
      this->num_shards_ = config["num_shards"].get<int64_t>();
    } else {
      this->num_shards_ = 1;
    }
    if (config.count("max_window_size")) {
      CHECK(config["max_window_size"].is<int64_t>());
      this->max_window_size_ = config["max_window_size"].get<int64_t>();
    } else {
      CHECK(partial_update) << "Key \"max_window_size\" not found.";
    }
    if (config.count("model_name")) {
      CHECK(config["model_name"].is<std::string>());
      this->model_name_ = config["model_name"].get<std::string>();
    } else {
      CHECK(partial_update) << "Key \"model_name\" not found.";
    }
    if (config.count("top_p")) {
      CHECK(config["top_p"].is<double>());
      this->top_p_ = config["top_p"].get<double>();
    } else {
      CHECK(partial_update) << "Key \"top_p\" not found.";
    }
    if (config.count("mean_gen_len")) {
      CHECK(config["mean_gen_len"].is<int64_t>());
      this->mean_gen_len_ = config["mean_gen_len"].get<int64_t>();
    } else {
      CHECK(partial_update) << "Key \"mean_gen_len\" not found.";
    }
    // NOTE: for backward compact
    // max gen len is optional
    if (config.count("max_gen_len")) {
      CHECK(config["max_gen_len"].is<int64_t>());
      this->max_gen_len_ = config["max_gen_len"].get<int64_t>();
    }
    if (config.count("shift_fill_factor")) {
      CHECK(config["shift_fill_factor"].is<double>());
      this->shift_fill_factor_ = config["shift_fill_factor"].get<double>();
    } else {
      CHECK(partial_update) << "Key \"shift_fill_factor\" not found.";
    }
    if (config.count("conv_template")) {
      ICHECK(config["conv_template"].is<std::string>());
      std::string conv_template = config["conv_template"].get<std::string>();
      this->conversation_ = Conversation::FromTemplate(conv_template);
      if (config.count("conv_config")) {
        // conv_config can override conv_template
        this->conversation_.LoadJSONOverride(config["conv_config"], true);
      }
    } else if (config.count("conv_config")) {
      // without conv template, conv_config needs to be a complete config
      this->conversation_.LoadJSONOverride(config["conv_config"], false);
    } else {
      CHECK(partial_update) << "Key \"conv_template\" and \"conv_config\" not found.";
    }
  }

  /*!
   * \brief Load JSON config and override options.
   * \param config_str A json config string that partially specifies some of the options.
   * \param partial_update Whether it's a partial update or full update, if set to true,
   *        we perform a partial update on some of the provided options; if set to false, all
   *        options must be provided.
   * \note This function overrides existing configurations.
   */
  void LoadJSONOverride(const std::string& config_str, bool partial_update = false) {
    picojson::value config_json;
    std::string err = picojson::parse(config_json, config_str);
    if (!err.empty()) {
      LOG(FATAL) << err;
      return;
    }
    LoadJSONOverride(config_json, partial_update);
  }

  std::string GetConfigJSON() const { return SerializeConfigToJSONValue().serialize(true); }

  /*!
   * \brief Reload model, tokenizers and configurations from the specified model path.
   * \param executable The module to reload.
   * \param model_path The path to search for models.
   * \param app_config_json The JSON string used to partially override the configuration loaded from
   * disk, default to empty string.
   */
  void Reload(TVMArgValue reload_lib, String model_path, String app_config_json = "") {
    // Step 1. Process config json string.
    {
      std::ifstream config_istream((model_path + "/mlc-chat-config.json").c_str());
      std::ostringstream config_ostream;
      ICHECK(config_istream);
      config_ostream << config_istream.rdbuf();
      std::string config_str = config_ostream.str();
      LoadJSONOverride(config_str, false);
      if (!app_config_json.empty()) {
        // Override configuration from app_config_json.
        LoadJSONOverride(app_config_json, true);
      }
    }
    // Step 2. Set tokenizer.
    this->tokenizer_ = TokenizerFromPath(model_path);
    // Step 3. Initialize vm, we use the packed function mechanism
    // so there is no explicit abi dependency on these extra
    // classes other than basic tvm runtime.
    this->ft_.Init(reload_lib, device_, this->num_shards_);
    auto fsample_topp_from_prob_ptr =
        tvm::runtime::Registry::Get("vm.builtin.sample_top_p_from_prob");
    ICHECK(fsample_topp_from_prob_ptr)
        << "Cannot find env function vm.builtin.sample_top_p_from_prob";
    fsample_topp_from_prob_ = *fsample_topp_from_prob_ptr;
    auto fsample_topp_from_logits_ptr =
        tvm::runtime::Registry::Get("vm.builtin.sample_top_p_from_logits");
    ICHECK(fsample_topp_from_logits_ptr)
        << "Cannot find env function vm.builtin.sample_top_p_from_logits";
    fsample_topp_from_logits_ = *fsample_topp_from_logits_ptr;
    // Step 4. Load params in nd-array cache.
    this->params_ = ft_.LoadParams(model_path, device_);
    // Step 5. KV cache creation.
    this->kv_cache_ = ft_.create_kv_cache_func_();
    // Step 6. Pre-allocate fixed size ndarray
    this->temperature_arr_ = NDArray::Empty({}, DataType::Float(32), device_);
    float temperature = static_cast<float>(this->temperature_);
    this->temperature_arr_.CopyFromBytes(&temperature, sizeof(float));
    if (ft_.use_disco) {
      Device null_device{DLDeviceType(0), 0};
      this->input_tokens_decode_ =
          Downcast<DRef>(ft_.Empty(ShapeTuple({1, 1}), DataType::Int(32), null_device));
    }
    // Step 7. Reset chat
    this->ResetChat();
  }

  void ResetChat() {
    // TODO(mlc-team): add conversation_.Reset to preserve system prompt
    // and initial message.
    // this->conversation_ = Conversation::Create(this->conversation_.conv_template);
    this->conversation_.Reset();
    this->ResetRuntimeStats();
    this->ResetKVCache();
    this->total_seq_len_ = 0;
  }

  /*! \brief reset the runtime stats. */
  void ResetRuntimeStats() {
    this->prefill_total_tokens = 0;
    this->decode_total_tokens = 0;
    this->embed_total_time = 0;
    this->prefill_total_time = 0;
    this->decode_total_time = 0;
    this->sample_total_time = 0;
  }

  static std::string GetConcatPrompt(const std::vector<std::string>& prompt_array,
                                     size_t prefix_end, size_t suffix_start) {
    std::ostringstream os;
    for (size_t i = 0; i < prefix_end; ++i) {
      os << prompt_array[i];
    }
    for (size_t i = suffix_start; i < prompt_array.size(); ++i) {
      os << prompt_array[i];
    }
    return os.str();
  }

  /**
   * \brief Get input tokens based on history
   * \param place_in_prompt The place of the input message in the prompt.
   */
  std::vector<int32_t> GetInputTokens(PlaceInPrompt place_in_prompt = PlaceInPrompt::kAll) {
    std::vector<int32_t> tokens;
    std::vector<std::string> prompts;

    if (this->total_seq_len_ == 0) {
      prompts = this->conversation_.GetPromptArray(place_in_prompt);
      if (this->conversation_.add_bos) {
        tokens.insert(tokens.begin(), bos_token_id_);
      }
      if (this->conversation_.prefix_tokens.size() != 0) {
        tokens.insert(tokens.begin(), this->conversation_.prefix_tokens.begin(),
                      this->conversation_.prefix_tokens.end());
      }
    } else {
      prompts = this->conversation_.GetPromptArrayLastRound(place_in_prompt);
    }
    // first try to encode all
    std::string all_prompt = GetConcatPrompt(prompts, 0, 0);
    std::vector<int32_t> encoded = this->tokenizer_->Encode(all_prompt);
    tokens.insert(tokens.end(), encoded.begin(), encoded.end());
    if (this->total_seq_len_ + tokens.size() + this->mean_gen_len_ < this->max_window_size_) {
      return tokens;
    }
    // need shift window and re-encode
    this->total_seq_len_ = 0;
    this->ResetKVCache();
    tokens.clear();
    if (this->conversation_.add_bos) {
      tokens.insert(tokens.begin(), bos_token_id_);
    }
    if (this->conversation_.prefix_tokens.size() != 0) {
      tokens.insert(tokens.begin(), this->conversation_.prefix_tokens.begin(),
                    this->conversation_.prefix_tokens.end());
    }
    std::vector<std::string> all_prompts = this->conversation_.GetPromptArray();
    // get estimate of the fragment
    size_t ctx_length = this->tokenizer_->Encode(all_prompts[0]).size();
    size_t start_re_encode_pos = 0;
    for (int i = all_prompts.size() - 1; i > 0; --i) {
      ctx_length += this->tokenizer_->Encode(all_prompts[i]).size();
      if (ctx_length >= this->shift_fill_factor_ * this->max_window_size_ &&
          i + 2 < all_prompts.size()) {
        start_re_encode_pos = i;
        break;
      }
    }
    // keep system
    if (this->conversation_.system.empty()) {
      all_prompt = GetConcatPrompt(all_prompts, 0, start_re_encode_pos);
    } else {
      all_prompt = GetConcatPrompt(all_prompts, 1, start_re_encode_pos);
    }
    encoded = this->tokenizer_->Encode(all_prompt);
    tokens.insert(tokens.end(), encoded.begin(), encoded.end());
    if (tokens.size() >= this->max_window_size_) {
      LOG(WARNING)
          << "The prompt tokens are more than `max_window_size`, the input will be truncated.";
      ICHECK_GT(this->max_window_size_, this->mean_gen_len_);
      std::vector<int32_t> truncated_tokens(
          tokens.end() - (this->max_window_size_ - this->mean_gen_len_), tokens.end());
      return truncated_tokens;
    } else if (tokens.size() + this->mean_gen_len_ >= this->max_window_size_) {
      LOG(WARNING)
          << "The prompt tokens are too long and the generated text may be incomplete, due to "
             "limited `max_window_size`. ";
    }
    return tokens;
  }

  // get statically allocated input token
  NDArray GetInputTokenNDArray(const std::vector<int32_t>& token_ids) {
    // try realloc
    if (!input_token_ids_.defined()) {
      int64_t init_size = 2048;
      while (init_size < static_cast<int64_t>(token_ids.size())) {
        init_size *= 2;
      }
      input_token_ids_ = NDArray::Empty({1, init_size}, DataType::Int(32), device_);
    } else {
      int64_t init_size = input_token_ids_->shape[1];
      while (init_size < static_cast<int64_t>(token_ids.size())) {
        init_size *= 2;
      }
      if (init_size != input_token_ids_->shape[1]) {
        input_token_ids_ = NDArray::Empty({1, init_size}, DataType::Int(32), device_);
      }
    }
    ICHECK_LE(token_ids.size(), input_token_ids_->shape[1]) << "Input tokens exceed window size";
    NDArray view = input_token_ids_.CreateView(
        ShapeTuple({1, static_cast<int64_t>(token_ids.size())}), input_token_ids_->dtype);
    if (token_ids.size() > 0) {
      view.CopyFromBytes(token_ids.data(), token_ids.size() * sizeof(int32_t));
    }
    return view;
  }

  std::vector<int32_t> PrepareBeforeEmbedding(std::string inp, bool append_conversation = true,
                                              PlaceInPrompt place_in_prompt = PlaceInPrompt::kAll) {
    if (conversation_.separator_style == SeparatorStyle::kLM ||
        conversation_.separator_style == SeparatorStyle::kCodeCompletion) {
      this->ResetChat();
    }
    if (reset_stats_per_prefill_) {
      this->ResetRuntimeStats();
    }
    output_ids_.clear();
    appeared_token_ids_.clear();
    output_message_.clear();
    stop_triggered_ = false;
    if (append_conversation) {
      conversation_.AppendMessage(conversation_.roles[0], inp);
      conversation_.AppendReplyHeader(conversation_.roles[1]);
    }

    return this->GetInputTokens(place_in_prompt);
  }

  /*!
   * \brief Given the text input, generate the embedding of the tokenized prompt.
   * \param inp The input text string.
   * \param append_conversation Whether to append the input message to conversation.
   * \param place_in_prompt The place of the input message in the prompt.
   * \return the embedding of the tokenized prompt.
   */
  ObjectRef EmbedStep(std::string inp, bool append_conversation = true,
                      PlaceInPrompt place_in_prompt = PlaceInPrompt::kAll) {
    std::vector<int32_t> prompt_tokens =
        PrepareBeforeEmbedding(inp, append_conversation, place_in_prompt);
    int64_t token_len = static_cast<int64_t>(prompt_tokens.size());
    if (token_len == 0) {
      return NDArray::Empty({}, DataType::Float(32), device_);
    }

    CHECK(ft_.embed_func_.defined())
        << "In order to use the embedding functionality, make sure you "
           "build the model in MLC-LLM with `sep_embed` option on.";
    auto tstart = std::chrono::high_resolution_clock::now();

    NDArray input_data = this->GetInputTokenNDArray(prompt_tokens);
    ObjectRef embedding = ft_.embed_func_(ft_.CopyToWorker0(input_data), params_);

    int32_t new_seq_len = total_seq_len_ + token_len;
    total_seq_len_ = new_seq_len;

    auto tend = std::chrono::high_resolution_clock::now();

    this->embed_total_time += static_cast<double>((tend - tstart).count()) / 1e9;

    return embedding;
  }

  /*!
   * \brief Prefill given embeddings. Can optionally decode the output next token.
   * \param embedding The embedding to prefill with.
   * \param decode_next_token Whether to decode next token.
   */
  void PrefillWithEmbedStep(NDArray embedding, bool decode_next_token = true) {
    if (ft_.use_disco) {
      LOG(FATAL) << "NotImplementedError: Distributed inference is not supported for this model";
      throw;
    }
    if (embedding.Shape().size() == 0) {
      return;
    }
    auto tstart = std::chrono::high_resolution_clock::now();
    int64_t token_len = embedding.Shape()[1];
    NDArray logits_on_device = this->ForwardEmbeddings(embedding, total_seq_len_);

    if (!decode_next_token) {
      auto tend = std::chrono::high_resolution_clock::now();
      this->prefill_total_time += static_cast<double>((tend - tstart).count()) / 1e9;
      this->prefill_total_tokens += token_len;
      return;
    }

    int32_t next_token = this->SampleTokenFromLogits(logits_on_device, temperature_, top_p_);

    auto tend = std::chrono::high_resolution_clock::now();

    this->prefill_total_time += static_cast<double>((tend - tstart).count()) / 1e9;
    this->prefill_total_tokens += token_len;
    this->ProcessNextToken(next_token);
  }

  /*!
   * \brief Generate the next token given a prompt. Can optionally decode the output next token.
   * \param inp The input text string.
   * \param append_conversation Whether to append the input message to conversation.
   * \param decode_next_token Whether to decode next token.
   * \param place_in_prompt The place of the input message in the prompt.
   */
  void PrefillStep(std::string inp, bool append_conversation = true, bool decode_next_token = true,
                   PlaceInPrompt place_in_prompt = PlaceInPrompt::kAll) {
    if (ft_.embed_func_.defined() && ft_.prefill_with_embed_func_.defined()) {
      // Temporarily placed inside `PrefillStep` for compatibility in transition.
      // Will be separated out in the future.
      if (ft_.use_disco) {
        LOG(FATAL) << "NotImplementedError: Distributed inference is not supported for this model";
      }
      NDArray embedding = Downcast<NDArray>(EmbedStep(inp, append_conversation, place_in_prompt));
      PrefillWithEmbedStep(embedding, decode_next_token);
      return;
    }

    std::vector<int32_t> prompt_tokens =
        this->PrepareBeforeEmbedding(inp, append_conversation, place_in_prompt);
    int64_t token_len = static_cast<int64_t>(prompt_tokens.size());
    if (token_len == 0) return;
    if (ft_.use_disco) {
      // exclude load shard time from prefill
      this->ft_.sess->SyncWorker(0);
    }
    auto tstart = std::chrono::high_resolution_clock::now();

    int32_t new_seq_len = total_seq_len_ + token_len;
    NDArray logits_on_device = this->ForwardTokens(prompt_tokens, new_seq_len);
    total_seq_len_ = new_seq_len;

    if (!decode_next_token) {
      auto tend = std::chrono::high_resolution_clock::now();
      this->prefill_total_time += static_cast<double>((tend - tstart).count()) / 1e9;
      this->prefill_total_tokens += token_len;
      return;
    }

    int32_t next_token = this->SampleTokenFromLogits(logits_on_device, temperature_, top_p_);

    auto tend = std::chrono::high_resolution_clock::now();

    this->prefill_total_time += static_cast<double>((tend - tstart).count()) / 1e9;
    this->prefill_total_tokens += token_len;
    this->ProcessNextToken(next_token);
  }

  void DecodeStep() {
    ICHECK(!output_ids_.empty());
    int32_t last_token = output_ids_.back();
    tvm::runtime::NDArray input_data = GetInputTokenNDArray({last_token});

    auto tstart = std::chrono::high_resolution_clock::now();

    NDArray logits_on_device = this->ForwardTokens({last_token}, total_seq_len_ + 1);
    total_seq_len_ += 1;

    int32_t next_token = this->SampleTokenFromLogits(logits_on_device, temperature_, top_p_);

    auto tend = std::chrono::high_resolution_clock::now();

    this->decode_total_time += static_cast<double>((tend - tstart).count()) / 1e9;
    this->decode_total_tokens += 1;
    this->ProcessNextToken(next_token);
  }

  bool Stopped() { return stop_triggered_; }

  std::string GetMessage() {
    // remove non-utf8 characters
    size_t effective_end = FindEffectiveUTF8Pos(output_message_);
    while (effective_end > 0 && output_message_[effective_end - 1] == '\n') {
      --effective_end;
    }
    size_t effective_begin = 0;
    while (effective_begin < effective_end && output_message_[effective_begin] == ' ') {
      ++effective_begin;
    }
    std::string cropped_message =
        output_message_.substr(effective_begin, effective_end - effective_begin);
    return cropped_message;
  }

  // do some quick evaluation of the pipeline
  void Evaluate(int64_t token_len, int64_t generate_len) {
    this->ResetKVCache();
    std::vector<int32_t> tokens;
    for (int i = 0; i < token_len - 1; ++i) {
      tokens.push_back(2);
    }
    tokens.insert(tokens.begin(), bos_token_id_);

    std::vector<int32_t> first_sample_data = {6234};

    // warm up: skip first run
    this->ForwardTokens(tokens, token_len);
    this->ForwardTokens(first_sample_data, token_len + 1);
    this->ResetKVCache();

    // encoding
    auto encoding_start = std::chrono::high_resolution_clock::now();
    this->ForwardTokens(tokens, token_len);
    TVMSynchronize(device_.device_type, device_.device_id, nullptr);
    auto encoding_end = std::chrono::high_resolution_clock::now();
    double encoding_ms = static_cast<double>((encoding_end - encoding_start).count()) / 1e6;
    LOG(INFO) << "encoding-time=" << encoding_ms << "ms, ";

    double decoding_ms_total = 0;
    // start encoding
    for (int i = 0; i < generate_len; ++i) {
      auto decoding_start = std::chrono::high_resolution_clock::now();
      this->UpdateLogitsOrProbOnCPUSync(this->ForwardTokens(first_sample_data, token_len + i + 1));
      TVMSynchronize(device_.device_type, device_.device_id, nullptr);
      auto decoding_end = std::chrono::high_resolution_clock::now();
      double decoding_ms = static_cast<double>((decoding_end - decoding_start).count()) / 1e6;
      decoding_ms_total += decoding_ms;
      LOG(INFO) << "[i: " << token_len + i + 1 << "] decoding-time=" << decoding_ms << "ms"
                << " tok/s: " << 1000.0 * (i + 1) / decoding_ms_total << ".";
    }
  }

  std::string RawGenerate(std::string prompt, int64_t generate_len) {
    CHECK_GE(generate_len, 0) << "The input generate is expected to be non-negative.";

    this->ResetKVCache();
    this->ResetRuntimeStats();

    std::vector<int32_t> tokens = tokenizer_->Encode(prompt);
    int64_t input_length = tokens.size();

    NDArray logits_on_device;
    // prefill
    {
      auto tstart = std::chrono::high_resolution_clock::now();
      logits_on_device = this->ForwardTokens(tokens, tokens.size());
      tokens.push_back(this->SampleTokenFromLogits(logits_on_device, temperature_, top_p_));
      auto tend = std::chrono::high_resolution_clock::now();

      this->prefill_total_time = static_cast<double>((tend - tstart).count()) / 1e9;
      this->prefill_total_tokens = input_length;
    }

    // decode
    {
      auto tstart = std::chrono::high_resolution_clock::now();
      for (int64_t len = 1; len < generate_len; ++len) {
        logits_on_device = this->ForwardTokens({tokens.back()}, tokens.size());
        tokens.push_back(this->SampleTokenFromLogits(logits_on_device, temperature_, top_p_));
      }
      auto tend = std::chrono::high_resolution_clock::now();

      this->decode_total_time = static_cast<double>((tend - tstart).count()) / 1e9;
      this->decode_total_tokens = generate_len;
    }

    std::string output = tokenizer_->Decode({tokens.begin() + input_length, tokens.end()});
    return output;
  }

 private:
  picojson::value SerializeConfigToJSONValue() const {
    picojson::object config;
    config["temperature"] = picojson::value(this->temperature_);
    config["repetition_penalty"] = picojson::value(this->repetition_penalty_);
    config["top_p"] = picojson::value(this->top_p_);
    config["mean_gen_len"] = picojson::value(this->mean_gen_len_);
    config["max_gen_len"] = picojson::value(this->max_gen_len_);
    config["shift_fill_factor"] = picojson::value(this->shift_fill_factor_);
    config["conv_config"] = this->conversation_.SerializeToJSON();
    return picojson::value(config);
  }
  /*!
   * \brief Sample output token from logits on device
   */
  int32_t SampleTokenFromLogits(NDArray logits_on_device, float temperature, float top_p) {
    if (repetition_penalty_ == 1.0f) {
      if (temperature_ < 1e-6f) {
        this->UpdateLogitsOrProbOnCPUSync(logits_on_device);
      } else {
        this->UpdateLogitsOrProbOnCPUSync(this->Softmax(logits_on_device, temperature_));
      }
    } else {
      this->UpdateLogitsOrProbOnCPUSync(logits_on_device);
      this->ApplyRepetitionPenaltyOnCPU();
      if (temperature_ >= 1e-6f) {
        this->ApplySoftmaxWithTemperatureOnCPU();
      }
    }
    auto tstart = std::chrono::high_resolution_clock::now();
    int next_token;
    if (temperature_ < 1e-6f) {
      next_token = this->SampleFromLogitsOnCPU();
    } else {
      next_token = this->SampleFromProbOnCPU();
    }
    auto tend = std::chrono::high_resolution_clock::now();
    this->sample_total_time += static_cast<double>((tend - tstart).count()) / 1e9;
    return next_token;
  }

  /*!
   * \brief Add a generated token and check for stop condition.
   * \param next_token The next token.
   */
  void ProcessNextToken(int32_t next_token) {
    ICHECK(!stop_triggered_) << "Cannot call process when it is stopped";

    stop_triggered_ =
        std::any_of(this->conversation_.stop_tokens.begin(), this->conversation_.stop_tokens.end(),
                    [next_token](int32_t token) { return token == next_token; });

    if (!stop_triggered_) {
      output_ids_.push_back(next_token);
      appeared_token_ids_.insert(next_token);
    }

    output_message_ = tokenizer_->Decode(output_ids_);

    if (!conversation_.stop_str.empty()) {
      size_t stop_pos = output_message_.rfind(conversation_.stop_str);
      if (stop_pos != std::string::npos) {
        stop_triggered_ = true;
        if (ft_.support_backtracking_kv_) {
          // back tracking, find the first set of token that is smaller
          // than the length
          size_t backoff = 0;
          for (; backoff < output_ids_.size(); ++backoff) {
            output_ids_.pop_back();
            output_message_ = tokenizer_->Decode(output_ids_);
            if (output_message_.length() <= stop_pos) break;
          }
          // resize kv to remove the context
          ft_.fkvcache_array_popn_(kv_cache_, backoff);
          total_seq_len_ -= backoff;
        }
      }
    }

    if (static_cast<int64_t>(output_ids_.size()) >= max_gen_len_) {
      stop_triggered_ = true;
    } else if (total_seq_len_ >= max_window_size_) {
      stop_triggered_ = true;
    }
    if (stop_triggered_) {
      conversation_.FinishReply(output_message_);
    }
  }

  // run forward compute
  NDArray ForwardTokens(std::vector<int32_t> input_tokens, int64_t cur_pos) {
    ObjectRef ret{nullptr};
    if (input_tokens.size() > 1 && ft_.prefill_func_.defined()) {
      ObjectRef input_data = ft_.CopyToWorker0(this->GetInputTokenNDArray(input_tokens));
      ShapeTuple cur_pos_shape = ShapeTuple({cur_pos});
      ret = ft_.prefill_func_(input_data, cur_pos_shape, kv_cache_, params_);
    } else {
      // running decode function when prefill is not available
      for (int i = 0; i < input_tokens.size(); ++i) {
        ObjectRef input_data;
        if (ft_.use_disco) {
          ft_.sess->CopyToWorker0(this->GetInputTokenNDArray({input_tokens[i]}),
                                  input_tokens_decode_);
          input_data = input_tokens_decode_;
        } else {
          input_data = ft_.CopyToWorker0(this->GetInputTokenNDArray({input_tokens[i]}));
        }
        int64_t pos = cur_pos + i + 1 - input_tokens.size();
        ShapeTuple pos_shape = ShapeTuple({cur_pos});
        ret = ft_.decode_func_(input_data, pos_shape, kv_cache_, params_);
      }
    }
    if (ft_.use_disco) {
      Array<ObjectRef> result = Downcast<DRef>(ret)->DebugGetFromRemote(0);
      return Downcast<NDArray>(result[0]);
    } else {
      return Downcast<Array<NDArray>>(ret)[0];
    }
  }

  // run forward compute with embeddings
  NDArray ForwardEmbeddings(NDArray embeddings, int64_t cur_pos) {
    if (ft_.use_disco) {
      LOG(FATAL) << "NotImplementedError: Distributed inference is not supported for this model";
      throw;
    }
    Array<ObjectRef> ret;
    CHECK(ft_.prefill_with_embed_func_.defined());
    ret = ft_.prefill_with_embed_func_(embeddings, ShapeTuple({cur_pos}), kv_cache_, params_);
    return Downcast<NDArray>(ret[0]);
  }

  NDArray Softmax(NDArray input, float temperature) {
    NDArray ret;
    ret = ft_.softmax_func_(input, temperature_arr_);
    return ret;
  }

  void ApplyRepetitionPenaltyOnCPU() {
    CHECK(logits_on_cpu_.defined()) << "Logits on CPU not defined!";
    CHECK(logits_on_cpu_.DataType() == DataType::Float(32)) << "Logits data type is not float32!";
    float* logits_raw_data = static_cast<float*>(logits_on_cpu_->data);
    for (const int32_t& token_id : this->appeared_token_ids_) {
      if (logits_raw_data[token_id] <= 0) {
        logits_raw_data[token_id] *= this->repetition_penalty_;
      } else {  // logits > 0
        logits_raw_data[token_id] /= this->repetition_penalty_;
      }
    }
  }

  void ApplySoftmaxWithTemperatureOnCPU() {
    CHECK(logits_on_cpu_.defined()) << "Logits on CPU not defined!";
    CHECK(logits_on_cpu_.DataType() == DataType::Float(32)) << "Logits data type is not float32!";
    int vocab_size = logits_on_cpu_->shape[logits_on_cpu_->ndim - 1];
    float* logits_raw_data = static_cast<float*>(logits_on_cpu_->data);
    float m = std::numeric_limits<float>::min();
    float inv_temp = 1.0f / this->temperature_;
    double d = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
      float x = logits_raw_data[i] * inv_temp;
      float m_prev = m;
      m = std::max(m, x);
      d = d * std::exp(m_prev - m) + std::exp(x - m);
    }
    for (int i = 0; i < vocab_size; ++i) {
      float x = logits_raw_data[i] * inv_temp;
      logits_raw_data[i] = std::exp(x - m) / d;
    }
  }

  void UpdateLogitsOrProbOnCPUSync(NDArray logits_or_prob) {
    if (!logits_on_cpu_.defined()) {
      logits_on_cpu_ = logits_or_prob.CopyTo(DLDevice{kDLCPU, 0});
    } else {
      ICHECK_EQ(logits_on_cpu_->shape[0], logits_or_prob->shape[0])
          << "Expect size of logits remain unchanged";
      logits_on_cpu_.CopyFrom(logits_or_prob);
    }
    TVMSynchronize(device_.device_type, device_.device_id, nullptr);
  }

  // Clear kv cache
  void ResetKVCache() { ft_.reset_kv_cache_func_(kv_cache_); }

  void ProcessSystemPrompts() {
    this->PrefillStep(/*inp=*/"", /*append_conversation=*/false, /*decode_next_token=*/false);
  }

  // Utils
  static double GetRandomNumber() { return RandomGenerator::GetInstance().GetRandomNumber(); }

  int32_t SampleFromLogitsOnCPU() {
    ICHECK(logits_on_cpu_.defined()) << "logits_on_cpu_ is not defined";
    ICHECK_EQ(logits_on_cpu_->ndim, 3) << "logits_on_cpu_ should be 3D";
    ICHECK_EQ(logits_on_cpu_->shape[0], 1) << "logits_on_cpu_ should be 1 batch";
    return fsample_topp_from_logits_(logits_on_cpu_, temperature_, top_p_, GetRandomNumber());
  }

  int32_t SampleFromProbOnCPU() {
    ICHECK(logits_on_cpu_.defined()) << "logits_on_cpu_ is not defined";
    ICHECK_EQ(logits_on_cpu_->ndim, 3) << "logits_on_cpu_ should be 3D";
    ICHECK_EQ(logits_on_cpu_->shape[0], 1) << "logits_on_cpu_ should be 1 batch";
    return fsample_topp_from_prob_(logits_on_cpu_, top_p_, GetRandomNumber());
  }

  //----------------------------
  // Statistics
  //----------------------------
  bool reset_stats_per_prefill_ = true;
  double embed_total_time = 0;
  double decode_total_time = 0;
  double sample_total_time = 0;
  double prefill_total_time = 0;
  int64_t decode_total_tokens = 0;
  int64_t prefill_total_tokens = 0;
  //----------------------------
  // Conversation
  //----------------------------
  // model name
  std::string model_name_;
  // conversation
  Conversation conversation_;
  // total sequence len,
  int64_t total_seq_len_{0};
  // max window size, mean generation length
  int64_t max_window_size_{768}, mean_gen_len_{128}, max_gen_len_{512};
  // size of the vocab table
  int64_t vocab_size_;
  // number of shards in distributed inference
  int64_t num_shards_;
  // shift window fill factor
  double shift_fill_factor_{0.3};
  // temperature
  double temperature_{0.8};
  // pre-allocated ndarray for temperature
  NDArray temperature_arr_;
  // repetition penalty
  double repetition_penalty_{1.0};
  // top_p
  double top_p_{0.95};
  // output ids till now (refresh after encoding step)
  std::vector<int32_t> output_ids_;
  // appeared token ids till now (refresh after encoding step)
  std::unordered_set<int32_t> appeared_token_ids_;
  // output message till now (refresh after encoding step)
  std::string output_message_;
  // Whether encounter stop str
  bool stop_triggered_{false};
  //----------------------------
  // Tokenizer
  //----------------------------
  // internal tokenizer
  std::unique_ptr<Tokenizer> tokenizer_;
  // bos token
  int32_t bos_token_id_{1};
  // eos token id
  int32_t eos_token_id_{2};
  //----------------------------
  // TVM related states
  //----------------------------
  // runtime device
  Device device_;

  FunctionTable ft_;
  // sample top p from logits
  PackedFunc fsample_topp_from_logits_;
  // sample top p from prob
  PackedFunc fsample_topp_from_prob_;
  // input token id
  NDArray input_token_ids_{nullptr};
  // local params
  ObjectRef params_;
  // KV cache
  ObjectRef kv_cache_;
  // Temp logits on cpu
  NDArray logits_on_cpu_{nullptr};
  // pre-allocated ndarray for decode function's input tokens
  DRef input_tokens_decode_{nullptr};
};

/*!
 * \brief A chat module implementation that exposes
 *  the functions as tvm::runtime::Module.
 *
 * We do it so that the module is accessible to any
 * language that tvm runtime can access.
 */
class LLMChatModule : public ModuleNode {
 public:
  // clear global memory manager
  static void ClearGlobalMemoryManager() {
    // Step 0. Clear the previously allocated memory.
    const PackedFunc* fclear_memory_manager =
        tvm::runtime::Registry::Get("vm.builtin.memory_manager.clear");
    ICHECK(fclear_memory_manager) << "Cannot find env function vm.builtin.memory_manager.clear";
    (*fclear_memory_manager)();
  }

  // overrides
  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "reload") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        chat_ = nullptr;
        ClearGlobalMemoryManager();
        chat_ = std::make_unique<LLMChat>(LLMChat(device_));
        ICHECK(2 <= args.size() && args.size() <= 3);
        if (args.size() == 2) {
          // args: reload_lib, model_path
          chat_->Reload(args[0], args[1]);
        } else if (args.size() == 3) {
          // args: reload_lib, model_path, app_config_json (used for overriding config)
          chat_->Reload(args[0], args[1], args[2]);
        }
      });
    } else if (name == "unload") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        chat_ = nullptr;
        ClearGlobalMemoryManager();
      });
    } else if (name == "evaluate") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.size(), 2);
        GetChat()->Evaluate(args[0], args[1]);
      });
    } else if (name == "raw_generate") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.size(), 2);
        std::string s = GetChat()->RawGenerate(args[0], args[1]);
        *rv = s;
      });
    } else if (name == "prefill") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK(1 <= args.size() && args.size() <= 3);
        if (args.size() == 1) {
          // args: inp (with decode_next_token = true, place_in_prompt = kAll)
          GetChat()->PrefillStep(args[0]);
        } else if (args.size() == 2) {
          // args: inp, decode_next_token (with place_in_prompt = kAll)
          GetChat()->PrefillStep(args[0], true, args[1]);
        } else if (args.size() == 3) {
          // args: inp, decode_next_token, place_in_prompt
          PlaceInPrompt place_in_prompt = static_cast<PlaceInPrompt>(static_cast<int>(args[2]));
          GetChat()->PrefillStep(args[0], true, args[1], place_in_prompt);
        }
      });
    } else if (name == "embed") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK(1 <= args.size() && args.size() <= 2);
        if (args.size() == 1) {
          // args: inp (with place_in_prompt = kAll)
          *rv = GetChat()->EmbedStep(args[0]);
        } else if (args.size() == 2) {
          // args: inp, place_in_prompt
          PlaceInPrompt place_in_prompt = static_cast<PlaceInPrompt>(static_cast<int>(args[1]));
          *rv = GetChat()->EmbedStep(args[0], true, place_in_prompt);
        }
      });
    } else if (name == "prefill_with_embed") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK(1 <= args.size() && args.size() <= 2);
        if (args.size() == 1) {
          // args: embedding (with decode_next_token = true)
          GetChat()->PrefillWithEmbedStep(args[0]);
        } else if (args.size() == 2) {
          // args: embedding, decode_next_token
          GetChat()->PrefillWithEmbedStep(args[0], args[1]);
        }
      });
    } else if (name == "decode") {
      return PackedFunc(
          [this, sptr_to_self](TVMArgs args, TVMRetValue* rv) { GetChat()->DecodeStep(); });
    } else if (name == "reset_chat") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.size(), 0);
        GetChat()->ResetChat();
      });
    } else if (name == "load_json_override") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.size(), 2);
        std::string config_str = args[0];
        bool partial_update = args[1];
        GetChat()->LoadJSONOverride(config_str, partial_update);
      });
    } else if (name == "get_role0") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        *rv = GetChat()->conversation_.roles[0];
      });
    } else if (name == "get_role1") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        *rv = GetChat()->conversation_.roles[1];
      });
    } else if (name == "stopped") {
      return PackedFunc(
          [this, sptr_to_self](TVMArgs args, TVMRetValue* rv) { *rv = GetChat()->Stopped(); });
    } else if (name == "get_message") {
      return PackedFunc(
          [this, sptr_to_self](TVMArgs args, TVMRetValue* rv) { *rv = GetChat()->GetMessage(); });
    } else if (name == "runtime_stats_text") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        *rv = GetChat()->RuntimeStatsText();
      });
    } else if (name == "reset_runtime_stats") {
      return PackedFunc(
          [this, sptr_to_self](TVMArgs args, TVMRetValue* rv) { GetChat()->ResetRuntimeStats(); });
    } else if (name == "get_config_json") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        *rv = GetChat()->GetConfigJSON();
      });
    } else if (name == "process_system_prompts") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        GetChat()->ProcessSystemPrompts();
      });
    } else {
      return PackedFunc(nullptr);
    }
  }

  void Init(DLDevice device) { device_ = device; }

  LLMChat* GetChat() {
    ICHECK(chat_ != nullptr) << "Chat is not initialized via reload";
    return chat_.get();
  }

  const char* type_key() const final { return "mlc.llm_chat"; }

 private:
  std::unique_ptr<LLMChat> chat_ = nullptr;
  DLDevice device_;
};

std::vector<std::string> CountUTF8(const std::string& s) {
  // assume that the string is always valid utf8
  std::vector<std::string> ret;
  for (size_t pos = 0; pos < s.size();) {
    if ((s[pos] & 0x80) == 0x00) {
      ret.push_back(s.substr(pos, 1));
      pos += 1;
    } else if (pos + 1 < s.size() && (s[pos] & 0xE0) == 0xC0 && (s[pos + 1] & 0xC0) == 0x80) {
      ret.push_back(s.substr(pos, 2));
      pos += 2;
    } else if (pos + 1 < s.size() && (s[pos] & 0xF0) == 0xE0 && (s[pos + 1] & 0xC0) == 0x80 &&
               (s[pos + 2] & 0xC0) == 0x80) {
      ret.push_back(s.substr(pos, 3));
      pos += 3;
    } else if (pos + 2 < s.size() && (s[pos] & 0xF8) == 0xF0 && (s[pos + 1] & 0xC0) == 0x80 &&
               (s[pos + 2] & 0xC0) == 0x80 && (s[pos + 3] & 0xC0) == 0x80) {
      ret.push_back(s.substr(pos, 4));
      pos += 4;
    } else {
      LOG(FATAL) << "Invalid UTF8 string";
    }
  }
  return std::move(ret);
}

/*!
 * \brief Get the diff of new message and current message (the delta message).
 * \param curr_message The current message.
 * \param new_message The new message
 * \return The delta message.
 * \note The main complication here is that new_mdg can be different from previous message, so we
 need to find the diff, delete previous messages that are different, then print it out.
 This logic is only needed for simple stdout.

 For UI apps that can directly update output text we can simply do last_reply.text =
 chat->GetMessage();
 */
std::string GetDeltaMessage(std::string curr_message, std::string new_message) {
  std::vector<std::string> cur_utf8_chars = CountUTF8(curr_message);
  std::vector<std::string> new_utf8_chars = CountUTF8(new_message);
  // Step 1. Find the index of the first UTF8 char that differs
  size_t pos = std::mismatch(cur_utf8_chars.begin(), cur_utf8_chars.end(), new_utf8_chars.begin(),
                             new_utf8_chars.end())
                   .first -
               cur_utf8_chars.begin();
  // Step 2. Delete the previous message since `pos`
  std::string print = "";
  for (size_t j = pos; j < cur_utf8_chars.size(); ++j) {
    print += "\b \b";
  }
  // Step 3. Print the new message since `pos`
  for (size_t j = pos; j < new_utf8_chars.size(); ++j) {
    print += new_utf8_chars[j];
  }
  return print;
}

// register as a system function that can be queried
TVM_REGISTER_GLOBAL("mlc.get_delta_message").set_body_typed(GetDeltaMessage);

tvm::runtime::Module CreateChatModule(DLDevice device) {
  ObjectPtr<LLMChatModule> n = make_object<LLMChatModule>();
  n->Init(device);
  return Module(n);
}

// register as a system function that can be queried
TVM_REGISTER_GLOBAL("mlc.llm_chat_create").set_body_typed([](int device_type, int device_id) {
  return CreateChatModule(DLDevice{static_cast<DLDeviceType>(device_type), device_id});
});

TVM_REGISTER_GLOBAL("mlc.random.set_seed").set_body_typed([](int seed) {
  RandomGenerator::GetInstance().SetSeed(seed);
});

}  // namespace llm
}  // namespace mlc
