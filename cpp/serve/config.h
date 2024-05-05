/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/config.h
 */
#ifndef MLC_LLM_SERVE_CONFIG_H_
#define MLC_LLM_SERVE_CONFIG_H_

#include <picojson.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/object.h>

#include <optional>

#include "../metadata/model.h"
#include "../support/result.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm;
using namespace tvm::runtime;

/****************** GenerationConfig ******************/

/*! \brief The response format of a request. */
struct ResponseFormat {
  String type = "text";
  Optional<String> schema = NullOpt;
};

/*! \brief The generation configuration of a request. */
class GenerationConfigNode : public Object {
 public:
  int n = 1;
  double temperature = 0.8;
  double top_p = 0.95;
  double frequency_penalty = 0.0;
  double presence_penalty = 0.0;
  double repetition_penalty = 1.0;
  bool logprobs = false;
  int top_logprobs = 0;
  std::vector<std::pair<int, float>> logit_bias;
  int seed;
  bool ignore_eos = false;

  int max_tokens = 128;
  Array<String> stop_strs;
  std::vector<int> stop_token_ids;

  ResponseFormat response_format;

  String AsJSONString() const;

  static constexpr const char* _type_key = "mlc.serve.GenerationConfig";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(GenerationConfigNode, Object);
};

class GenerationConfig : public ObjectRef {
 public:
  TVM_DLL explicit GenerationConfig(
      std::optional<int> n, std::optional<double> temperature, std::optional<double> top_p,
      std::optional<double> frequency_penalty, std::optional<double> presense_penalty,
      std::optional<double> repetition_penalty, std::optional<bool> logprobs,
      std::optional<int> top_logprobs, std::optional<std::vector<std::pair<int, float>>> logit_bias,
      std::optional<int> seed, std::optional<bool> ignore_eos, std::optional<int> max_tokens,
      std::optional<Array<String>> stop_strs, std::optional<std::vector<int>> stop_token_ids,
      std::optional<ResponseFormat> response_format, Optional<String> default_config_json_str);

  TVM_DLL explicit GenerationConfig(String config_json_str,
                                    Optional<String> default_config_json_str);

  /*! \brief Get the default generation config from the model config. */
  TVM_DLL static GenerationConfig GetDefaultFromModelConfig(const picojson::object& json);

  TVM_DEFINE_OBJECT_REF_METHODS(GenerationConfig, ObjectRef, GenerationConfigNode);
};

/****************** Engine config ******************/

/*!
 * \brief The engine mode in MLC LLM.
 * We provide three preset modes: "local", "interactive" and "server".
 * The default mode is "local".
 * The choice of mode decides the values of "max_batch_size", "max_total_sequence_length"
 * and "prefill_chunk_size" when they are not explicitly specified.
 * 1. Mode "local" refers to the local server deployment which has low
 * request concurrency. So the max batch size will be set to 4, and max
 * total sequence length and prefill chunk size are set to the context
 * window size (or sliding window size) of the model.
 * 2. Mode "interactive" refers to the interactive use of server, which
 * has at most 1 concurrent request. So the max batch size will be set to 1,
 * and max total sequence length and prefill chunk size are set to the context
 * window size (or sliding window size) of the model.
 * 3. Mode "server" refers to the large server use case which may handle
 * many concurrent request and want to use GPU memory as much as possible.
 * In this mode, we will automatically infer the largest possible max batch
 * size and max total sequence length.
 */
enum class EngineMode : int {
  kLocal = 0,
  kInteractive = 1,
  kServer = 2,
};

/*! \brief The speculative mode. */
enum class SpeculativeMode : int {
  /*! \brief Disable speculative decoding. */
  kDisable = 0,
  /*! \brief The normal speculative decoding (small draft) mode. */
  kSmallDraft = 1,
  /*! \brief The eagle-style speculative decoding. */
  kEagle = 2,
};

/*! \brief The kind of cache. */
enum class KVStateKind : int {
  kKVCache = 0,
  kRNNState = 1,
};

class InferrableEngineConfig;

/*! \brief The configuration of engine execution config. */
class EngineConfigNode : public Object {
 public:
  /*************** Models ***************/

  /*! \brief The path to the model directory. */
  String model;
  /*! \brief The path or identifier to the model library. */
  String model_lib;
  /*! \brief The path to the additional models' directories. */
  Array<String> additional_models;
  /*! \brief The path to the additional models' libraries. */
  Array<String> additional_model_libs;

  /*************** KV cache config and engine capacities ***************/

  /*!
   * \brief The engine mode in MLC LLM.
   * \sa EngineMode
   */
  EngineMode mode = EngineMode::kLocal;
  /*!
   * \brief A number in (0, 1) denoting the fraction of GPU memory used by the server in total.
   * It is used to infer to maximum possible KV cache capacity.
   * When it is unspecified, it defaults to 0.85.
   * Under mode "local" or "interactive", the actual memory usage may be
   * significantly smaller than this number. Under mode "server", the actual
   * memory usage may be slightly larger than this number.
   */
  float gpu_memory_utilization = 0.85;
  /*! \brief The number of consecutive tokens handled in each page in paged KV cache. */
  int kv_cache_page_size = 16;
  /*!
   * \brief The maximum number of sequences that are allowed to be
   * processed by the KV cache at any time.
   */
  int max_num_sequence = 4;
  /*! \brief The maximum length allowed for a single sequence in the engine. */
  int max_total_sequence_length = 4096;
  /*!
   * \brief The maximum total number of tokens whose KV data are allowed
   * to exist in the KV cache at any time.
   */
  int max_single_sequence_length = 4096;
  /*! \brief The maximum total sequence length in a prefill. */
  int prefill_chunk_size = 1024;
  /*! \brief The maximum history size for RNN state. KV cache does not need this. */
  int max_history_size = 0;
  /*! \brief The kind of cache. Whether it's KV cache or RNN state. */
  KVStateKind kv_state_kind = KVStateKind::kKVCache;

  /*************** Speculative decoding ***************/

  /*! \brief The speculative mode. */
  SpeculativeMode speculative_mode = SpeculativeMode::kDisable;
  /*! \brief The number of tokens to generate in speculative proposal (draft). */
  int spec_draft_length = 4;

  /*************** Debug ***************/
  bool verbose = false;

  TVM_DLL String AsJSONString() const;

  static constexpr const char* _type_key = "mlc.serve.EngineConfig";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(EngineConfigNode, Object);
};

class EngineConfig : public ObjectRef {
 public:
  /*! \brief Create EngineConfig from JSON object and inferred config. */
  TVM_DLL static EngineConfig FromJSONAndInferredConfig(
      const picojson::object& json, const InferrableEngineConfig& inferred_config);

  /*!
   * \brief Get all the models and model libs from the JSON string for engine initialization.
   * \return The parsed models/model libs from config or error message.
   */
  TVM_DLL static Result<std::vector<std::pair<std::string, std::string>>>
  GetModelsAndModelLibsFromJSONString(const std::string& json_str);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(EngineConfig, ObjectRef, EngineConfigNode);
};

/*! \brief A subset of engine config that is inferrable. */
struct InferrableEngineConfig {
  std::optional<int64_t> max_num_sequence;
  std::optional<int64_t> max_total_sequence_length;
  std::optional<int64_t> max_single_sequence_length;
  std::optional<int64_t> prefill_chunk_size;
  std::optional<int64_t> max_history_size;
  std::optional<KVStateKind> kv_state_kind;

  /*! \brief Infer the config for KV cache from a given initial config. */
  TVM_DLL static Result<InferrableEngineConfig> InferForKVCache(
      EngineMode mode, Device device, double gpu_memory_utilization,
      const std::vector<picojson::object>& model_configs,
      const std::vector<ModelMetadata>& model_metadata, InferrableEngineConfig init_config,
      bool verbose);
  /*! \brief Infer the config for RNN state from a given initial config. */
  TVM_DLL static Result<InferrableEngineConfig> InferForRNNState(
      EngineMode mode, Device device, double gpu_memory_utilization,
      const std::vector<picojson::object>& model_configs,
      const std::vector<ModelMetadata>& model_metadata, InferrableEngineConfig init_config,
      bool verbose);
};

/****************** Config utils ******************/

/*! \brief Check if the models use KV cache or RNN state. */
Result<bool> ModelsUseKVCache(const std::vector<picojson::object>& model_configs);

inline std::string EngineModeToString(EngineMode mode) {
  return mode == EngineMode::kLocal         ? "local"
         : mode == EngineMode::kInteractive ? "interactive"
                                            : "server";
}

inline EngineMode EngineModeFromString(const std::string& mode) {
  if (mode == "local") {
    return EngineMode::kLocal;
  } else if (mode == "interactive") {
    return EngineMode::kInteractive;
  } else if (mode == "server") {
    return EngineMode::kServer;
  } else {
    LOG(FATAL) << "Invalid engine mode string: " << mode;
  }
}

inline std::string SpeculativeModeToString(SpeculativeMode speculative_mode) {
  return speculative_mode == SpeculativeMode::kDisable      ? "disable"
         : speculative_mode == SpeculativeMode::kSmallDraft ? "small_draft"
                                                            : "eagle";
}

inline SpeculativeMode SpeculativeModeFromString(const std::string& speculative_mode) {
  if (speculative_mode == "disable") {
    return SpeculativeMode::kDisable;
  } else if (speculative_mode == "small_draft") {
    return SpeculativeMode::kSmallDraft;
  } else if (speculative_mode == "eagle") {
    return SpeculativeMode::kEagle;
  } else {
    LOG(FATAL) << "Invalid speculative mode string: " << speculative_mode;
  }
}

inline std::string KVStateKindToString(KVStateKind kv_state_kind) {
  return kv_state_kind == KVStateKind::kKVCache ? "kv_cache" : "rnn_State";
}

inline KVStateKind KVStateKindFromString(const std::string& kv_state_kind) {
  if (kv_state_kind == "kv_cache") {
    return KVStateKind::kKVCache;
  } else if (kv_state_kind == "rnn_state") {
    return KVStateKind::kRNNState;
  } else {
    LOG(FATAL) << "Invalid kv state kind string: " << kv_state_kind;
  }
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_CONFIG_H_
