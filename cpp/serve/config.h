/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/config.h
 */
#ifndef MLC_LLM_SERVE_CONFIG_H_
#define MLC_LLM_SERVE_CONFIG_H_

#include <picojson.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/int_tuple.h>
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
  Optional<String> schema = std::nullopt;
  /*!
   * \brief Create debug config from JSON.
   * \param config_json The json string for generation config
   * \returns The converted result.
   */
  static Result<ResponseFormat> FromJSON(const picojson::object& config_json);

  /**
   * \return serialized json value of the config.
   */
  picojson::object AsJSON() const;
};

enum class SpecialRequestKind : int {
  kNone = 0,
  kQueryEngineMetrics = 1,
};

enum class DisaggRequestKind : int {
  kNone = 0,
  kPrepareReceive = 1,
  kRemoteSend = 2,
  kStartGeneration = 3,
};

/*! \brief Controls the behavior of inference with grammar constraint. */
enum class GrammarExecutionMode : int {
  /*! \brief If grammar is provided for a request, use the grammar to constrain the output token. */
  kConstraint = 0,
  /*! \brief If grammar is provided for a request, not only constrain the output, but also use the
   * jump-forward decoding to predict the next tokens. This is the default option. */
  kJumpForward = 1,
};

/*! \brief The config for disaggregation requests. */
class DisaggConfig {
 public:
  DisaggRequestKind kind = DisaggRequestKind::kNone;
  std::vector<IntTuple> kv_append_metadata;
  // "kv_window_begin" and "kv_window_end" denote the KV interval of interests.
  // "kv_window_end" supports Python style negative indexing.
  // The concrete meaning varies for different special request kind:
  // - For "prepare_receive", the begin is always 0, and "[0:end]" denotes
  // the KV range to prefill on a prefill instance.
  // - For "remote_send", "[begin:end]" means the KV range to compute prefill
  // and send to the decode instance.
  // - For "start_generation", the end is always nullopt, and "[begin:]" denotes
  // the KV range to prefill locally on the decode instance.
  std::optional<int> kv_window_begin = std::nullopt;
  std::optional<int> kv_window_end = std::nullopt;
  std::optional<int> dst_group_offset = std::nullopt;

  static Result<DisaggConfig> FromJSON(const picojson::object& config_json);
  picojson::object AsJSON() const;
};

/*! \brief The debug configuration of a request. */
class DebugConfig {
 public:
  bool ignore_eos = false;
  bool pinned_system_prompt = false;
  SpecialRequestKind special_request = SpecialRequestKind::kNone;
  /*! \brief The grammar execution mode. */
  GrammarExecutionMode grammar_execution_mode = GrammarExecutionMode::kJumpForward;
  DisaggConfig disagg_config;

  /*!
   * \brief Create debug config from JSON.
   * \param config_json The json string for generation config
   * \returns The converted result.
   */
  static Result<DebugConfig> FromJSON(const picojson::object& config_json);

  /**
   * \return serialized json value of the config.
   */
  picojson::object AsJSON() const;
};

/*! \brief The generation configuration of a request. */
class GenerationConfigNode : public Object {
 public:
  int n = 1;
  double temperature = 1.0;
  double top_p = 1.0;
  double frequency_penalty = 0.0;
  double presence_penalty = 0.0;
  double repetition_penalty = 1.0;
  bool logprobs = false;
  int top_logprobs = 0;
  std::vector<std::pair<int, float>> logit_bias;
  int seed;
  // -1 means infinite
  int max_tokens = -1;
  Array<String> stop_strs;
  std::vector<int> stop_token_ids;

  ResponseFormat response_format;
  DebugConfig debug_config;

  picojson::object AsJSON() const;

  static constexpr const char* _type_key = "mlc.serve.GenerationConfig";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(GenerationConfigNode, Object);
};

class GenerationConfig : public ObjectRef {
 public:
  /*!
   * \brief Run validation of generation config and ensure values are in bound.
   * \return The validtaed Generation config or error.
   */
  static Result<GenerationConfig> Validate(GenerationConfig cfg);

  /*!
   * \brief Create generation config from JSON.
   * \param config_json The json string for generation config
   * \param default_config The default config
   */
  static Result<GenerationConfig> FromJSON(const picojson::object& config_json,
                                           const GenerationConfig& default_config);

  /*! \brief Get the default generation config from the model config. */
  static GenerationConfig GetDefaultFromModelConfig(const picojson::object& json);

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

/*! \brief The prefix cache mode. */
enum class PrefixCacheMode : int {
  /*! \brief Disable prefix cache. */
  kDisable = 0,
  /*! \brief The paged radix tree based prefix cache mode. */
  kRadix = 1,
};

/*! \brief The speculative mode. */
enum class SpeculativeMode : int {
  /*! \brief Disable speculative decoding. */
  kDisable = 0,
  /*! \brief The normal speculative decoding (small draft) mode. */
  kSmallDraft = 1,
  /*! \brief The eagle-style speculative decoding. */
  kEagle = 2,
  /*! \brief The Medusa-style speculative decoding. */
  kMedusa = 3,
};

/*! \brief The prefill mode. */
enum class PrefillMode : int {
  /*! \brief Only chunked prefill is enabled. */
  kChunked = 0,
  /*!
   * \brief The hybrid prefill or split-fuse prefill is enabled, some decode steps will be fused
   * to prefill
   */
  kHybrid = 1,
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
  int64_t max_total_sequence_length = 4096;
  /*!
   * \brief The maximum total number of tokens whose KV data are allowed
   * to exist in the KV cache at any time.
   */
  int64_t max_single_sequence_length = 4096;
  /*! \brief The maximum total sequence length in a prefill. */
  int64_t prefill_chunk_size = 1024;
  /*! \brief The maximum history size for RNN state. KV cache does not need this. */
  int max_history_size = 0;

  /*************** Prefix cache ***************/

  /*! \brief The prefix cache mode. */
  PrefixCacheMode prefix_cache_mode = PrefixCacheMode::kRadix;
  /*! \brief The maximum number of recycling sequences in prefix cache, default as max_num_sequence.
   * And set 0 to disable prefix cache, set -1 to have infinite capacity prefix cache. */
  int prefix_cache_max_num_recycling_seqs = -1;

  /*************** Speculative decoding ***************/

  /*! \brief The speculative mode. */
  SpeculativeMode speculative_mode = SpeculativeMode::kDisable;
  /*!
   * \brief The number of tokens to generate in speculative proposal (draft).
   * Being 0 means to enable adaptive speculative mode, where the draft length
   * will be automatically adjusted based on engine state.
   */
  int spec_draft_length = 0;
  /*! \brief The number of tokens to generate in speculative tree decoding */
  int spec_tree_width = 1;

  /*************** Prefill mode ***************/

  /*! \brief The prefill mode. */
  PrefillMode prefill_mode = PrefillMode::kHybrid;

  /*************** Debug ***************/
  bool verbose = false;

  String AsJSONString() const;

  static constexpr const char* _type_key = "mlc.serve.EngineConfig";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(EngineConfigNode, Object);
};

class EngineConfig : public ObjectRef {
 public:
  /*! \brief Create EngineConfig from JSON object and inferred config. */
  static EngineConfig FromJSONAndInferredConfig(const picojson::object& json,
                                                const InferrableEngineConfig& inferred_config);

  /*!
   * \brief Get all the models and model libs from the JSON string for engine initialization.
   * \return The parsed models/model libs from config or error message.
   */
  static Result<std::vector<std::pair<std::string, std::string>>>
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

  /*! \brief Infer the config for KV cache from a given initial config. */
  static Result<InferrableEngineConfig> InferForKVCache(
      EngineMode mode, Device device, double gpu_memory_utilization,
      const std::vector<picojson::object>& model_configs,
      const std::vector<ModelMetadata>& model_metadata, InferrableEngineConfig init_config,
      bool verbose);
  /*! \brief Infer the config for RNN state from a given initial config. */
  static Result<InferrableEngineConfig> InferForRNNState(
      EngineMode mode, Device device, double gpu_memory_utilization,
      const std::vector<picojson::object>& model_configs,
      const std::vector<ModelMetadata>& model_metadata, InferrableEngineConfig init_config,
      bool verbose);
};

/****************** Config utils ******************/

/*! \brief Check if the models use KV cache or RNN state. */
Result<bool> ModelsUseKVCache(const std::vector<picojson::object>& model_configs);

inline std::string EngineModeToString(EngineMode mode) {
  if (mode == EngineMode::kLocal) {
    return "local";
  } else if (mode == EngineMode::kInteractive) {
    return "interactive";
  } else if (mode == EngineMode::kServer) {
    return "server";
  } else {
    LOG(FATAL) << "Invalid engine mode: " << static_cast<int>(mode);
    throw;
  }
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
    throw;
  }
}

inline std::string PrefixCacheModeToString(PrefixCacheMode prefix_cache_mode) {
  if (prefix_cache_mode == PrefixCacheMode::kDisable) {
    return "disable";
  } else if (prefix_cache_mode == PrefixCacheMode::kRadix) {
    return "radix";
  } else {
    LOG(FATAL) << "Invalid prefix cache mode: " << static_cast<int>(prefix_cache_mode);
  }
}

inline PrefixCacheMode PrefixCacheModeFromString(const std::string& prefix_cache_mode) {
  if (prefix_cache_mode == "disable") {
    return PrefixCacheMode::kDisable;
  } else if (prefix_cache_mode == "radix") {
    return PrefixCacheMode::kRadix;
  } else {
    LOG(FATAL) << "Invalid prefix cache mode string: " << prefix_cache_mode;
    throw;
  }
}

inline std::string SpeculativeModeToString(SpeculativeMode speculative_mode) {
  if (speculative_mode == SpeculativeMode::kDisable) {
    return "disable";
  } else if (speculative_mode == SpeculativeMode::kSmallDraft) {
    return "small_draft";
  } else if (speculative_mode == SpeculativeMode::kEagle) {
    return "eagle";
  } else if (speculative_mode == SpeculativeMode::kMedusa) {
    return "medusa";
  } else {
    LOG(FATAL) << "Invalid speculative mode: " << static_cast<int>(speculative_mode);
  }
}

inline SpeculativeMode SpeculativeModeFromString(const std::string& speculative_mode) {
  if (speculative_mode == "disable") {
    return SpeculativeMode::kDisable;
  } else if (speculative_mode == "small_draft") {
    return SpeculativeMode::kSmallDraft;
  } else if (speculative_mode == "eagle") {
    return SpeculativeMode::kEagle;
  } else if (speculative_mode == "medusa") {
    return SpeculativeMode::kMedusa;
  } else {
    LOG(FATAL) << "Invalid speculative mode string: " << speculative_mode;
    throw;
  }
}

inline std::string PrefillModeToString(PrefillMode prefill_mode) {
  if (prefill_mode == PrefillMode::kChunked) {
    return "chunked";
  } else if (prefill_mode == PrefillMode::kHybrid) {
    return "hybrid";
  } else {
    LOG(FATAL) << "Invalid prefill mode: " << static_cast<int>(prefill_mode);
  }
}

inline PrefillMode PrefillModeFromString(const std::string& prefill_mode) {
  if (prefill_mode == "chunked") {
    return PrefillMode::kChunked;
  } else if (prefill_mode == "hybrid") {
    return PrefillMode::kHybrid;
  } else {
    LOG(FATAL) << "Invalid prefill mode string: " << prefill_mode;
    throw;
  }
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_CONFIG_H_
