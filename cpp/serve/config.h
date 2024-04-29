/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/config.h
 */
#ifndef MLC_LLM_SERVE_CONFIG_H_
#define MLC_LLM_SERVE_CONFIG_H_

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/object.h>

#include <optional>

#include "../json_ffi/config.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm;
using namespace tvm::runtime;
using namespace mlc::llm::json_ffi;

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
  explicit GenerationConfig(String config_json_str);

  /*!
   * \brief Create a generation config from a ChatCompletionRequest.
   * If the request does not contain a generation config, the model-defined
   * generation config will be used.
   */
  static Optional<GenerationConfig> Create(
      const std::string& json_str, std::string* err, const Conversation& conv_template,
      const ModelDefinedGenerationConfig& model_defined_gen_config);

  TVM_DEFINE_OBJECT_REF_METHODS(GenerationConfig, ObjectRef, GenerationConfigNode);
};

/****************** Engine config ******************/

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
enum KVStateKind {
  kAttention = 0,
  kRNNState = 1,
};

/*! \brief The configuration of engine execution config. */
class EngineConfigNode : public Object {
 public:
  /*************** Models ***************/

  /*! \brief The path to the model directory. */
  String model;
  /*! \brief The path to the model library. */
  String model_lib_path;
  /*! \brief The path to the additional models' directories. */
  Array<String> additional_models;
  /*! \brief The path to the additional models' libraries. */
  Array<String> additional_model_lib_paths;

  /*************** KV cache config and engine capacities ***************/

  /*! \brief The number of consecutive tokens handled in each page in paged KV cache. */
  int kv_cache_page_size;
  /*!
   * \brief The maximum number of sequences that are allowed to be
   * processed by the KV cache at any time.
   */
  int max_num_sequence;
  /*! \brief The maximum length allowed for a single sequence in the engine. */
  int max_total_sequence_length;
  /*!
   * \brief The maximum total number of tokens whose KV data are allowed
   * to exist in the KV cache at any time.
   */
  int max_single_sequence_length;
  /*! \brief The maximum total sequence length in a prefill. */
  int prefill_chunk_size;
  /*! \brief The maximum history size for RNN state. KV cache does not need this. */
  int max_history_size;
  /*! \brief The kind of cache. Whether it's KV cache or RNN state. */
  KVStateKind kv_state_kind;

  /*************** Speculative decoding ***************/

  /*! \brief The speculative mode. */
  SpeculativeMode speculative_mode;
  /*! \brief The number of tokens to generate in speculative proposal (draft). */
  int spec_draft_length = 4;

  String AsJSONString() const;

  static constexpr const char* _type_key = "mlc.serve.EngineConfig";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(EngineConfigNode, Object);
};

class EngineConfig : public ObjectRef {
 public:
  explicit EngineConfig(String model, String model_lib_path, Array<String> additional_models,
                        Array<String> additional_model_lib_paths, int kv_cache_page_size,
                        int max_num_sequence, int max_total_sequence_length,
                        int max_single_sequence_length, int prefill_chunk_size,
                        int max_history_size, KVStateKind kv_state_kind,
                        SpeculativeMode speculative_mode, int spec_draft_length);

  /*! \brief Create EngineConfig from JSON string. */
  static EngineConfig FromJSONString(const std::string& json_str);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(EngineConfig, ObjectRef, EngineConfigNode);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_CONFIG_H_
